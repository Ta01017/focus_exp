"""metadata.json inference through FusionDiff's original conditional DDPM."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent.parent))
from metadata_dataset import base_record, bool01, load_metadata, prepare_item, restore_a_size, save_inputs, write_run_files
from diffusion_sampling import validated_sampling_steps


def main():
    p=argparse.ArgumentParser(); p.add_argument('--metadata',required=True); p.add_argument('--output-dir',required=True)
    p.add_argument('--checkpoint'); p.add_argument('--device',default='cuda:0'); p.add_argument('--start-index',type=int,default=0)
    p.add_argument('--max-samples',type=int,default=-1); p.add_argument('--overwrite',type=bool01,default=False); p.add_argument('--save-inputs',type=bool01,default=False)
    p.add_argument('--size-policy',choices=('error','resize_b_to_a','center_crop_common'),default='error'); p.add_argument('--sampling-steps',type=int); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--config',default=str(ROOT/'config.json')); args=p.parse_args()
    with open(args.config,encoding='utf-8') as f: config=json.load(f)
    trained_steps=validated_sampling_steps(config,args.sampling_steps)
    if not args.checkpoint: raise FileNotFoundError('FusionDiff has no checkpoint in this repository; --checkpoint with an official trained weight is required. Random weights are forbidden.')
    ckpt=Path(args.checkpoint).expanduser().resolve()
    if not ckpt.is_file(): raise FileNotFoundError('FusionDiff checkpoint not found: '+str(ckpt))
    import Diffusion as diffusion_module
    from Diffusion import GaussianDiffusion
    from Condition_Noise_Predictor.UNet import NoisePred
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; pass --device cpu')
    diffusion_module.device=device
    c=config['Condition_Noise_Predictor']; u=c['UNet']
    model=NoisePred(u['in_channels'],u['out_channels'],u['model_channels'],u['num_res_blocks'],u['dropout'],u['time_embed_dim_mult'],u['down_sample_mult'])
    state=torch.load(str(ckpt),map_location=device); model.load_state_dict(state.get('state_dict',state),strict=True); model.to(device).eval()
    diffusion=GaussianDiffusion(trained_steps,config['diffusion_model']['beta_schedule_type'])
    meta,items=load_metadata(args.metadata,args.start_index,args.max_samples); out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True); records=[]
    for index,item in items:
        rec=base_record(index,item,meta,out); started=time.perf_counter()
        try:
            target=Path(rec['prediction'])
            sample=prepare_item(item,index,meta,args.size_policy); rec['original_width'],rec['original_height']=sample['original_size']
            if target.exists() and not args.overwrite: rec.update(success=True,error='skipped_existing'); records.append(rec); continue
            def tensor(im): return torch.from_numpy(np.asarray(im,dtype=np.float32).transpose(2,0,1)/127.5-1).unsqueeze(0).to(device)
            a,b=tensor(sample['a']),tensor(sample['b']); h,w=a.shape[-2:]; ph,pw=(-h)%8,(-w)%8
            if ph or pw:
                mode='reflect' if h>ph and w>pw else 'replicate'; a=F.pad(a,(0,pw,0,ph),mode=mode); b=F.pad(b,(0,pw,0,ph),mode=mode)
            torch.manual_seed(args.seed+index)
            if device.type=='cuda': torch.cuda.manual_seed_all(args.seed+index)
            with torch.inference_mode(): pred=diffusion.p_sample_loop(model,a,b,c['concat_type'],config['diffusion_model']['add_noise'],[1,1,0,1])[0,:,:h,:w]
            arr=((pred.clamp(-1,1).cpu().numpy().transpose(1,2,0)+1)*127.5).round().astype(np.uint8)
            restore_a_size(Image.fromarray(arr),sample).save(target,'PNG')
            if args.save_inputs: save_inputs(sample,out)
            rec['success']=True
        except Exception as exc: rec['error']=f'{type(exc).__name__}: {exc}'
        rec['runtime_seconds']=round(time.perf_counter()-started,6); records.append(rec)
    write_run_files(out,records,vars(args)|{'metadata':str(meta),'checkpoint_loaded':str(ckpt),'actual_sampling_steps':trained_steps})
    if not any(r['success'] for r in records): raise SystemExit(2)
if __name__=='__main__': main()
