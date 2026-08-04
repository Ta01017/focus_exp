"""metadata.json inference preserving ReDiffuse's rotation-equivariant DDPM."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT.parent))
from metadata_dataset import base_record,bool01,load_metadata,prepare_item,restore_a_size,save_inputs,write_run_files
from diffusion_sampling import validated_sampling_steps
from runtime_check import require_official_b_conv


def main():
    p=argparse.ArgumentParser(); p.add_argument('--metadata',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--checkpoint')
    p.add_argument('--device',default='cuda:0'); p.add_argument('--start-index',type=int,default=0); p.add_argument('--max-samples',type=int,default=-1)
    p.add_argument('--overwrite',type=bool01,default=False); p.add_argument('--save-inputs',type=bool01,default=False); p.add_argument('--size-policy',choices=('error','resize_b_to_a','center_crop_common'),default='error')
    p.add_argument('--sampling-steps',type=int); p.add_argument('--seed',type=int,default=0); p.add_argument('--config',default=str(ROOT/'config.json')); args=p.parse_args()
    require_official_b_conv()
    with open(args.config,encoding='utf-8') as f: config=json.load(f)
    trained_steps=validated_sampling_steps(config,args.sampling_steps)
    ckpt=Path(args.checkpoint).expanduser().resolve() if args.checkpoint else ROOT/'weights/model.pt'
    if not ckpt.is_file(): raise FileNotFoundError('Author ReDiffuse model.pt required: '+str(ckpt))
    import Diffusion as diffusion_module
    from Diffusion import GaussianDiffusion
    from Condition_Noise_Predictor.Rot_E_UNet import NoisePred
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; pass --device cpu')
    diffusion_module.device=device
    c=config['Condition_Noise_Predictor']; u=c['UNet']; model=NoisePred(u['in_channels'],u['out_channels'],u['model_channels'],u['num_res_blocks'],u['dropout'],u['time_embed_dim_mult'],u['down_sample_mult'])
    state=torch.load(str(ckpt),map_location=device); model.load_state_dict(state.get('state_dict',state),strict=True); model.to(device).eval()
    diffusion=GaussianDiffusion(trained_steps,config['diffusion_model']['beta_schedule_type']); meta,items=load_metadata(args.metadata,args.start_index,args.max_samples)
    out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True); records=[]
    for index,item in items:
        rec=base_record(index,item,meta,out); started=time.perf_counter()
        try:
            target=Path(rec['prediction'])
            sample=prepare_item(item,index,meta,args.size_policy); rec['original_width'],rec['original_height']=sample['original_size']
            if target.exists() and not args.overwrite: rec.update(success=True,error='skipped_existing'); records.append(rec); continue
            a_rgb=np.asarray(sample['a']); b_rgb=np.asarray(sample['b'])
            # Author validation path diffuses luminance and fuses chroma by distance from neutral 128.
            def ycc(x): return np.asarray(Image.fromarray(x).convert('YCbCr'))
            ay,by=ycc(a_rgb),ycc(b_rgb)
            def tensor(y): return torch.from_numpy(y.astype(np.float32)/127.5-1).unsqueeze(0).unsqueeze(0).to(device)
            a,b=tensor(ay[:,:,0]),tensor(by[:,:,0]); h,w=a.shape[-2:]; ph,pw=(-h)%8,(-w)%8
            if ph or pw:
                mode='reflect' if h>ph and w>pw else 'replicate'; a=F.pad(a,(0,pw,0,ph),mode=mode); b=F.pad(b,(0,pw,0,ph),mode=mode)
            torch.manual_seed(args.seed+index)
            if device.type=='cuda': torch.cuda.manual_seed_all(args.seed+index)
            with torch.inference_mode(): pred=diffusion.p_sample_loop(model,a,b,c['concat_type'],config['diffusion_model']['add_noise'],[1,1,0,1])[0,0,:h,:w]
            y=((pred.clamp(-1,1).cpu().numpy()+1)*127.5)
            da=np.abs(ay[:,:,1:].astype(np.float32)-128); db=np.abs(by[:,:,1:].astype(np.float32)-128); chroma=(ay[:,:,1:]*da+by[:,:,1:]*db)/(da+db+1e-8)
            result=np.concatenate((y[:,:,None],chroma),axis=2).clip(0,255).astype(np.uint8)
            restore_a_size(Image.fromarray(result,'YCbCr').convert('RGB'),sample).save(target,'PNG')
            if args.save_inputs: save_inputs(sample,out)
            rec['success']=True
        except Exception as exc: rec['error']=f'{type(exc).__name__}: {exc}'
        rec['runtime_seconds']=round(time.perf_counter()-started,6); records.append(rec)
    write_run_files(out,records,vars(args)|{'metadata':str(meta),'checkpoint_loaded':str(ckpt.resolve()),'actual_sampling_steps':trained_steps})
    if not any(r['success'] for r in records): raise SystemExit(2)
if __name__=='__main__': main()
