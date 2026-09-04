"""Per-sample zero-shot optimization for metadata.json image pairs."""
import argparse, sys, time
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.optim.lr_scheduler import MultiStepLR

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT.parent))
from metadata_dataset import base_record,bool01,load_metadata,prepare_item,restore_a_size,save_inputs,write_run_files
from network.loss import Loss
from network.skip import skip
from util.common_utils import blur_2th,get_noise,torch_to_np


def optimize(a_image,b_image,config,device,iterations):
    # The official implementation optimizes luminance and carries A's chroma.
    a=np.asarray(a_image.convert('YCbCr')); b=np.asarray(b_image.convert('YCbCr'))
    y1=torch.from_numpy(a[:,:,0].astype(np.float32)/255).unsqueeze(0).unsqueeze(0).to(device)
    y2=torch.from_numpy(b[:,:,0].astype(np.float32)/255).unsqueeze(0).unsqueeze(0).to(device); ys=[y1,y2]
    size=list(y1.shape[-2:]); config=dict(config); config['img_size']=size; config['num_iter']=iterations
    loss_fn=Loss(config,device); net_inputx=torch.cat(ys,dim=1).to(device)
    netx=skip(config['input_channelx'],1,channels=[128]*5,channels_skip=16,upsample_mode='bilinear',attention_mode=config['attention'],need_bias=False,pad=config['pad'],act_fun='LeakyReLU',scales=config['scales']).to(device)
    inputs=[]; nets=[]
    for _ in range(2):
        inputs.append(get_noise(spatial_size=size,input_channel=config['input_channelm'],input_type=config['input_typem']).to(device))
        nets.append(skip(config['input_channelm'],1,channels=[128]*3,channels_skip=16,upsample_mode='bilinear',attention_mode=config['attention'],need_bias=False,pad=config['pad'],act_fun='LeakyReLU',scales=config['scales']).to(device))
    params=[{'params':netx.parameters()},{'params':net_inputx}]
    for net,inp in zip(nets,inputs): params += [{'params':net.parameters()},{'params':inp}]
    optimizer=torch.optim.Adam(params,lr=config['lr']); scheduler=MultiStepLR(optimizer,milestones=[200,400,800],gamma=.5)
    savedx=net_inputx.detach().clone(); noisex=savedx.clone(); saved=[x.detach().clone() for x in inputs]; noises=[x.clone() for x in saved]
    maximum=torch.zeros_like(y1)
    for y in ys: maximum=torch.max(torch.abs(blur_2th(y)),maximum)
    scores=[1-torch.sign(maximum-torch.abs(blur_2th(y))) for y in ys]; out_x=None
    for step in range(iterations):
        optimizer.zero_grad()
        nx=savedx+(noisex.normal_()*config['reg_noise_std']) if config['reg_noise_std']>0 else savedx
        current=[s+(n.normal_()*config['reg_noise_std']) if config['reg_noise_std']>0 else s for s,n in zip(saved,noises)]
        out_x=[F.interpolate(x,size=size,mode='bilinear',align_corners=False) for x in netx(nx)]
        out_m=[]
        for net,inp in zip(nets,current): out_m.append([F.interpolate(x,size=size,mode='bilinear',align_corners=False) for x in net(inp)])
        losses=loss_fn(out_x,out_m,ys,scores,step); losses['total_loss'].backward(); optimizer.step(); scheduler.step()
    y=np.uint8(np.clip(torch_to_np(out_x[0]).squeeze()*255,0,255)); merged=np.dstack((y,a[:,:,1],a[:,:,2]))
    return Image.fromarray(merged,'YCbCr').convert('RGB')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--metadata',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--checkpoint')
    p.add_argument('--device',default='cuda:0'); p.add_argument('--start-index',type=int,default=0); p.add_argument('--max-samples',type=int,default=-1)
    p.add_argument('--overwrite',type=bool01,default=False); p.add_argument('--save-inputs',type=bool01,default=False); p.add_argument('--size-policy',choices=('error','resize_b_to_a','center_crop_common'),default='error')
    p.add_argument('--iterations',type=int,default=1300); p.add_argument('--max-optimize-side',type=int,default=1024); p.add_argument('--seed',type=int,default=17); p.add_argument('--config',default=str(ROOT/'config/test.yaml')); args=p.parse_args()
    if args.checkpoint: raise ValueError('ZMFF is zero-shot and does not accept a checkpoint')
    if args.iterations<1: raise ValueError('--iterations must be positive')
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; pass --device cpu')
    with open(args.config,encoding='utf-8') as f: config=yaml.safe_load(f)
    meta,items=load_metadata(args.metadata,args.start_index,args.max_samples); out=Path(args.output_dir).resolve(); out.mkdir(parents=True,exist_ok=True); records=[]
    for index,item in items:
        rec=base_record(index,item,meta,out); rec['actual_iterations']=0; started=time.perf_counter()
        try:
            target=Path(rec['prediction'])
            sample=prepare_item(item,index,meta,args.size_policy); rec['original_width'],rec['original_height']=sample['original_size']
            if target.exists() and not args.overwrite: rec.update(success=True,error='skipped_existing'); records.append(rec); continue
            # Critical isolation: seed and construct every network/optimizer inside this sample iteration.
            torch.manual_seed(args.seed+index); np.random.seed(args.seed+index)
            if device.type=='cuda': torch.cuda.manual_seed_all(args.seed+index)
            optimize_a, optimize_b = sample['a'], sample['b']
            if args.max_optimize_side > 0 and max(optimize_a.size) > args.max_optimize_side:
                scale = args.max_optimize_side / max(optimize_a.size)
                optimize_size = (max(1, round(optimize_a.width * scale)),
                                 max(1, round(optimize_a.height * scale)))
                optimize_a = optimize_a.resize(optimize_size, Image.Resampling.LANCZOS)
                optimize_b = optimize_b.resize(optimize_size, Image.Resampling.LANCZOS)
                print(f"[ZMFF] sample={sample['sample_id']} optimize_size={optimize_size} output_size={sample['a'].size}")
            result=optimize(optimize_a,optimize_b,config,device,args.iterations); rec['actual_iterations']=args.iterations
            if result.size != sample['a'].size:
                result = result.resize(sample['a'].size, Image.Resampling.BICUBIC)
            restore_a_size(result,sample).save(target,'PNG')
            if args.save_inputs: save_inputs(sample,out)
            rec['success']=True
        except Exception as exc:
            rec['error']=f'{type(exc).__name__}: {exc}'
            print(f"[ZMFF ERROR] index={index} sample={rec['sample_id']} {rec['error']}", file=sys.stderr)
        rec['runtime_seconds']=round(time.perf_counter()-started,6); records.append(rec)
    write_run_files(out,records,vars(args)|{'metadata':str(meta),'checkpoint_loaded':None,'zero_shot_reinitialize_per_sample':True})
    if not any(r['success'] for r in records): raise SystemExit(2)
if __name__=='__main__': main()
