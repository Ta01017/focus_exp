import time
import torch
from tqdm import tqdm
import json
import argparse
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from dataset import MFI_Dataset, MetadataMFI_Dataset
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from metadata_training import warn_split_overlap
from diffusion_checkpoint import (rgb_contract, load_model_init, resume_training,
                                  save_checkpoint)
from Diffusion import GaussianDiffusion
from Condition_Noise_Predictor.UNet import NoisePred
from utils import tensorboard_writer, logger, save_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(config_path, args=None):
    timestr = time.strftime('%Y%m%d_%H%M%S')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    contract = rgb_contract("FusionDiff", config["diffusion_model"]["T"])
    Path("data_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    if args:
        Path("train_manifest.json").write_text(
            json.dumps({"arguments": vars(args), "config": str(Path(config_path).resolve())}, indent=2),
            encoding="utf-8")

    # train dataset
    train_datasePath = config["dataset"]["train"]["path"]
    train_phase = config["dataset"]["train"]["phase"]
    train_batch_size = config["dataset"]["train"]["batch_size"]
    train_use_dataTransform = config["dataset"]["train"]["use_dataTransform"]
    train_resize = config["dataset"]["train"]["resize"]
    train_imgSize = config["dataset"]["train"]["imgSize"]
    train_shuffle = config["dataset"]["train"]["shuffle"]
    train_drop_last = config["dataset"]["train"]["drop_last"]
    if args and args.dataset_format == "metadata":
        if not args.train_metadata or not args.val_metadata:
            raise ValueError("metadata mode requires --train-metadata and --val-metadata")
        train_dataset = MetadataMFI_Dataset(args.train_metadata, "train", train_resize, train_imgSize,
                                            args.seed, args.start_index, args.max_samples)
        val_cfg = config["dataset"].get("valid", config["dataset"]["train"])
        val_dataset = MetadataMFI_Dataset(args.val_metadata, "valid", val_cfg.get("resize", train_resize),
                                          val_cfg.get("imgSize", train_imgSize), args.seed, 0, args.max_samples)
        warn_split_overlap(train_dataset, val_dataset, args.fail_on_split_overlap)
        print("[FusionDiff] dataset_format=metadata\n[FusionDiff] color_space=RGB\n"
              "[FusionDiff] input_order=A,B\n[FusionDiff] target=image")
    else:
        train_dataset = MFI_Dataset(train_datasePath, phase=train_phase, use_dataTransform=train_use_dataTransform,
                                    resize=train_resize, imgSzie=train_imgSize)
        val_dataset = None
    # You can modify the "num_workers" parameter for different GPU devices
    effective_drop_last = train_drop_last and len(train_dataset) >= train_batch_size
    if train_drop_last and not effective_drop_last:
        print("WARNING: dataset is smaller than batch_size; disabling drop_last for smoke")
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=train_shuffle,
                                  drop_last=effective_drop_last,pin_memory=True,
                                  num_workers=args.num_workers if args else 4)
    val_dataloader = (DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
                      if val_dataset is not None else None)

    # Condition Noise Predictor
    in_channels = config["Condition_Noise_Predictor"]["UNet"]["in_channels"]
    out_channels = config["Condition_Noise_Predictor"]["UNet"]["out_channels"]
    model_channels = config["Condition_Noise_Predictor"]["UNet"]["model_channels"]
    num_res_blocks = config["Condition_Noise_Predictor"]["UNet"]["num_res_blocks"]
    dropout = config["Condition_Noise_Predictor"]["UNet"]["dropout"]
    time_embed_dim_mult = config["Condition_Noise_Predictor"]["UNet"]["time_embed_dim_mult"]
    down_sample_mult = config["Condition_Noise_Predictor"]["UNet"]["down_sample_mult"]
    model = NoisePred(in_channels, out_channels, model_channels, num_res_blocks, dropout, time_embed_dim_mult,
                      down_sample_mult)

    # whether to use the pre-training model
    use_preTrain_model = config["Condition_Noise_Predictor"]["use_preTrain_model"]
    if use_preTrain_model:
        preTrain_Model_path = config["Condition_Noise_Predictor"]["preTrain_Model_path"]
        model.load_state_dict(torch.load(preTrain_Model_path, map_location=device))
        print(f"using pre-trained model：{preTrain_Model_path}")
    model = model.to(device)

    # channel splicing mode
    concat_type = config["Condition_Noise_Predictor"]["concat_type"]
    assert concat_type in ["ABX", "AXB", "XAB"], "Check that the 'concat_type' parameter is correct"

    # optimizer
    init_lr = config["optimizer"]["init_lr"]
    use_lr_scheduler = config["optimizer"]["use_lr_scheduler"]
    StepLR_size = config["optimizer"]["StepLR_size"]
    StepLR_gamma = config["optimizer"]["StepLR_gamma"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr)
    learningRate_scheduler = (lr_scheduler.StepLR(optimizer, step_size=StepLR_size, gamma=StepLR_gamma)
                              if use_lr_scheduler else None)

    # diffusion model
    T = config["diffusion_model"]["T"]
    beta_schedule_type = config["diffusion_model"]["beta_schedule_type"]
    loss_scale = config["diffusion_model"]["loss_scale"]
    diffusion = GaussianDiffusion(T, beta_schedule_type)

    # log
    writer = tensorboard_writer(timestr)
    log = logger(timestr)
    print(f"time: {timestr}")
    log.write(f"time: {timestr} \n")
    print(f"using {len(train_dataset)} images for train")
    log.write(f"using {len(train_dataset)} images for train  \n\n")
    log.write(f"config:  \n")
    log.write(json.dumps(config, ensure_ascii=False, indent=4))
    if use_lr_scheduler:
        log.write(
            f"\n learningRate_scheduler = lr_scheduler.StepLR(optimizer, step_size={StepLR_size}, gamma={StepLR_gamma})  \n\n")

    # hyper-parameter
    epochs = config["hyperParameter"]["epochs"]
    start_epoch = config["hyperParameter"]["start_epoch"]
    loss_step = config["hyperParameter"]["loss_step"]
    save_model_epoch_step = config["hyperParameter"]["save_model_epoch_step"]
    train_step_sum = len(train_dataloader)
    num_train_step = 0
    if args and args.init_checkpoint:
        load_model_init(args.init_checkpoint, model, "FusionDiff", bool(args.allow_legacy_checkpoint), device)
        print(f"initialized model only: {Path(args.init_checkpoint).resolve()}")
    if args and args.resume:
        start_epoch, num_train_step, _ = resume_training(
            args.resume, model, optimizer, learningRate_scheduler, "FusionDiff", device)
        print(f"fully resumed: {Path(args.resume).resolve()} epoch={start_epoch} step={num_train_step}")
    best_val_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        # train
        model.train()
        loss_sum = 0.0
        writer.add_scalar('lr_epoch: ', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        for train_step, train_images in tqdm(enumerate(train_dataloader), desc="train step"):
            optimizer.zero_grad()
            if isinstance(train_images, dict):
                train_sourceImg1, train_sourceImg2 = train_images["a"].to(device), train_images["b"].to(device)
                clearImg = train_images["target"].to(device)
            else:
                train_sourceImg1, train_sourceImg2, clearImg = [image.to(device) for image in train_images[:3]]

            t = torch.randint(0, T, (clearImg.shape[0],), device=device).long()
            scale_loss = diffusion.train_losses(model, train_sourceImg1, train_sourceImg2, clearImg, t, concat_type, loss_scale)
            writer.add_scalar('loss_step: ', float(scale_loss.detach().item()), num_train_step)

            if train_step % loss_step == 0:
                print(
                    f" [epoch] {epoch}/{epochs}    "
                    f"[epoch_step] {train_step}/{train_step_sum}     "
                    f"[train_step] {num_train_step}     "
                    f"[loss] {scale_loss.item() / loss_scale :.6f}     "
                    f"[scale_loss] {scale_loss.item() :.6f}     "
                    f"[lr] {optimizer.state_dict()['param_groups'][0]['lr'] :.6f}     "
                    f"[t] {t.cpu().numpy()}")

                log.write(f" [epoch] {epoch}/{epochs}    "
                          f"[epoch_step] {train_step}/{train_step_sum}     "
                          f"[train_step] {num_train_step}     "
                          f"[loss] {scale_loss.item() / loss_scale :.6f}     "
                          f"[scale_loss] {scale_loss.item() :.6f}     "
                          f"[lr] {optimizer.state_dict()['param_groups'][0]['lr'] :.6f}     "
                          f"[t] {t.cpu().numpy()}"
                          f"\n")

            scale_loss.backward()
            optimizer.step()

            loss_sum += float(scale_loss.detach().item())
            num_train_step += 1
            if args and args.max_train_steps >= 0 and num_train_step >= args.max_train_steps:
                break

        val_average = None
        if val_dataloader is not None:
            model.eval()
            torch.manual_seed(args.seed)
            val_sum, val_count = 0.0, 0
            with torch.no_grad():
                for val_images in val_dataloader:
                    va, vb, vg = (val_images[k].to(device) for k in ("a", "b", "target"))
                    vt = (torch.zeros((vg.shape[0],), dtype=torch.long, device=device)
                          if args.validation_mode == "smoke" else
                          torch.randint(0, T, (vg.shape[0],), device=device))
                    val_loss = diffusion.train_losses(model, va, vb, vg, vt, concat_type, loss_scale)
                    val_loss_value = float(val_loss.detach().item())
                    val_sum += val_loss_value; val_count += 1
                    if args.validation_mode == "smoke": break
            if val_count:
                val_average = val_sum / val_count
                print(f"validation loss: {val_average / loss_scale:.6f} samples={val_count}")
        checkpoint_args = dict(method="FusionDiff", model=model, optimizer=optimizer,
                               scheduler=learningRate_scheduler, epoch=epoch,
                               global_step=num_train_step, config=config, contract=contract)
        save_checkpoint("checkpoints/latest.pt", **checkpoint_args)
        save_checkpoint(f"checkpoints/epoch_{epoch:04d}.pt", **checkpoint_args)
        if args.validation_mode != "smoke" and val_average is not None and val_average < best_val_loss:
            best_val_loss = val_average
            save_checkpoint("checkpoints/best_val_loss.pt", **checkpoint_args)
        if args and args.max_train_steps >= 0 and num_train_step >= args.max_train_steps:
            writer.close()
            print("training smoke limit reached")
            return

        aver_loss = loss_sum / train_step_sum

        if epoch % save_model_epoch_step == 0:
            save_model(model, epoch, timestr)
        if epoch == epochs - 1:
            save_model(model, epoch, timestr)

        # update learning rate
        if learningRate_scheduler is not None:
            learningRate_scheduler.step()
        writer.add_scalar('aver_loss_epoch: ', aver_loss, epoch)
        log.write("\n")

    print("End of training")
    log.write("End of training \n")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--dataset-format", choices=["directory", "metadata"], default="directory")
    parser.add_argument("--train-metadata")
    parser.add_argument("--val-metadata")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int, default=-1)
    parser.add_argument("--resume")
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--allow-legacy-checkpoint", type=int, choices=(0, 1), default=0)
    parser.add_argument("--validation-mode", choices=("smoke", "loss"), default="loss")
    parser.add_argument("--fail-on-split-overlap", type=int, choices=(0, 1), default=0)
    parsed = parser.parse_args()
    torch.manual_seed(parsed.seed)
    random.seed(parsed.seed)
    np.random.seed(parsed.seed)
    train(parsed.config, parsed)
