from contextlib import nullcontext
from pathlib import Path
import sys
import os
from datetime import datetime
ROOT = Path(__file__).resolve().parents[3]  
sys.path.insert(0, str(ROOT))
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
import model.utils.trainingScr.trainingCard as trainingCardMod
import model.utils.modules.encoderAndHead as encoderAndHead
import model.utils.modules.qat as qatMod
import model.utils.loss.loss as loss
import model.utils.inferScr.infer as inferMod
import yaml
import math
import json
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time
from torch.amp import autocast
import torch.quantization as tq
def cosine_lr(epoch, total_epochs, base_lr, min_lr, warmup_epochs):
    # epoch: 1..total_epochs
    if epoch <= warmup_epochs:
        lr = base_lr * epoch / max(1, warmup_epochs)
        warmup = True
        return lr, warmup

    # cosine annealing after warmup
    t = (epoch - warmup_epochs) / max(1, (total_epochs - warmup_epochs))  # 0..1
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * t))
    warmup = False
    return lr, warmup

def ddp_setup():
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ.get("RANK", -1))
    world = int(os.environ.get("WORLD_SIZE", -1))
    print(f"[ddp_setup] rank={rank} local_rank={local_rank} world={world}", flush=True)

    torch.cuda.set_device(local_rank)
    print(f"[ddp_setup] rank={rank} set_device cuda:{local_rank}", flush=True)

    dist.init_process_group(backend="nccl")
    print(f"[ddp_setup] rank={rank} init_process_group OK", flush=True)
    return local_rank

def ddp_cleanup():
    dist.destroy_process_group()

def _init_training(pkl_path,world_size, local_rank, device="cuda", warmup_epochs=10, batch = 64, num_workers=16, prefetch_factor=4, ckpt_path = None, isQAT = False, lr_rate = 3e-4):
    is_main = (local_rank == 0)
    if is_main:
        print("\033[1;37;42m[Hint] Initializing training...\033[0m")
        print("[Status] DataLoader: offline, Model: offline\n Starting initialization DataLoader...")
    try:
        transform = trainingCardMod.FeatureAugment(noise_std=0.05, drop_prob=0.1)
        dataset = trainingCardMod.CreatDataset(
                    pkl_path=pkl_path,
                    min_conf=0.3,
                    transform=transform)
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=dist.get_rank(),      
            shuffle=True,
            drop_last=True,
        )
    except Exception as e:
        print(f"Error during DataLoader initialization: {e}")
    if is_main:
        print("[Device] Using device:", device)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,             
        sampler=sampler,           
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,           
        persistent_workers=(num_workers > 0),
        prefetch_factor=prefetch_factor,
    )
    
    device = torch.device(f"cuda:{local_rank}")
    try:
        if is_main:
            print("[Status] DataLoader: Online, Model: offline\n Starting initialization Model...")
        
        model = encoderAndHead.Model(
                    in_channels=512,
                    out_channels=512,
                    warmup_epochs = warmup_epochs,
                    proj_dim = 128).to(device)
        # disable inplace before training (safe even if already applied)
        model.apply(disable_inplace)
        
        if ckpt_path is not None:
            ckpt_path = Path(ckpt_path)

            if not ckpt_path.exists():
                raise FileNotFoundError(f"[ERROR] Checkpoint not found: {ckpt_path}")

            print(f"[INFO] Loading checkpoint from: {ckpt_path}")

            ckpt = torch.load(ckpt_path, map_location="cpu")

            # 常見格式保護
            if "model" in ckpt:
                model.load_state_dict(ckpt["model"], strict=True)
            else:
                model.load_state_dict(ckpt, strict=True)


            print("[INFO] Checkpoint loaded successfully.")
        else:
            ckpt = None
            print("[Status] No checkpoint, start with empty")
            
        if bool(isQAT):
            model = qatMod.QATWrapper(model)
            model.qconfig = tq.get_default_qat_qconfig("fbgemm")
            tq.prepare_qat(model, inplace=True)



        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr_rate,
            weight_decay=1e-4
        )
        if ckpt_path is not None and not bool(isQAT):
            optimizer.load_state_dict(ckpt["optimizer"])


        criterion = loss.NTXentLoss(temperature=0.2).to(device)
        klsim  = loss.KLSimilarityLoss(tau_t=0.07, tau_s=0.2).to(device)
    except Exception as e:
        print(f"Error during Model initialization: {e}")
    if is_main:
        print("[Status] DataLoader: Online, Model: Online")
        print("\033[1;37;42m[Hint] All initialization complete.\033[0m")
    rank = dist.get_rank()

    print(f"[Rank {rank}] len(dataset)={len(dataset)} batch={batch} drop_last=True world={world_size}", flush=True)
    print(f"[Rank {rank}] len(dataloader)={len(dataloader)}", flush=True)

    return dataloader, sampler, model, optimizer, criterion, klsim, ckpt


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
def preprocess(yolo_weight = None, dataset_path=None, preprocess_res_path=None):
    yolov7_feats = trainingCardMod.PreProcess(weight_path=yolo_weight,
                                        dataset_path=dataset_path,
                                        preprocess_res_path="model/res/checkpoints",)
    try:
        
        yolov7_feats._preprocess_yolov7()
    except Exception as e:
        print(f"Error during preprocessing: {e}")

def disable_inplace(m: nn.Module):
    # 強制關閉 inplace
    if isinstance(m, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Hardswish, nn.LeakyReLU, nn.ELU, nn.CELU)):
        if hasattr(m, "inplace"):
            m.inplace = False
    if hasattr(m, "inplace") and getattr(m, "inplace") is True:
        m.inplace = False


def append_epoch_log_json(
    json_path: str | Path,
    record: dict
):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []

    data.append(record)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)  

def train(
    epochs,
    dataloader,
    sampler,
    model,
    optimizer,
    criterion,
    klsim,
    local_rank,
    ckpt,
    device="cuda",
    save_interval=10,
    save_dir="training/tracking",
    warmup_epochs=10,
    base_lr=3e-4,
    beta=0.9,
    min_lr=3e-6,
    max_norm =1,
    isQAT = False,
):
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 not supported on this GPU")
    
    # torch.autograd.set_detect_anomaly(True)

    # -------- DDP role --------
    is_main = (local_rank == 0)

    if is_main:
        os.makedirs(save_dir, exist_ok=True)
        if isQAT:
            os.makedirs(f"{save_dir}/qat", exist_ok=True)
            save_dir = f"{save_dir}/qat"
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = Path(save_dir) / "log" / f"training_log_{run_id}.json"
 
        with log_path.open("w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

    # -------- device (DDP 固定寫法) --------
    device = torch.device(f"cuda:{local_rank}")
    amp_dtype = torch.bfloat16
    

    

    avg_loss = 0.0
    lr = base_lr
    status = True
    
    if ckpt is not None:
        start_epoch = int(ckpt.get("epoch", 0)) + 1
    else:
        start_epoch = 1
    if isQAT:
        autocast_context = nullcontext()
    else:
        autocast_context = autocast("cuda", dtype=torch.bfloat16)

    phase_epochs = int(epochs)
    end_epoch = start_epoch + phase_epochs  # range is [start_epoch, end_epoch) 
    
    if is_main:
        print("\033[1;37;42m[Hint] Starting training...\033[0m")
        pbar = tqdm(range(start_epoch, end_epoch), desc="Training", dynamic_ncols=True)
    else:
        pbar = range(start_epoch, end_epoch)

    for epoch in pbar:
        # DDP sampler set_epoch
        if sampler is not None:
            sampler.set_epoch(epoch)

        effective_epoch = epoch - start_epoch + 1
        lr, status = cosine_lr(effective_epoch, phase_epochs, base_lr, min_lr, warmup_epochs)
        set_lr(optimizer, lr)

        model.train()
        if hasattr(model.module, "rmb"):
            model.module.rmb.set_epoch(epoch)
        
        if effective_epoch == 3 and isQAT is not False:
            model.apply(torch.quantization.disable_observer)
            print(f"[Rank {dist.get_rank()}] Disabled observers for QAT at epoch {epoch}")
        if effective_epoch == 5 and isQAT is not False:
            model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
            print(f"[Rank {dist.get_rank()}] Frozen batchnorm stats for QAT at epoch {epoch}")
        total_loss = 0.0


        for step, (v1, v2, t) in enumerate(dataloader, start=1):
            v1 = v1.to(device, non_blocking=True)
            v2 = v2.to(device, non_blocking=True)
            t  = t.to(device, non_blocking=True).detach()

            with autocast_context:
                
                v = torch.cat([v1, v2], dim=0)        # [2B, C, H, W]
                z = model(v)                          # [2B, D]
                z1, z2 = torch.chunk(z, 2, dim=0)     # 各 [B, D]

                
                loss_nt = criterion(z1, z2)
                loss_kl = 0.5 * (klsim(t, z1) + klsim(t, z2))
                if epoch > warmup_epochs:
                    last_stage = int(0.8 * epochs)

                    if epoch >= last_stage:
                        # 最後 20% 固定
                        use_beta = 0.5
                    else:
                        progress = (epoch - warmup_epochs) / (last_stage - warmup_epochs)
                        use_beta = 0.9 - progress * (0.9 - 0.5)
                else:
                    use_beta = beta
                loss = use_beta * loss_nt + (1.0 - use_beta) * loss_kl


            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_every = 10
            if max_norm and max_norm > 0 and (step % clip_every == 0):
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                total_norm = None
            optimizer.step()
            
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(dataloader))


        if is_main:
            pbar.set_postfix(
                {
                    "avg_loss": f"{avg_loss:.4f}",
                    "total_loss": f"{total_loss:.4f}",
                    "lr": f"{lr:.2e}",
                    "warmup": str(status),
                },
                refresh=True,
            )
    
            append_epoch_log_json(
                log_path,
                {
                    "epoch": epoch,
                    "avg_loss": round(avg_loss, 6),
                    "nt_loss": float(loss_nt.detach().item()),
                    "kl_loss": float(loss_kl.detach().item()),
                    "lr": lr,
                    "warmup": status,
                    "beta": use_beta,
                    "total_norm": float(total_norm) if total_norm is not None else None,
                },
            )


            if save_interval is not None and epoch % save_interval == 0:
                state = {
                    "epoch": epoch,
                    "model": model.module.state_dict(), 
                    "optimizer": optimizer.state_dict(),
                    "loss": avg_loss,
                }
                torch.save(state, os.path.join(save_dir, f"epoch_{epoch}.pt"))
                torch.save(state, os.path.join(save_dir, "epoch_last.pt"))

    if is_main:
        print("\033[1;37;42m [Hint] Training completed\033[0m")
    if dist.is_initialized():
        dist.barrier()


             
def inference(
    ckpt_path: str,
    yolo_weight: str,
    datasets_root: str = "tracking/utils/valScr/datasets",
    device: str = "cuda",
    tau: float = 0.2,
    topk: int = 10,
    min_conf: float = 0.3,
    iou_thr: float = 0.5,
):
    """
    Run validation ONCE after training
    """
    # build MainInfer ( reuse your abstraction)
    infer_engine = inferMod.MainInfer(
        yolo_weight=yolo_weight,
        ckpt_path=ckpt_path,
    )

    pic_pre = Path(datasets_root) /  "pre"
    pic_cur = Path(datasets_root) /  "cur"

    pairs = []
    for p in sorted(pic_pre.iterdir()):
        if p.is_file() and (pic_cur / p.name).exists():
            pairs.append((str(p), str(pic_cur / p.name)))

    all_metrics = []
    used = 0
    print("\033[1;37;42m[Hint] Starting Inference...\033[0m")
    for frame_pre, frame_cur in pairs:
        
        res, metrics = infer_engine.infer_two_img(
            frame_cur=frame_cur,
            frame_pre=frame_pre,
            tau=tau,
            topk=topk,
            min_conf=min_conf,
            isVal=True,                      
            datasets_root=datasets_root,
            iou_thr=iou_thr,
        )

        if metrics is None:
            continue

        used += 1
        all_metrics.append(metrics)

        print(f"[VAL] {Path(frame_pre).stem}: {metrics}")

    # aggregate
    if not all_metrics:
        print(f"\033[1;37;41m[VAL] No valid pairs\033[0m \n plz check {pic_pre} or {pic_cur} ")
        return None

    mean_metrics = {}
    for k in all_metrics[0].keys():
        vals = [m[k] for m in all_metrics if m[k] == m[k]]  # skip nan
        mean_metrics[k] = sum(vals) / len(vals) if vals else float("nan")

    print("\n========== Validation Summary ==========")
    for k, v in mean_metrics.items():
        print(f"{k:>15s}: {v:.4f}")

    return {
        "num_pairs": len(pairs),
        "num_used": used,
        "mean_metrics": mean_metrics,
        "per_pair": all_metrics,
    }

def load_config_yaml(config_path: str):
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if cfg is None:
            raise ValueError(f"Empty yaml file: {path}")
        return cfg

def _preprocess(path:str =None):
    train_cfg_path =  Path(f"{path}/training_conf.yaml")
    cfg =load_config_yaml(train_cfg_path)
    yolov7_pkl_path =  str(cfg["yoloWeight"])
    yolov7_feats_path_pkl = str(cfg["yoloFeats"])
    dataset_path = str(cfg["datasetPath"])
    if cfg["isPreprocess"]:
        preprocess(
                yolo_weight=yolov7_pkl_path,
                dataset_path=dataset_path,
                preprocess_res_path=yolov7_feats_path_pkl,
            )
def _train(path:str =None):
    train_cfg_path = Path(f"{path}/training_conf.yaml")
    cfg = load_config_yaml(train_cfg_path)

    local_rank = None
    try:

        local_rank = ddp_setup()
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        pkl_path = os.path.join(
            "model/res/checkpoints",
            f"rank{rank}.pkl"
        )
        if cfg["QAT"]['isQAT'] is not False:
            print(f"[Rank {dist.get_rank()}] QAT is enabled.")
            epochs = cfg["QAT"]["epoch"]
            lr_rate = cfg["QAT"]["lr_rate"]
            min_lr=cfg["QAT"]["minLr"]
            print(f"[info] QAT epochs: {epochs}, lr_rate: {lr_rate}, min_lr: {min_lr}")
        else:
            epochs = cfg["epoch"]
            lr_rate = cfg["baseLr"]
            min_lr=cfg["minLr"]
            print(f"[info] Training epochs: {epochs}, lr_rate: {lr_rate}, min_lr: {min_lr}")

        if cfg["isTraining"]:
            dataloader, sampler, model, optimizer, criterion, klsim, ckpt= _init_training(
                pkl_path=pkl_path,
                local_rank=local_rank,
                world_size=world_size,
                ckpt_path= cfg["ckpt"],
                warmup_epochs=cfg["warmupEpochs"],
                batch=cfg["batch_size"],
                num_workers=cfg["num_workers"],
                prefetch_factor=cfg["prefetch_factor"],
                isQAT=cfg["QAT"]['isQAT'],
                lr_rate=lr_rate,
            )
            try:
                
                train(
                    epochs=epochs,
                    dataloader=dataloader,
                    sampler=sampler,
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    klsim=klsim,
                    local_rank=local_rank,
                    ckpt= ckpt,
                    save_dir=f"model/res/checkpoints",
                    warmup_epochs=cfg["warmupEpochs"],
                    base_lr=lr_rate,
                    min_lr=min_lr,
                    max_norm = cfg["max_norm"],
                    isQAT = cfg['QAT']["isQAT"],
                    
                )
            except Exception as e:
                print(f"Error during training: {e}")
        else:
            print("\033[1;37;42m[Hint] isTraining is False stop training....\033[0m")

        
        dist.barrier(device_ids=[local_rank])


    finally:
        if dist.is_available() and dist.is_initialized():
            ddp_cleanup()

def _inference():
    train_cfg_path = ROOT /"training/conf/training_conf.yaml"
    cfg =load_config_yaml(train_cfg_path)
    yolov7_pkl_path = "/home/soic/Desktop/tracking/v24/training/yolov7/weights/yolov7_best.pt"
    if cfg["isInference"]:
        inference(
            ckpt_path=f"{ROOT}/training/tracking/res/checkpoints/epoch_last.pt",
            yolo_weight=yolov7_pkl_path,
            datasets_root=f"/home/soic/Desktop/tracking/v24/training/tracking/dataset/val",
            device = (
                "cuda" if torch.cuda.is_available()
                else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                else "cpu"
            )
        )

if __name__ == "__main__":
    
    train_cfg_path = "model/conf/training_conf.yaml"
    cfg =load_config_yaml(train_cfg_path)
    
    
    
    yolov7_pkl_path = "model/yolov7/weights/yolov7_best.pt"
    yolov7_feats_path_pkl = str(ROOT / cfg["yoloFeats"])
    dataset_path = str(ROOT / cfg["datasetPath"])
    
