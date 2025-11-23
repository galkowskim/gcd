import sys
import os
import torch
import pandas as pd
import torchvision
import omegaconf
from pathlib import Path
from argparse import ArgumentParser
from datasets import load_dataset
from PIL import Image
import torchvision.transforms as T

from proxies import *
from losses import *
from classifiers import *
from dae import *

import logging
logging.basicConfig(level = logging.INFO)
log = logging.getLogger(__name__)

def init_class_from_string(class_name):
    return getattr(sys.modules[__name__], class_name)

def get_cfg(args):
    path_cfg = (Path(args.log_dir_path) if args.log_dir_path else Path(args.direction_path).parents[2]) / '.hydra' / 'config.yaml'
    cfg = omegaconf.OmegaConf.load(path_cfg)
    return cfg

def init_loss(cfg):
    loss_kwargs = cfg.strategy.ce_loss_kwargs
    loss_kwargs['make_output_dir'] = False
    clf_kwargs = loss_kwargs.components.clf
    clf_name = clf_kwargs.pop('_target_').split('.')[-1]
    log.info(f'Using {clf_name} classifier')
    if clf_name == 'DenseNet':
        clf_kwargs.use_probs_and_query_label = False
    elif clf_name == 'ResNet':
        clf_kwargs.use_softmax_and_query_label = True
    clf = init_class_from_string(clf_name)(**clf_kwargs)
    comps_kwargs = loss_kwargs.components
    comps_name = comps_kwargs._target_.split('.')[-1]
    comps_kwargs.pop('_target_')
    comps_kwargs.pop('clf', None)
    comps_kwargs.pop('src_img_path', None)
    comps = init_class_from_string(comps_name)(
        src_img_path = None,
        clf = clf,
        **comps_kwargs)
    loss_kwargs.pop('components')
    loss_name = 'CounterfactualLossFromGeneralComponents'
    loss_kwargs.weight_cls = cfg.strategy.ce_loss_kwargs.weight_cls
    loss_kwargs.weight_lpips = cfg.strategy.ce_loss_kwargs.weight_lpips
    loss = init_class_from_string(loss_name)(components = comps, **loss_kwargs)
    return loss

def init_dae(cfg):
    dae_kwargs = cfg.strategy.dae_kwargs
    dae_kwargs['make_output_dir'] = False
    dae_type = cfg.strategy.dae_type
    if dae_type == 'default':
        dae_class = DAE
    elif dae_type == 'chexpert':
        dae_class = DAECheXpert
    else:
        raise NotImplementedError('DAE type not recognized')
    log.info(f'Using {dae_class}')
    dae = dae_class(**dae_kwargs)
    return dae

def load_direction(path, device):
    grad = torch.load(Path(path), map_location='cpu').to(device)
    if torch.count_nonzero(grad) == 0:
        raise ValueError("Direction tensor contains only zeros.")
    return grad

def iter_hf_by_label(dataset_name, split, label, start_index, n_samples, token=None, cache_dir=None):
    kwargs = {"streaming": True}
    if token is None:
        token = os.getenv('HF_TOKEN', None)
    if cache_dir is None:
        cache_dir = os.getenv('DATASET_CACHE', None)
    if token is not None:
        kwargs["token"] = token
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    ds = load_dataset(dataset_name, split=split, **kwargs)
    count = 0
    emitted = 0
    for ex in ds:
        if ex.get('label', None) == label:
            if count >= start_index:
                yield ex
                emitted += 1
                if emitted >= n_samples:
                    break
            count += 1

def pil_to_tensor_01(img: Image.Image, image_size: int):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    transform = T.Compose([
        T.CenterCrop(image_size),
        T.Resize(image_size),
        T.ToTensor(),    # [0,1]
    ])
    return transform(img)

def to_dae_range(x01: torch.Tensor):
    return (x01 - 0.5) * 2

def from_dae_range(xm1p1: torch.Tensor):
    return (xm1p1 + 1) / 2

def main(args):
    log.info("Batch direction transfer (HF ImageNet)")
    cfg = get_cfg(args)
    device = torch.device(cfg.device)
    loss = init_loss(cfg)
    dae = init_dae(cfg)
    direction = load_direction(args.direction_path, device)

    out_dir = (Path(args.log_dir_path) if args.log_dir_path else Path(args.direction_path).parents[2]) / 'direction_transfer' / args.subdir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []

    image_size = dae.config.img_size if hasattr(dae, 'config') else 256
    n_done = 0
    for ex in iter_hf_by_label(
        dataset_name=args.dataset_name,
        split=args.split,
        label=args.target_class,
        start_index=args.start_index,
        n_samples=args.n_samples,
        token=args.hf_token,
        cache_dir=args.hf_cache_dir
    ):
        img_pil = ex['image'] if isinstance(ex['image'], Image.Image) else Image.fromarray(ex['image'])
        idx = ex['__index_level_0__'] if '__index_level_0__' in ex else n_done
        # Prepare tensors
        img01 = pil_to_tensor_01(img_pil, image_size).unsqueeze(0).to(device)
        img_m1p1 = to_dae_range(img01)
        # Encode
        latent_sem = dae.encode(img_m1p1)
        latent_ddim = dae.encode_stochastic(img_m1p1, latent_sem)
        # One-step transfer using fixed step size
        step = torch.tensor([[args.step_size]], device=device)
        grad = direction
        latent_sem_new = latent_sem - step * grad
        img_trans = dae.render(latent_ddim, latent_sem_new, T=args.T_render)
        # Save pair
        orig_name = f"original_{n_done+1}.png"
        trans_name = f"inpaint_{n_done+1}.png"
        torchvision.utils.save_image(img01, out_dir / orig_name)
        torchvision.utils.save_image(img_trans, out_dir / trans_name)
        # Log classifier probabilities (optional)
        with torch.no_grad():
            comps = loss.get_components(img_trans)
            prob_after = loss.get_query_label_probability(comps['predictions']).item()
            pred_before_logits = loss.components.classifier(img01)
            prob_before = loss.get_query_label_probability(pred_before_logits).item()
        csv_rows.append({
            "idx": int(idx),
            "orig_path": orig_name,
            "inpaint_path": trans_name,
            "prob_before": prob_before,
            "prob_after": prob_after
        })
        n_done += 1
        if n_done % 25 == 0:
            log.info(f"Processed {n_done}/{args.n_samples}")
    # Save CSV
    pd.DataFrame(csv_rows).to_csv(out_dir / "pairs.csv", index=False)
    log.info(f"Saved {n_done} pairs to {out_dir}")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--direction-path', type=str, required=True, help='Path to direction .pt file from proxy run')
    parser.add_argument('--log-dir-path', type=str, default=None, help='Log dir path with .hydra config (defaults to infer from direction path)')
    parser.add_argument('--dataset-name', type=str, default='imagenet-1k', help='HF dataset name')
    parser.add_argument('--split', type=str, default='train', help='HF split')
    parser.add_argument('--target-class', type=int, required=True, help='HF class id to sample')
    parser.add_argument('--n-samples', type=int, default=500, help='Number of images to transfer')
    parser.add_argument('--start-index', type=int, default=0, help='Skip N matching samples before collecting')
    parser.add_argument('--hf-token', type=str, default=None, help='HF token (or set HF_TOKEN env)')
    parser.add_argument('--hf-cache-dir', type=str, default=None, help='HF cache (or set DATASET_CACHE env)')
    parser.add_argument('--step-size', type=float, default=1.0, help='Step size multiplier for the direction')
    parser.add_argument('--T-render', type=int, default=100, help='Number of DDIM steps for rendering')
    parser.add_argument('--subdir-name', type=str, default='hf_batch', help='Output subdir name under direction_transfer')
    args = parser.parse_args()
    main(args)

