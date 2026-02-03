#!/usr/bin/env python3
#
# Verify NEW-ATOM oracle via directional-derivative test.
# We create candidate Gaussians along top-error pixels (ray + depth),
# compute dL/dalpha for candidate opacities, then compare to finite-diff ΔL.
#

import os
import sys
import random
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn

# Ensure repo root is on sys.path when running as a script.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import render, GaussianModel
from scene import Scene
from utils.general_utils import inverse_sigmoid, safe_state
from utils.loss_utils import l1_loss, ssim
from utils.sh_utils import RGB2SH
from utils.graphics_utils import fov2focal

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except Exception:
    FUSED_SSIM_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def compute_loss(rendered, gt, lambda_dssim):
    ll1 = l1_loss(rendered, gt)
    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(rendered.unsqueeze(0), gt.unsqueeze(0))
    else:
        ssim_value = ssim(rendered, gt)
    loss = (1.0 - lambda_dssim) * ll1 + lambda_dssim * (1.0 - ssim_value)
    return loss


def parse_float_list(s):
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    vals = []
    for p in parts:
        if not p:
            continue
        vals.append(float(p))
    return vals


def rankdata(a):
    a = np.asarray(a)
    n = a.size
    if n == 0:
        return a
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and np.isclose(sorted_a[j + 1], sorted_a[i], atol=1e-12, rtol=0.0):
            j += 1
        if j > i:
            avg_rank = 0.5 * (i + j)
            ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    return ranks


def spearman_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return float("nan")
    rx = rankdata(x)
    ry = rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return np.corrcoef(rx, ry)[0, 1]


def topk_overlap(pred, actual, k):
    if k is None or k <= 0:
        return float("nan")
    n = len(pred)
    if n == 0:
        return float("nan")
    k = min(k, n)
    pred_idx = np.argsort(pred)[:k]
    actual_idx = np.argsort(actual)[:k]
    inter = len(set(pred_idx.tolist()) & set(actual_idx.tolist()))
    return inter / float(k)


def pixel_to_world(view, px, py, depth, fx, fy, cx, cy):
    # Camera space ray
    x_cam = (px + 0.5 - cx) / fx
    y_cam = (py + 0.5 - cy) / fy
    p_cam = torch.tensor([x_cam * depth, y_cam * depth, depth, 1.0], device="cuda", dtype=torch.float32)
    # Use the same normalized transform as the renderer (includes scene translate/scale).
    view_inv = view.world_view_transform.inverse()
    p_world = (view_inv @ p_cam)[:3]
    return p_world


def build_pcd_invdepth(view, gaussians):
    # Project Gaussian centers into a sparse inverse-depth map using nearest (min depth).
    pts = gaussians.get_xyz.detach()
    if pts.numel() == 0:
        return None
    device = pts.device
    H = int(view.image_height)
    W = int(view.image_width)

    ones = torch.ones((pts.shape[0], 1), device=device, dtype=pts.dtype)
    pts_hom = torch.cat([pts, ones], dim=1)

    viewmat = view.world_view_transform
    projmat = view.full_proj_transform

    p_view = pts_hom @ viewmat
    z = p_view[:, 2]

    p_clip = pts_hom @ projmat
    ndc = p_clip[:, :3] / (p_clip[:, 3:4] + 1e-8)

    finite = torch.isfinite(z) & torch.isfinite(ndc).all(dim=1)
    x = ((ndc[:, 0] + 1.0) * W - 1.0) * 0.5
    y = ((ndc[:, 1] + 1.0) * H - 1.0) * 0.5

    mask = finite & (z > 0)
    if mask.sum() == 0:
        return None

    xi = x[mask].round().long()
    yi = y[mask].round().long()
    mask2 = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    if mask2.sum() == 0:
        return None
    xi = xi[mask2]
    yi = yi[mask2]
    idx = yi * W + xi
    z_sel = z[mask][mask2]

    depth_flat = torch.full((H * W,), float("inf"), device=device, dtype=pts.dtype)
    if hasattr(depth_flat, "scatter_reduce_"):
        depth_flat.scatter_reduce_(0, idx, z_sel, reduce="amin", include_self=True)
    else:
        # Fallback for older torch
        depth_flat_cpu = depth_flat.cpu().numpy()
        idx_cpu = idx.cpu().numpy()
        z_cpu = z_sel.cpu().numpy()
        for i, zi in zip(idx_cpu, z_cpu):
            if zi < depth_flat_cpu[i]:
                depth_flat_cpu[i] = zi
        depth_flat = torch.from_numpy(depth_flat_cpu).to(device)

    depth_map = depth_flat.view(H, W)
    invdepth = torch.zeros_like(depth_map)
    valid = torch.isfinite(depth_map)
    invdepth[valid] = 1.0 / (depth_map[valid] + 1e-8)
    return invdepth


def main():
    parser = ArgumentParser(description="Verify NEW-ATOM oracle via directional derivative test.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    optim = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--view_index", default=0, type=int)
    parser.add_argument("--num_pixels", default=200, type=int)
    parser.add_argument("--depth_multipliers", default="0.7,1.0,1.3", type=str)
    parser.add_argument("--max_candidates", default=800, type=int)
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--alpha0", default=1e-3, type=float)
    parser.add_argument("--epsilon", default=1e-3, type=float)
    parser.add_argument("--central_diff", action="store_true")
    parser.add_argument("--scale_factor", default=1.0, type=float)
    parser.add_argument("--scale_min_mult", default=0.25, type=float)
    parser.add_argument("--scale_max_mult", default=4.0, type=float)
    parser.add_argument("--use_dataset_depth", action="store_true")
    parser.add_argument("--use_pcd_depth", action="store_true")
    parser.add_argument("--pcd_fallback_render", action="store_true")
    parser.add_argument("--depth_min", default=0.01, type=float)
    parser.add_argument("--depth_max", default=100.0, type=float)
    parser.add_argument("--topk", default=20, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--out_dir", default="", type=str)
    parser.add_argument("--no_plot", action="store_true")
    # Prefer cfg_args merge when available, but allow running without it.
    args_cmdline = parser.parse_args()
    cfg_args_path = ""
    if getattr(args_cmdline, "model_path", None):
        cfg_args_path = os.path.join(args_cmdline.model_path, "cfg_args")
    if cfg_args_path and os.path.exists(cfg_args_path):
        args = get_combined_args(parser)
    else:
        args = args_cmdline

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    safe_state(False)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    opt = optim.extract(args)

    # Resolve which iteration to load.
    load_iteration = args.iteration
    pc_dir = os.path.join(dataset.model_path, "point_cloud")
    if load_iteration is not None:
        if load_iteration == -1:
            if not os.path.isdir(pc_dir):
                raise RuntimeError(
                    f"No point_cloud directory found at: {pc_dir}\n"
                    "You passed --iteration -1 which expects a trained model.\n"
                    "Fix: point --model_path to a trained run, or pass --iteration <iter>.\n"
                )
            iters = []
            for name in os.listdir(pc_dir):
                if name.startswith("iteration_"):
                    try:
                        iters.append(int(name.split("_")[-1]))
                    except ValueError:
                        pass
            if not iters:
                raise RuntimeError(
                    f"No iteration_* folders found in: {pc_dir}\n"
                    "Fix: run training to generate checkpoints, or pass the correct --model_path."
                )
            load_iteration = max(iters)
        else:
            if not os.path.isdir(pc_dir):
                raise RuntimeError(
                    f"No point_cloud directory found at: {pc_dir}\n"
                    "Fix: point --model_path to a trained run."
                )
            iter_dir = os.path.join(pc_dir, f"iteration_{load_iteration}")
            if not os.path.isdir(iter_dir):
                raise RuntimeError(
                    f"Missing iteration folder: {iter_dir}\n"
                    "Fix: pass an existing --iteration or correct --model_path."
                )

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    views = scene.getTrainCameras() if args.split == "train" else scene.getTestCameras()
    if not views:
        raise RuntimeError(f"No cameras found for split={args.split}")
    if args.view_index < 0 or args.view_index >= len(views):
        raise RuntimeError(f"Invalid --view_index {args.view_index} (0..{len(views)-1})")
    view = views[args.view_index]

    # Base render for error + depth map.
    render_pkg = render(view, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp)
    image = render_pkg["render"]
    if view.alpha_mask is not None:
        image = image * view.alpha_mask.cuda()
    gt_image = view.original_image.cuda()
    # Choose depth source.
    depth_source = "render_invdepth"
    invdepth = render_pkg["depth"].detach()
    if invdepth.ndim == 3:
        invdepth = invdepth[0]
    elif invdepth.ndim == 4:
        invdepth = invdepth[0, 0]
    invdepth_pcd = None
    if args.use_pcd_depth:
        invdepth_pcd = build_pcd_invdepth(view, gaussians)
        if invdepth_pcd is not None:
            if invdepth_pcd.ndim == 3:
                invdepth_pcd = invdepth_pcd[0]
            elif invdepth_pcd.ndim == 4:
                invdepth_pcd = invdepth_pcd[0, 0]
            if args.pcd_fallback_render:
                # Use PCD depth where available, otherwise render depth.
                mask = invdepth_pcd > 0
                invdepth = invdepth.clone()
                invdepth[mask] = invdepth_pcd[mask]
                depth_source = "pcd+render_invdepth"
            else:
                invdepth = invdepth_pcd
                depth_source = "pcd_invdepth"
    if depth_source.startswith("render") and args.use_dataset_depth and getattr(view, "invdepthmap", None) is not None and getattr(view, "depth_reliable", False):
        invdepth = view.invdepthmap.detach().clone()
        depth_source = "dataset_invdepth"
    if invdepth.ndim == 3:
        invdepth = invdepth[0]
    elif invdepth.ndim == 4:
        invdepth = invdepth[0, 0]
    pcd_ratio = float((invdepth_pcd > 0).float().mean().item()) if invdepth_pcd is not None else float("nan")
    # Renderer returns inverse depth. Convert to depth for ray placement.
    depth = torch.zeros_like(invdepth)
    inv_mask = invdepth > 1e-6
    depth[inv_mask] = 1.0 / (invdepth[inv_mask] + 1e-8)

    # Error map for candidate selection.
    err = torch.mean(torch.abs(image - gt_image), dim=0)
    if view.alpha_mask is not None:
        err = err * view.alpha_mask[0].cuda()

    valid = inv_mask & (depth > args.depth_min) & (depth < args.depth_max)
    if valid.ndim == 3:
        valid = valid[0]
    if view.alpha_mask is not None:
        valid = valid & (view.alpha_mask[0].cuda() > 0.5)

    coords = valid.nonzero(as_tuple=False)
    if coords.numel() == 0:
        raise RuntimeError("No valid pixels with depth; cannot build candidates.")

    scores = err[valid]
    topk = min(args.num_pixels, scores.numel())
    vals, idx = torch.topk(scores, k=topk, largest=True)
    pix = coords[idx]  # (K, 2) with (y, x)

    # Build candidates along ray with depth multipliers.
    depth_multipliers = parse_float_list(args.depth_multipliers)
    if not depth_multipliers:
        depth_multipliers = [1.0]

    fx = fov2focal(view.FoVx, view.image_width)
    fy = fov2focal(view.FoVy, view.image_height)
    cx = view.image_width * 0.5
    cy = view.image_height * 0.5

    cand_xyz = []
    cand_dc = []
    cand_px = []
    cand_py = []
    cand_depth = []
    cand_mul = []

    for yx in pix:
        py = int(yx[0].item())
        px = int(yx[1].item())
        d = float(depth[py, px].item())
        if d <= 0:
            continue
        rgb = gt_image[:, py, px].unsqueeze(0)  # (1,3)
        dc = RGB2SH(rgb).squeeze(0)  # (3,)
        for m in depth_multipliers:
            dd = d * float(m)
            if dd <= 0:
                continue
            p_world = pixel_to_world(view, px, py, dd, fx, fy, cx, cy)
            cand_xyz.append(p_world)
            cand_dc.append(dc)
            cand_px.append(px)
            cand_py.append(py)
            cand_depth.append(dd)
            cand_mul.append(float(m))

    if not cand_xyz:
        raise RuntimeError("No candidates built (depth/pixel selection empty).")

    cand_xyz = torch.stack(cand_xyz, dim=0)
    cand_dc = torch.stack(cand_dc, dim=0)
    cand_px = torch.tensor(cand_px, device="cuda", dtype=torch.int64)
    cand_py = torch.tensor(cand_py, device="cuda", dtype=torch.int64)
    cand_depth = torch.tensor(cand_depth, device="cuda", dtype=torch.float32)
    cand_mul = torch.tensor(cand_mul, device="cuda", dtype=torch.float32)

    # Downsample candidates if too many.
    if cand_xyz.shape[0] > args.max_candidates:
        perm = torch.randperm(cand_xyz.shape[0], device="cuda")[: args.max_candidates]
        cand_xyz = cand_xyz[perm]
        cand_dc = cand_dc[perm]
        cand_px = cand_px[perm]
        cand_py = cand_py[perm]
        cand_depth = cand_depth[perm]
        cand_mul = cand_mul[perm]

    n_cand = cand_xyz.shape[0]

    # Candidate SH features.
    cand_features_dc = cand_dc.view(n_cand, 3, 1).transpose(1, 2).contiguous()
    n_sh = (gaussians.max_sh_degree + 1) ** 2
    n_rest = n_sh - 1
    if n_rest > 0:
        cand_features_rest = torch.zeros((n_cand, n_rest, 3), device="cuda")
    else:
        cand_features_rest = torch.zeros((n_cand, 0, 3), device="cuda")

    # Candidate scaling / rotation / opacity.
    base_scale = gaussians.get_scaling.detach()
    med_scale = torch.median(base_scale, dim=0).values
    min_scale = med_scale * float(args.scale_min_mult)
    max_scale = med_scale * float(args.scale_max_mult)

    # Approximate world-space sigma from pixel size at depth.
    cand_sigma_x = cand_depth / fx
    cand_sigma_y = cand_depth / fy
    cand_sigma = torch.maximum(cand_sigma_x, cand_sigma_y) * float(args.scale_factor)
    cand_sigma = torch.clamp(cand_sigma, min=min_scale.min().item(), max=max_scale.max().item())
    cand_scale = torch.stack([cand_sigma, cand_sigma, cand_sigma], dim=1)
    cand_scaling = torch.log(cand_scale)
    cand_rotation = torch.zeros((n_cand, 4), device="cuda")
    cand_rotation[:, 0] = 1.0

    alpha0 = float(args.alpha0)
    alpha0 = min(max(alpha0, 1e-6), 1.0 - 1e-6)
    logit0 = inverse_sigmoid(torch.tensor(alpha0, device="cuda"))
    cand_opacity = logit0.view(1, 1).repeat(n_cand, 1)

    # Build combined model (base + candidates).
    combined = GaussianModel(gaussians.max_sh_degree, opt.optimizer_type)
    combined.active_sh_degree = gaussians.active_sh_degree
    combined._xyz = nn.Parameter(torch.cat([gaussians._xyz.detach(), cand_xyz], dim=0), requires_grad=False)
    combined._features_dc = nn.Parameter(
        torch.cat([gaussians._features_dc.detach(), cand_features_dc], dim=0), requires_grad=False
    )
    combined._features_rest = nn.Parameter(
        torch.cat([gaussians._features_rest.detach(), cand_features_rest], dim=0), requires_grad=False
    )
    combined._scaling = nn.Parameter(torch.cat([gaussians._scaling.detach(), cand_scaling], dim=0), requires_grad=False)
    combined._rotation = nn.Parameter(torch.cat([gaussians._rotation.detach(), cand_rotation], dim=0), requires_grad=False)
    combined._opacity = nn.Parameter(torch.cat([gaussians._opacity.detach(), cand_opacity], dim=0), requires_grad=True)
    combined.max_radii2D = torch.zeros((combined._xyz.shape[0]), device="cuda")
    # Exposure data is only needed when train_test_exp is enabled.
    if dataset.train_test_exp:
        if hasattr(gaussians, "exposure_mapping") and hasattr(gaussians, "_exposure"):
            combined.exposure_mapping = gaussians.exposure_mapping
            combined.pretrained_exposures = getattr(gaussians, "pretrained_exposures", None)
            combined._exposure = gaussians._exposure
        else:
            combined.exposure_mapping = {view.image_name: 0}
            combined.pretrained_exposures = None
            combined._exposure = nn.Parameter(torch.eye(3, 4, device="cuda")[None], requires_grad=False)

    # Evaluate gradients for candidates.
    render_pkg = render(view, combined, pipe, background, use_trained_exp=dataset.train_test_exp)
    image = render_pkg["render"]
    if view.alpha_mask is not None:
        image = image * view.alpha_mask.cuda()
    loss = compute_loss(image, gt_image, opt.lambda_dssim)
    loss.backward()

    if combined._opacity.grad is None:
        raise RuntimeError("Opacity gradients are None for combined model.")

    base_loss = loss.item()
    alpha = torch.sigmoid(combined._opacity.detach())
    dlogit = combined._opacity.grad.detach()
    dL_dalpha = dlogit / (alpha * (1.0 - alpha) + 1e-6)

    base_count = gaussians._xyz.shape[0]
    radii = render_pkg.get("radii", None)
    if radii is not None:
        cand_radii = radii[base_count:]
        visible_ratio = float((cand_radii > 0).float().mean().item()) if cand_radii.numel() > 0 else 0.0
    else:
        visible_ratio = float("nan")
    cand_slice = slice(base_count, base_count + n_cand)
    dL_dalpha_cand = dL_dalpha[cand_slice].squeeze(-1).cpu().numpy()

    num_samples = min(args.num_samples, n_cand)
    sample_ids = np.random.choice(n_cand, size=num_samples, replace=False)

    pred_changes = []
    actual_changes = []
    meta_rows = []

    eps = float(args.epsilon)
    with torch.no_grad():
        for idx in sample_ids:
            gidx = base_count + int(idx)
            pred = eps * float(dL_dalpha[gidx].item())
            pred_changes.append(pred)

            cur_alpha = float(alpha[gidx].item())
            if args.central_diff:
                alpha_plus = min(max(cur_alpha + eps, 1e-4), 1.0 - 1e-4)
                alpha_minus = min(max(cur_alpha - eps, 1e-4), 1.0 - 1e-4)

                combined._opacity.data[gidx] = inverse_sigmoid(torch.tensor(alpha_plus, device="cuda"))
                render_pkg_p = render(view, combined, pipe, background, use_trained_exp=dataset.train_test_exp)
                image_p = render_pkg_p["render"]
                if view.alpha_mask is not None:
                    image_p = image_p * view.alpha_mask.cuda()
                loss_p = compute_loss(image_p, gt_image, opt.lambda_dssim).item()

                combined._opacity.data[gidx] = inverse_sigmoid(torch.tensor(alpha_minus, device="cuda"))
                render_pkg_m = render(view, combined, pipe, background, use_trained_exp=dataset.train_test_exp)
                image_m = render_pkg_m["render"]
                if view.alpha_mask is not None:
                    image_m = image_m * view.alpha_mask.cuda()
                loss_m = compute_loss(image_m, gt_image, opt.lambda_dssim).item()

                actual = 0.5 * (loss_p - loss_m)
                actual_changes.append(actual)
            else:
                alpha_plus = min(max(cur_alpha + eps, 1e-4), 1.0 - 1e-4)
                combined._opacity.data[gidx] = inverse_sigmoid(torch.tensor(alpha_plus, device="cuda"))
                render_pkg_p = render(view, combined, pipe, background, use_trained_exp=dataset.train_test_exp)
                image_p = render_pkg_p["render"]
                if view.alpha_mask is not None:
                    image_p = image_p * view.alpha_mask.cuda()
                loss_p = compute_loss(image_p, gt_image, opt.lambda_dssim).item()
                actual = loss_p - base_loss
                actual_changes.append(actual)

            # Restore original alpha0.
            combined._opacity.data[gidx] = inverse_sigmoid(torch.tensor(cur_alpha, device="cuda"))

            meta_rows.append(
                (
                    int(idx),
                    int(cand_px[idx].item()),
                    int(cand_py[idx].item()),
                    float(cand_depth[idx].item()),
                    float(cand_mul[idx].item()),
                    pred,
                    actual_changes[-1],
                )
            )

    pred_arr = np.array(pred_changes, dtype=np.float64)
    actual_arr = np.array(actual_changes, dtype=np.float64)

    denom = np.sum((actual_arr - actual_arr.mean()) ** 2)
    r2 = 1.0 - (np.sum((actual_arr - pred_arr) ** 2) / denom) if denom > 0 else float("nan")
    corr = np.corrcoef(pred_arr, actual_arr)[0, 1] if actual_arr.size > 1 else float("nan")
    spear = spearman_corr(pred_arr, actual_arr)
    topk = topk_overlap(pred_arr, actual_arr, args.topk)

    out_dir = args.out_dir or os.path.join(dataset.model_path, "oracle_verify", "new_atom")
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "new_atom_oracle.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("cand_id,px,py,depth,depth_mul,pred_change,actual_change\n")
        for row in meta_rows:
            f.write(",".join(map(str, row)) + "\n")

    summary_path = os.path.join(out_dir, "summary.txt")
    inv_min = float(invdepth[inv_mask].min().item()) if inv_mask.any() else 0.0
    inv_max = float(invdepth[inv_mask].max().item()) if inv_mask.any() else 0.0
    depth_min = float(depth[inv_mask].min().item()) if inv_mask.any() else 0.0
    depth_max = float(depth[inv_mask].max().item()) if inv_mask.any() else 0.0
    scale_min = float(cand_scale.min().item()) if cand_scale.numel() > 0 else 0.0
    scale_max = float(cand_scale.max().item()) if cand_scale.numel() > 0 else 0.0

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"iteration: {args.iteration}\n")
        f.write(f"view_index: {args.view_index}\n")
        f.write(f"num_pixels: {args.num_pixels}\n")
        f.write(f"num_candidates: {n_cand}\n")
        f.write(f"num_samples: {num_samples}\n")
        f.write(f"alpha0: {alpha0}\n")
        f.write(f"epsilon: {eps}\n")
        f.write(f"scale_factor: {args.scale_factor}\n")
        f.write(f"depth_multipliers: {args.depth_multipliers}\n")
        f.write(f"base_loss: {base_loss}\n")
        f.write(f"depth_source: {depth_source}\n")
        f.write(f"pcd_coverage: {pcd_ratio}\n")
        f.write(f"invdepth_min: {inv_min}\n")
        f.write(f"invdepth_max: {inv_max}\n")
        f.write(f"depth_min: {depth_min}\n")
        f.write(f"depth_max: {depth_max}\n")
        f.write(f"cand_scale_min: {scale_min}\n")
        f.write(f"cand_scale_max: {scale_max}\n")
        f.write(f"visible_ratio: {visible_ratio}\n")
        f.write(f"r2: {r2}\n")
        f.write(f"pearson: {corr}\n")
        f.write(f"spearman: {spear}\n")
        f.write(f"topk_overlap: {topk}\n")

    if HAS_MPL and not args.no_plot:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(pred_arr, actual_arr, s=12, alpha=0.7)
        minv = min(pred_arr.min(), actual_arr.min())
        maxv = max(pred_arr.max(), actual_arr.max())
        ax.plot([minv, maxv], [minv, maxv], "r--", linewidth=1)
        ax.set_xlabel("Predicted ΔL (ε * dL/dα)")
        ax.set_ylabel("Actual ΔL")
        ax.set_title(f"New-atom oracle (R2={r2:.4f}, r={corr:.4f})")
        fig.tight_layout()
        fig_path = os.path.join(out_dir, "new_atom_oracle.png")
        fig.savefig(fig_path, dpi=150)

    print(f"New-atom oracle verification finished.")
    print(f"Candidates: {n_cand}, Samples: {num_samples}, epsilon: {eps}")
    print(f"R^2: {r2}, Pearson r: {corr}, Spearman: {spear}, TopK: {topk}")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
