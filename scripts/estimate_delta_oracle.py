#!/usr/bin/env python3
#
# Estimate δ-oracle quality from a candidate pool.
# Outputs CSV + plot of delta(K) and min-g(K) statistics.
#

import os
import sys
import random
import csv
import math
from argparse import ArgumentParser
from collections import defaultdict

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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


def compute_loss(rendered, gt, lambda_dssim):
    ll1 = l1_loss(rendered, gt)
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


def parse_int_list(s):
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    vals = []
    for p in parts:
        if not p:
            continue
        vals.append(int(p))
    return vals


def project_points(view, pts):
    H = int(view.image_height)
    W = int(view.image_width)
    device = pts.device
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
    valid = finite & (z > 0)
    xi = x.round().long()
    yi = y.round().long()
    valid = valid & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    return xi, yi, valid


def compute_gaussian_weights(gaussians, view, err_map, w_mode):
    xyz = gaussians.get_xyz.detach()
    xi, yi, valid = project_points(view, xyz)
    n = xyz.shape[0]
    w = torch.zeros((n,), device=xyz.device, dtype=xyz.dtype)
    if valid.any():
        if w_mode in ("error", "error_opacity"):
            err_vals = err_map[yi[valid], xi[valid]]
            w[valid] = err_vals
        elif w_mode == "opacity":
            op = torch.sigmoid(gaussians._opacity.detach()).squeeze(-1)
            w[valid] = op[valid]
        elif w_mode == "uniform":
            w[valid] = 1.0
        else:
            # default to error
            err_vals = err_map[yi[valid], xi[valid]]
            w[valid] = err_vals
        if w_mode == "error_opacity":
            op = torch.sigmoid(gaussians._opacity.detach()).squeeze(-1)
            w[valid] = w[valid] * op[valid]
    if torch.sum(w) <= 0:
        # Fallback: uniform over valid (or all if none valid).
        if valid.any():
            w[valid] = 1.0
        else:
            w.fill_(1.0)
    return w


def compute_gaussian_weights_multi(gaussians, views, err_maps, w_mode):
    xyz = gaussians.get_xyz.detach()
    n = xyz.shape[0]
    device = xyz.device
    w = torch.zeros((n,), device=device, dtype=xyz.dtype)
    counts = torch.zeros((n,), device=device, dtype=xyz.dtype)
    op = torch.sigmoid(gaussians._opacity.detach()).squeeze(-1)

    for view, err_map in zip(views, err_maps):
        xi, yi, valid = project_points(view, xyz)
        if not valid.any():
            continue
        if w_mode in ("error", "error_opacity"):
            err_vals = err_map[yi[valid], xi[valid]]
            w[valid] += err_vals
        elif w_mode == "uniform":
            w[valid] += 1.0
        elif w_mode == "opacity":
            w[valid] += 1.0
        else:
            err_vals = err_map[yi[valid], xi[valid]]
            w[valid] += err_vals
        counts[valid] += 1.0

    if w_mode in ("error", "error_opacity"):
        mask = counts > 0
        w[mask] = w[mask] / counts[mask]
    elif w_mode == "opacity":
        w = op.clone()
        w[counts == 0] = 0.0
    elif w_mode == "uniform":
        w = (counts > 0).float()

    if w_mode == "error_opacity":
        w = w * op

    if torch.sum(w) <= 0:
        if counts.sum() > 0:
            w = (counts > 0).float()
        else:
            w.fill_(1.0)
    return w


def sample_candidates_from_gaussians(gaussians, parent_idx, radius, scale_factor):
    if parent_idx.numel() == 0:
        return None
    device = gaussians.get_xyz.device
    parent_xyz = gaussians._xyz.detach()[parent_idx]
    parent_scaling = gaussians._scaling.detach()[parent_idx]
    parent_rotation = gaussians._rotation.detach()[parent_idx]
    parent_dc = gaussians._features_dc.detach()[parent_idx]
    parent_rest = gaussians._features_rest.detach()[parent_idx]

    scale = torch.exp(parent_scaling)
    z = torch.randn_like(scale)
    offset = z * scale * float(radius)
    cand_xyz = parent_xyz + offset

    if scale_factor is None:
        scale_factor = 1.0
    scale_factor = max(float(scale_factor), 1e-6)
    cand_scaling = parent_scaling + math.log(scale_factor)

    cand = {
        "xyz": cand_xyz,
        "features_dc": parent_dc,
        "features_rest": parent_rest,
        "scaling": cand_scaling,
        "rotation": parent_rotation,
    }
    return cand


def pixel_to_world(view, px, py, depth, fx, fy, cx, cy):
    x_cam = (px + 0.5 - cx) / fx
    y_cam = (py + 0.5 - cy) / fy
    p_cam = torch.tensor([x_cam * depth, y_cam * depth, depth, 1.0], device="cuda", dtype=torch.float32)
    view_inv = view.world_view_transform.inverse()
    p_world = (view_inv @ p_cam)[:3]
    return p_world


def build_pcd_invdepth(view, gaussians):
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
    parser = ArgumentParser(description="Estimate delta-oracle quality from candidate pool.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    optim = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--view_index", default=0, type=int)
    parser.add_argument("--num_pixels", default=200, type=int)
    parser.add_argument("--depth_multipliers", default="0.5,0.7,0.9,1.0", type=str)
    parser.add_argument("--max_candidates", default=800, type=int)
    parser.add_argument("--alpha0", default=1e-2, type=float)
    parser.add_argument("--scale_factor", default=3.0, type=float)
    parser.add_argument("--scale_min_mult", default=0.25, type=float)
    parser.add_argument("--scale_max_mult", default=4.0, type=float)
    parser.add_argument("--use_dataset_depth", action="store_true")
    parser.add_argument("--use_pcd_depth", action="store_true")
    parser.add_argument("--pcd_fallback_render", action="store_true")
    parser.add_argument("--depth_min", default=0.01, type=float)
    parser.add_argument("--depth_max", default=100.0, type=float)
    parser.add_argument("--k_list", default="50,100,200,400", type=str)
    parser.add_argument("--num_trials", default=20, type=int)
    parser.add_argument("--tau", default=0.0, type=float)
    parser.add_argument("--weight_num_views", default=1, type=int)
    parser.add_argument("--weight_view_stride", default=1, type=int)
    parser.add_argument("--weight_view_indices", default="", type=str)
    parser.add_argument("--proposal_mode", default="pixel_depth", choices=["pixel_depth", "gaussian_perturb"])
    parser.add_argument("--w_mode", default="error", choices=["error", "opacity", "error_opacity", "uniform"])
    parser.add_argument("--kc", default=400, type=int)
    parser.add_argument("--kf", default=400, type=int)
    parser.add_argument("--rc", default=2.0, type=float)
    parser.add_argument("--rf", default=0.5, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--out_dir", default="", type=str)
    parser.add_argument("--no_plot", action="store_true")

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

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    views = scene.getTrainCameras() if args.split == "train" else scene.getTestCameras()
    if not views:
        raise RuntimeError(f"No cameras found for split={args.split}")
    if args.view_index < 0 or args.view_index >= len(views):
        raise RuntimeError(f"Invalid --view_index {args.view_index} (0..{len(views)-1})")
    view = views[args.view_index]

    render_pkg = render(view, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp)
    image = render_pkg["render"]
    if view.alpha_mask is not None:
        image = image * view.alpha_mask.cuda()
    gt_image = view.original_image.cuda()

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

    # Error map for candidate selection / weighting (base view).
    err = torch.mean(torch.abs(image - gt_image), dim=0)
    if view.alpha_mask is not None:
        err = err * view.alpha_mask[0].cuda()

    # Default: use base view only.
    weight_view_indices = [args.view_index]
    weight_views = [view]
    weight_err_maps = [err]
    # Multi-view weighting only needed for gaussian_perturb.
    if args.proposal_mode != "pixel_depth":
        weight_view_indices = parse_int_list(args.weight_view_indices)
        if not weight_view_indices:
            weight_view_indices = [args.view_index + i * args.weight_view_stride for i in range(int(args.weight_num_views))]
        weight_view_indices = [i for i in weight_view_indices if 0 <= i < len(views)]
        if not weight_view_indices:
            weight_view_indices = [args.view_index]

        weight_views = []
        weight_err_maps = []
        for idx in weight_view_indices:
            v = views[idx]
            if idx == args.view_index:
                weight_views.append(v)
                weight_err_maps.append(err)
                continue
            with torch.no_grad():
                render_pkg_w = render(v, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp)
                img_w = render_pkg_w["render"]
                if v.alpha_mask is not None:
                    img_w = img_w * v.alpha_mask.cuda()
                gt_w = v.original_image.cuda()
                err_w = torch.mean(torch.abs(img_w - gt_w), dim=0)
                if v.alpha_mask is not None:
                    err_w = err_w * v.alpha_mask[0].cuda()
                weight_views.append(v)
                weight_err_maps.append(err_w)

    cand_xyz = None
    cand_features_dc = None
    cand_features_rest = None
    cand_scaling = None
    cand_rotation = None
    n_cand = 0

    if args.proposal_mode == "pixel_depth":
        depth = torch.zeros_like(invdepth)
        inv_mask = invdepth > 1e-6
        depth[inv_mask] = 1.0 / (invdepth[inv_mask] + 1e-8)

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
        pix = coords[idx]

        depth_multipliers = parse_float_list(args.depth_multipliers)
        if not depth_multipliers:
            depth_multipliers = [1.0]

        fx = fov2focal(view.FoVx, view.image_width)
        fy = fov2focal(view.FoVy, view.image_height)
        cx = view.image_width * 0.5
        cy = view.image_height * 0.5

        cand_xyz_list = []
        cand_dc_list = []
        cand_depth_list = []
        for yx in pix:
            py = int(yx[0].item())
            px = int(yx[1].item())
            d = float(depth[py, px].item())
            if d <= 0:
                continue
            rgb = gt_image[:, py, px].unsqueeze(0)
            dc = RGB2SH(rgb).squeeze(0)
            for m in depth_multipliers:
                dd = d * float(m)
                if dd <= 0:
                    continue
                p_world = pixel_to_world(view, px, py, dd, fx, fy, cx, cy)
                cand_xyz_list.append(p_world)
                cand_dc_list.append(dc)
                cand_depth_list.append(dd)

        if not cand_xyz_list:
            raise RuntimeError("No candidates built.")

        cand_xyz = torch.stack(cand_xyz_list, dim=0)
        cand_dc = torch.stack(cand_dc_list, dim=0)
        cand_depth = torch.tensor(cand_depth_list, device="cuda", dtype=torch.float32)

        if cand_xyz.shape[0] > args.max_candidates:
            perm = torch.randperm(cand_xyz.shape[0], device="cuda")[: args.max_candidates]
            cand_xyz = cand_xyz[perm]
            cand_dc = cand_dc[perm]
            cand_depth = cand_depth[perm]

        n_cand = cand_xyz.shape[0]
        cand_features_dc = cand_dc.view(n_cand, 3, 1).transpose(1, 2).contiguous()
        n_sh = (gaussians.max_sh_degree + 1) ** 2
        n_rest = n_sh - 1
        if n_rest > 0:
            cand_features_rest = torch.zeros((n_cand, n_rest, 3), device="cuda")
        else:
            cand_features_rest = torch.zeros((n_cand, 0, 3), device="cuda")

        base_scale = gaussians.get_scaling.detach()
        med_scale = torch.median(base_scale, dim=0).values
        min_scale = med_scale * float(args.scale_min_mult)
        max_scale = med_scale * float(args.scale_max_mult)
        cand_sigma_x = cand_depth / fx
        cand_sigma_y = cand_depth / fy
        cand_sigma = torch.maximum(cand_sigma_x, cand_sigma_y) * float(args.scale_factor)
        cand_sigma = torch.clamp(cand_sigma, min=min_scale.min().item(), max=max_scale.max().item())
        cand_scale = torch.stack([cand_sigma, cand_sigma, cand_sigma], dim=1)
        cand_scaling = torch.log(cand_scale)

        cand_rotation = torch.zeros((n_cand, 4), device="cuda")
        cand_rotation[:, 0] = 1.0
    else:
        # Proposal by perturbing existing Gaussians (no depth hypotheses).
        weight_tag = ",".join(str(i) for i in weight_view_indices)
        depth_source = f"proposal:{args.proposal_mode}/{args.w_mode}/views={weight_tag}"
        weights = compute_gaussian_weights_multi(gaussians, weight_views, weight_err_maps, args.w_mode)
        weights = weights / torch.sum(weights)

        kc = max(int(args.kc), 0)
        kf = max(int(args.kf), 0)
        if kc + kf <= 0:
            kc = int(args.max_candidates)
            kf = 0

        parent_idx_c = torch.multinomial(weights, kc, replacement=True) if kc > 0 else torch.empty((0,), device=weights.device, dtype=torch.long)
        parent_idx_f = torch.multinomial(weights, kf, replacement=True) if kf > 0 else torch.empty((0,), device=weights.device, dtype=torch.long)

        cand_parts = []
        if kc > 0:
            cand_parts.append(sample_candidates_from_gaussians(gaussians, parent_idx_c, args.rc, args.scale_factor))
        if kf > 0:
            cand_parts.append(sample_candidates_from_gaussians(gaussians, parent_idx_f, args.rf, args.scale_factor))

        cand_xyz_list = []
        cand_dc_list = []
        cand_rest_list = []
        cand_scaling_list = []
        cand_rotation_list = []
        for part in cand_parts:
            if part is None:
                continue
            cand_xyz_list.append(part["xyz"])
            cand_dc_list.append(part["features_dc"])
            cand_rest_list.append(part["features_rest"])
            cand_scaling_list.append(part["scaling"])
            cand_rotation_list.append(part["rotation"])

        if not cand_xyz_list:
            raise RuntimeError("No candidates built from Gaussian perturbation.")

        cand_xyz = torch.cat(cand_xyz_list, dim=0)
        cand_features_dc = torch.cat(cand_dc_list, dim=0)
        cand_features_rest = torch.cat(cand_rest_list, dim=0)
        cand_scaling = torch.cat(cand_scaling_list, dim=0)
        cand_rotation = torch.cat(cand_rotation_list, dim=0)
        n_cand = cand_xyz.shape[0]

    alpha0 = float(args.alpha0)
    alpha0 = min(max(alpha0, 1e-6), 1.0 - 1e-6)
    logit0 = inverse_sigmoid(torch.tensor(alpha0, device="cuda"))
    cand_opacity = logit0.view(1, 1).repeat(n_cand, 1)

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
    if dataset.train_test_exp and hasattr(gaussians, "_exposure"):
        combined.exposure_mapping = getattr(gaussians, "exposure_mapping", None)
        combined.pretrained_exposures = getattr(gaussians, "pretrained_exposures", None)
        combined._exposure = gaussians._exposure

    render_pkg2 = render(view, combined, pipe, background, use_trained_exp=dataset.train_test_exp)
    image2 = render_pkg2["render"]
    if view.alpha_mask is not None:
        image2 = image2 * view.alpha_mask.cuda()
    loss = compute_loss(image2, gt_image, opt.lambda_dssim)
    loss.backward()

    if combined._opacity.grad is None:
        raise RuntimeError("Opacity gradients are None for combined model.")

    alpha = torch.sigmoid(combined._opacity.detach())
    dlogit = combined._opacity.grad.detach()
    dL_dalpha = dlogit / (alpha * (1.0 - alpha) + 1e-6)

    base_count = gaussians._xyz.shape[0]
    g = dL_dalpha[base_count:].squeeze(-1).detach().cpu().numpy()

    radii = render_pkg2.get("radii", None)
    if radii is not None:
        cand_radii = radii[base_count:]
        visible_ratio = float((cand_radii > 0).float().mean().item()) if cand_radii.numel() > 0 else 0.0
    else:
        visible_ratio = float("nan")

    # Delta estimation
    k_list = parse_int_list(args.k_list)
    if not k_list:
        k_list = [50, 100, 200, 400]
    num_trials = max(1, int(args.num_trials))

    results = []
    rng = np.random.RandomState(args.seed)
    n = g.shape[0]

    for K in k_list:
        K = int(K)
        if K <= 0:
            continue
        deltas = []
        mins = []
        for _ in range(num_trials):
            if n >= 2 * K:
                perm = rng.permutation(n)[: 2 * K]
                a = perm[:K]
                b = perm[K:]
            else:
                a = rng.choice(n, size=K, replace=True)
                b = rng.choice(n, size=K, replace=True)
            m1 = float(np.min(g[a]))
            m2 = float(np.min(g[b]))
            m12 = float(min(m1, m2))
            deltas.append(max(m1, m2) - m12)
            mins.append(m12)
        results.append({
            "K": K,
            "delta_mean": float(np.mean(deltas)),
            "delta_std": float(np.std(deltas)),
            "min_g_mean": float(np.mean(mins)),
            "min_g_std": float(np.std(mins)),
        })

    tail_frac = float(np.mean(g < float(args.tau)))
    g_min = float(np.min(g))
    g_mean = float(np.mean(g))

    out_dir = args.out_dir or os.path.join(dataset.model_path, "oracle_verify", "delta_oracle")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "delta_oracle_summary.csv")

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        f.write("# per_run\n")
        writer = csv.writer(f)
        writer.writerow([
            "iteration","view_index","alpha0","scale_factor","depth_multipliers","depth_source",
            "pcd_coverage","visible_ratio","g_min","g_mean","tail_tau","tail_frac","num_candidates",
            "proposal_mode","w_mode","rc","rf","kc","kf"
        ])
        pcd_ratio = float((invdepth_pcd > 0).float().mean().item()) if invdepth_pcd is not None else float("nan")
        writer.writerow([
            args.iteration, args.view_index, alpha0, args.scale_factor, args.depth_multipliers, depth_source,
            pcd_ratio, visible_ratio, g_min, g_mean, args.tau, tail_frac, n_cand,
            args.proposal_mode, args.w_mode, args.rc, args.rf, args.kc, args.kf
        ])

        f.write("\n# by_K\n")
        writer.writerow(["K","delta_mean","delta_std","min_g_mean","min_g_std"])
        for r in results:
            writer.writerow([r["K"], r["delta_mean"], r["delta_std"], r["min_g_mean"], r["min_g_std"]])

    if HAS_MPL and not args.no_plot:
        ks = [r["K"] for r in results]
        dmean = [r["delta_mean"] for r in results]
        dstd = [r["delta_std"] for r in results]
        mmean = [r["min_g_mean"] for r in results]
        mstd = [r["min_g_std"] for r in results]

        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        axs[0].errorbar(ks, dmean, yerr=dstd, marker="o")
        axs[0].set_title("δ(K) vs K")
        axs[0].set_xlabel("K")
        axs[0].set_ylabel("delta")

        axs[1].errorbar(ks, mmean, yerr=mstd, marker="o")
        axs[1].set_title("min g(K) vs K")
        axs[1].set_xlabel("K")
        axs[1].set_ylabel("min g")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "delta_oracle_summary.png"), dpi=150)

    print(f"Wrote {out_csv}")
    if HAS_MPL and not args.no_plot:
        print(f"Wrote {os.path.join(out_dir, 'delta_oracle_summary.png')}")


if __name__ == "__main__":
    main()
