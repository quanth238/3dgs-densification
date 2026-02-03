#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
import random
import torch.nn as nn
import math
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func, inverse_sigmoid
from utils.sh_utils import RGB2SH
from utils.triangulation_utils import (
    get_intrinsics,
    match_pixel_epipolar,
    ray_from_pixel,
    closest_point_between_rays,
)
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def _select_eval_views(scene, split, num_views, seed):
    views = scene.getTestCameras() if split == "test" else scene.getTrainCameras()
    split_used = split
    if not views and split == "test":
        # Fallback if test cameras are missing
        views = scene.getTrainCameras()
        split_used = "train"
        print("Warning: no test cameras found; using train split for ab_log.")
    if not views:
        return [], [], split_used
    num_views = min(num_views, len(views))
    rng = random.Random(seed)
    indices = list(range(len(views)))
    rng.shuffle(indices)
    chosen = [views[i] for i in indices[:num_views]]
    return chosen, indices[:num_views], split_used

def _eval_views_metrics(views, gaussians, pipe, background, use_trained_exp, lpips_model=None):
    if not views:
        return float("nan"), float("nan"), float("nan")
    psnrs = []
    l1s = []
    lpipss = []
    with torch.no_grad():
        for view in views:
            render_pkg = render(view, gaussians, pipe, background, use_trained_exp=use_trained_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
            image = render_pkg["render"]
            if view.alpha_mask is not None:
                image = image * view.alpha_mask.cuda()
            gt_image = view.original_image.cuda()
            psnrs.append(psnr(image, gt_image).mean().item())
            l1s.append(l1_loss(image, gt_image).mean().item())
            if lpips_model is not None:
                lp = lpips_model(image.unsqueeze(0), gt_image.unsqueeze(0))
                lpipss.append(lp.item())
    mean_psnr = float(sum(psnrs) / len(psnrs))
    mean_l1 = float(sum(l1s) / len(l1s))
    mean_lpips = float(sum(lpipss) / len(lpipss)) if lpipss else float("nan")
    return mean_psnr, mean_l1, mean_lpips


def _parse_int_list(s):
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    out = []
    for p in parts:
        if p:
            out.append(int(p))
    return out


def _compute_loss_simple(rendered, gt, lambda_dssim):
    ll1 = l1_loss(rendered, gt)
    ssim_value = ssim(rendered, gt)
    return (1.0 - lambda_dssim) * ll1 + lambda_dssim * (1.0 - ssim_value)


def _select_view_indices(base_idx, total, num_views, stride, explicit_list):
    if total <= 0:
        return []
    if explicit_list:
        return [i for i in explicit_list if 0 <= i < total and i != base_idx]
    if num_views <= 0:
        return []
    stride = max(1, int(stride))
    indices = []
    for k in range(1, num_views + 1):
        idx = (base_idx + k * stride) % total
        if idx == base_idx:
            continue
        if idx not in indices:
            indices.append(idx)
    return indices


def _compute_gaussian_weights_responsibility(gaussians, views, pipe, background, lambda_dssim, use_trained_exp, w_mode):
    xyz = gaussians.get_xyz.detach()
    n = xyz.shape[0]
    device = xyz.device
    w = torch.zeros((n,), device=device, dtype=xyz.dtype)
    opacity_logit = gaussians._opacity
    if opacity_logit.grad is not None:
        opacity_logit.grad = None

    # Ensure gradients are tracked even if caller is inside no_grad.
    with torch.enable_grad():
        for v in views:
            render_pkg = render(
                v,
                gaussians,
                pipe,
                background,
                use_trained_exp=use_trained_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )
            img = render_pkg["render"]
            if v.alpha_mask is not None:
                img = img * v.alpha_mask.cuda()
            gt = v.original_image.cuda()
            loss = _compute_loss_simple(img, gt, lambda_dssim)
            grad = torch.autograd.grad(
                loss,
                opacity_logit,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
            if grad is None:
                continue
            alpha = torch.sigmoid(opacity_logit.detach())
            dL_dalpha = grad / (alpha * (1.0 - alpha) + 1e-6)
            w += torch.abs(dL_dalpha.squeeze(-1))

    if w_mode == "resp_opacity":
        w = w * torch.sigmoid(opacity_logit.detach()).squeeze(-1)

    if torch.sum(w) <= 0:
        w.fill_(1.0)
    return w


def _build_triangulate_candidates(view, gt_image, err_map, views, triang_view_indices, gaussians, opt):
    mask = torch.ones_like(err_map, dtype=torch.bool)
    if view.alpha_mask is not None:
        mask = mask & (view.alpha_mask[0].cuda() > 0.5)
    coords = mask.nonzero(as_tuple=False)
    if coords.numel() == 0:
        return None
    scores = err_map[mask]
    topk = min(int(opt.triang_num_pixels), scores.numel())
    vals, idx = torch.topk(scores, k=topk, largest=True)
    pix = coords[idx]

    if not triang_view_indices:
        return None

    fx, fy, cx, cy = get_intrinsics(view)

    base_scale = gaussians.get_scaling.detach()
    med_scale = torch.median(base_scale, dim=0).values
    min_scale = med_scale * float(opt.triang_scale_min_mult)
    max_scale = med_scale * float(opt.triang_scale_max_mult)
    if opt.triang_max_ray_dist > 0:
        max_ray_dist = float(opt.triang_max_ray_dist)
    else:
        max_ray_dist = float(torch.max(med_scale).item() * float(opt.triang_scale_factor) * 2.0)

    cand_xyz_list = []
    cand_dc_list = []
    cand_depth_list = []

    img_base = gt_image
    for yx in pix:
        py = int(yx[0].item())
        px = int(yx[1].item())
        rgb = img_base[:, py, px].unsqueeze(0)
        dc = RGB2SH(rgb).squeeze(0)

        matched = False
        for vidx in triang_view_indices:
            v2 = views[vidx]
            img2 = v2.original_image.cuda()
            match = match_pixel_epipolar(
                view, v2, img_base, img2, px, py,
                float(opt.triang_depth_min), float(opt.triang_depth_max),
                int(opt.triang_downscale),
                int(opt.triang_patch),
                int(opt.triang_max_samples),
                float(opt.triang_min_ncc),
            )
            if match is None:
                continue
            px2, py2, _ncc = match
            o1, d1 = ray_from_pixel(view, px, py)
            o2, d2 = ray_from_pixel(v2, px2, py2)
            res = closest_point_between_rays(o1, d1, o2, d2)
            if res is None:
                continue
            p_world, s, t, dist = res
            if s <= 0 or t <= 0:
                continue
            if dist > max_ray_dist:
                continue
            if s < float(opt.triang_depth_min) or s > float(opt.triang_depth_max):
                continue
            cand_xyz_list.append(p_world)
            cand_dc_list.append(dc)
            cand_depth_list.append(float(s))
            matched = True
            break
        if not matched:
            continue
        if len(cand_xyz_list) >= int(opt.triang_max_candidates):
            break

    if not cand_xyz_list:
        return None

    cand_xyz = torch.stack(cand_xyz_list, dim=0)
    cand_dc = torch.stack(cand_dc_list, dim=0)
    cand_depth = torch.tensor(cand_depth_list, device="cuda", dtype=torch.float32)
    n_cand = cand_xyz.shape[0]

    cand_features_dc = cand_dc.view(n_cand, 3, 1).transpose(1, 2).contiguous()
    n_sh = (gaussians.max_sh_degree + 1) ** 2
    n_rest = n_sh - 1
    if n_rest > 0:
        cand_features_rest = torch.zeros((n_cand, n_rest, 3), device="cuda")
    else:
        cand_features_rest = torch.zeros((n_cand, 0, 3), device="cuda")

    cand_sigma_x = cand_depth / fx
    cand_sigma_y = cand_depth / fy
    cand_sigma = torch.maximum(cand_sigma_x, cand_sigma_y) * float(opt.triang_scale_factor)
    cand_sigma = torch.clamp(cand_sigma, min=min_scale.min().item(), max=max_scale.max().item())
    cand_scale = torch.stack([cand_sigma, cand_sigma, cand_sigma], dim=1)
    cand_scaling = torch.log(cand_scale)

    cand_rotation = torch.zeros((n_cand, 4), device="cuda")
    cand_rotation[:, 0] = 1.0

    return {
        "xyz": cand_xyz,
        "features_dc": cand_features_dc,
        "features_rest": cand_features_rest,
        "scaling": cand_scaling,
        "rotation": cand_rotation,
    }


def _score_candidates_dalpha(view, gt_image, gaussians, cand, opt, pipe, background, use_trained_exp):
    if cand is None:
        return None, float("nan")
    cand_xyz = cand["xyz"]
    cand_features_dc = cand["features_dc"]
    cand_features_rest = cand["features_rest"]
    cand_scaling = cand["scaling"]
    cand_rotation = cand["rotation"]
    n_cand = cand_xyz.shape[0]
    if n_cand == 0:
        return None, float("nan")

    alpha0 = float(opt.triang_alpha0)
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
    if use_trained_exp and hasattr(gaussians, "_exposure"):
        combined.exposure_mapping = getattr(gaussians, "exposure_mapping", None)
        combined.pretrained_exposures = getattr(gaussians, "pretrained_exposures", None)
        combined._exposure = gaussians._exposure

    render_pkg = render(view, combined, pipe, background, use_trained_exp=use_trained_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
    image = render_pkg["render"]
    if view.alpha_mask is not None:
        image = image * view.alpha_mask.cuda()
    loss = _compute_loss_simple(image, gt_image, opt.lambda_dssim)
    loss.backward()

    if combined._opacity.grad is None:
        return None, float("nan")
    alpha = torch.sigmoid(combined._opacity.detach())
    dlogit = combined._opacity.grad.detach()
    dL_dalpha = dlogit / (alpha * (1.0 - alpha) + 1e-6)
    base_count = gaussians._xyz.shape[0]
    g_t = dL_dalpha[base_count:].squeeze(-1).detach()

    radii = render_pkg.get("radii", None)
    if radii is not None:
        cand_radii = radii[base_count:]
        visible_ratio = float((cand_radii > 0).float().mean().item()) if cand_radii.numel() > 0 else 0.0
    else:
        visible_ratio = float("nan")

    return g_t, visible_ratio


def _reset_optimizer_state(optimizer, idx):
    if optimizer is None:
        return
    if idx.numel() == 0:
        return
    idx = idx.long()
    for group in optimizer.param_groups:
        if not group.get("params"):
            continue
        p = group["params"][0]
        state = optimizer.state.get(p, None)
        if not state:
            continue
        for key in ("exp_avg", "exp_avg_sq"):
            if key in state and state[key].shape[0] > idx.max().item():
                state[key][idx] = 0


def _relocate_gaussians(gaussians, dst_idx, cand, cand_idx, opacity_logit):
    if dst_idx.numel() == 0:
        return
    with torch.no_grad():
        gaussians._xyz.data[dst_idx] = cand["xyz"][cand_idx]
        gaussians._features_dc.data[dst_idx] = cand["features_dc"][cand_idx]
        gaussians._features_rest.data[dst_idx] = cand["features_rest"][cand_idx]
        gaussians._scaling.data[dst_idx] = cand["scaling"][cand_idx]
        gaussians._rotation.data[dst_idx] = cand["rotation"][cand_idx]
        gaussians._opacity.data[dst_idx] = opacity_logit[cand_idx]
        if hasattr(gaussians, "max_radii2D"):
            gaussians.max_radii2D[dst_idx] = 0


def _init_ab_logger(opt, scene, dataset):
    if not opt.ab_log:
        return None
    log_dir = opt.ab_log_dir if opt.ab_log_dir else os.path.join(scene.model_path, "ab_logs", opt.densify_mode)
    os.makedirs(log_dir, exist_ok=True)
    csv_path = os.path.join(log_dir, "ab_metrics.csv")
    eval_views, eval_indices, split_used = _select_eval_views(scene, opt.ab_eval_split, opt.ab_eval_views, opt.ab_seed)

    lpips_model = None
    if opt.ab_lpips:
        from lpipsPyTorch.modules.lpips import LPIPS
        lpips_model = LPIPS(net_type=opt.ab_lpips_net).cuda()

    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(f"# densify_mode={opt.densify_mode}\n")
            f.write(f"# ab_eval_split={opt.ab_eval_split}, split_used={split_used}, ab_eval_views={opt.ab_eval_views}, ab_seed={opt.ab_seed}\n")
            f.write(f"# eval_indices={eval_indices}\n")
            f.write("iteration,num_points,mean_psnr,mean_l1,mean_lpips\n")

    return {
        "csv_path": csv_path,
        "eval_views": eval_views,
        "lpips_model": lpips_model
    }

def _ab_log_step(ab_state, iteration, gaussians, pipe, background, dataset):
    if ab_state is None:
        return
    mean_psnr, mean_l1, mean_lpips = _eval_views_metrics(
        ab_state["eval_views"], gaussians, pipe, background, dataset.train_test_exp, ab_state["lpips_model"]
    )
    num_points = int(gaussians.get_xyz.shape[0])
    with open(ab_state["csv_path"], "a", encoding="utf-8") as f:
        f.write(f"{iteration},{num_points},{mean_psnr},{mean_l1},{mean_lpips}\n")

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    ab_state = _init_ab_logger(opt, scene, dataset)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization
        Ll1depth_pure = 0.0
        if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
            invDepth = render_pkg["depth"]
            mono_invdepth = viewpoint_cam.invdepthmap.cuda()
            depth_mask = viewpoint_cam.depth_mask.cuda()

            Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
            Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
            loss += Ll1depth
            Ll1depth = Ll1depth.item()
        else:
            Ll1depth = 0

        loss.backward()

        dL_dalpha = None
        if opt.densify_mode == "opacity":
            with torch.no_grad():
                alpha = gaussians.get_opacity
                dlogit = gaussians._opacity.grad
                if dlogit is not None:
                    dL_dalpha = dlogit / (alpha * (1.0 - alpha) + 1e-6)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                if opt.densify_mode == "opacity":
                    if dL_dalpha is not None:
                        gaussians.add_opacity_densification_stats(dL_dalpha, visibility_filter, use_abs=opt.densify_opacity_use_abs)
                elif opt.densify_mode == "xyz":
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    if opt.densify_mode == "triangulate":
                        err_map = torch.mean(torch.abs(image - gt_image), dim=0)
                        if viewpoint_cam.alpha_mask is not None:
                            err_map = err_map * viewpoint_cam.alpha_mask[0].cuda()

                        train_views = scene.getTrainCameras()
                        tri_indices = _select_view_indices(
                            vind,
                            len(train_views),
                            int(opt.triang_num_views),
                            int(opt.triang_view_stride),
                            _parse_int_list(opt.triang_view_indices),
                        )
                        cand = _build_triangulate_candidates(
                            viewpoint_cam,
                            gt_image,
                            err_map,
                            train_views,
                            tri_indices,
                            gaussians,
                            opt,
                        )
                        if cand is not None:
                            with torch.enable_grad():
                                g_scores, _ = _score_candidates_dalpha(
                                    viewpoint_cam,
                                    gt_image,
                                    gaussians,
                                    cand,
                                    opt,
                                    pipe,
                                    background,
                                    dataset.train_test_exp,
                                )
                            if g_scores is not None and g_scores.numel() > 0:
                                n_cand = g_scores.shape[0]
                                n_gauss = gaussians._xyz.shape[0]
                                reloc_num = int(opt.triang_relocate_num) if int(opt.triang_relocate_num) > 0 else int(float(opt.triang_relocate_frac) * n_gauss)
                                insert_num = int(opt.triang_insert_num) if int(opt.triang_insert_num) > 0 else int(float(opt.triang_insert_frac) * n_gauss)
                                if not getattr(opt, "quiet", False):
                                    print(
                                        f"[triangulate] iter={iteration} n_gauss={n_gauss} "
                                        f"n_cand={n_cand} reloc_num={reloc_num} insert_num={insert_num}"
                                    )
                                if reloc_num + insert_num > 0:
                                    total_need = min(n_cand, reloc_num + insert_num)
                                    # Most negative g = best candidates
                                    cand_order = torch.argsort(g_scores)
                                    chosen = cand_order[:total_need]

                                    alpha0 = float(opt.triang_alpha0)
                                    alpha0 = min(max(alpha0, 1e-6), 1.0 - 1e-6)
                                    cand_opacity = inverse_sigmoid(torch.tensor(alpha0, device="cuda")).view(1, 1).repeat(n_cand, 1)

                                    if reloc_num > 0:
                                        reloc_num = min(reloc_num, chosen.numel())
                                        cand_reloc_idx = chosen[:reloc_num]
                                        if opt.triang_relocate_by in ("resp", "resp_opacity"):
                                            weight_indices = _select_view_indices(
                                                vind,
                                                len(train_views),
                                                int(opt.triang_weight_num_views),
                                                int(opt.triang_weight_view_stride),
                                                [],
                                            )
                                            weight_views = [train_views[i] for i in weight_indices]
                                            weights = _compute_gaussian_weights_responsibility(
                                                gaussians,
                                                weight_views,
                                                pipe,
                                                background,
                                                opt.lambda_dssim,
                                                dataset.train_test_exp,
                                                opt.triang_relocate_by,
                                            )
                                            dead_idx = torch.topk(weights, reloc_num, largest=False).indices
                                        else:
                                            op = torch.sigmoid(gaussians._opacity.detach()).squeeze(-1)
                                            dead_idx = torch.topk(op, reloc_num, largest=False).indices
                                        _relocate_gaussians(gaussians, dead_idx, cand, cand_reloc_idx, cand_opacity)
                                        _reset_optimizer_state(gaussians.optimizer, dead_idx)

                                    if insert_num > 0:
                                        insert_num = min(insert_num, chosen.numel() - reloc_num)
                                        if insert_num > 0:
                                            cand_ins_idx = chosen[reloc_num:reloc_num + insert_num]
                                            new_tmp_radii = torch.zeros((cand_ins_idx.numel(),), device="cuda")
                                            if not hasattr(gaussians, "tmp_radii") or gaussians.tmp_radii is None:
                                                gaussians.tmp_radii = torch.zeros((n_gauss,), device="cuda")
                                            gaussians.densification_postfix(
                                                cand["xyz"][cand_ins_idx],
                                                cand["features_dc"][cand_ins_idx],
                                                cand["features_rest"][cand_ins_idx],
                                                cand_opacity[cand_ins_idx],
                                                cand["scaling"][cand_ins_idx],
                                                cand["rotation"][cand_ins_idx],
                                                new_tmp_radii,
                                            )
                                elif not getattr(opt, "quiet", False):
                                    print(f"[triangulate] iter={iteration} skipping (reloc+insert=0)")
                            elif not getattr(opt, "quiet", False):
                                print(f"[triangulate] iter={iteration} no candidate scores")
                        elif not getattr(opt, "quiet", False):
                            print(f"[triangulate] iter={iteration} no candidates built")
                    else:
                        size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                        grad_threshold = opt.densify_grad_threshold if opt.densify_mode == "xyz" else opt.densify_opacity_threshold
                        gaussians.densify_and_prune(grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii, mode=opt.densify_mode)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if opt.ab_log and iteration % opt.ab_log_interval == 0:
                _ab_log_step(ab_state, iteration, gaussians, pipe, background, dataset)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                gaussians.exposure_optimizer.zero_grad(set_to_none = True)
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
