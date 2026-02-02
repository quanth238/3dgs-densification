#
# Verify opacity-gradient oracle via directional-derivative test.
# Saves scatter plot + CSV of predicted vs actual loss change.
#

import os
import random
from argparse import ArgumentParser

import numpy as np
import torch

from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from gaussian_renderer import render, GaussianModel
from scene import Scene
from utils.general_utils import inverse_sigmoid, safe_state
from utils.loss_utils import l1_loss, ssim

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


def clear_grads(gaussians):
    for t in [
        gaussians._xyz,
        gaussians._features_dc,
        gaussians._features_rest,
        gaussians._opacity,
        gaussians._scaling,
        gaussians._rotation,
    ]:
        if t.grad is not None:
            t.grad = None


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


def main():
    parser = ArgumentParser(description="Verify opacity-gradient oracle via directional derivative test.")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    optim = OptimizationParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--epsilon", default=1e-3, type=float)
    parser.add_argument("--eps_list", default="", type=str)
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--view_index", default=0, type=int)
    parser.add_argument("--view_indices", default="", type=str)
    parser.add_argument("--num_views", default=1, type=int)
    parser.add_argument("--view_stride", default=1, type=int)
    parser.add_argument("--include_invisible", action="store_true")
    parser.add_argument("--central_diff", action="store_true")
    parser.add_argument("--alpha_min", default=0.0, type=float)
    parser.add_argument("--alpha_max", default=1.0, type=float)
    parser.add_argument("--topk", default=20, type=int)
    parser.add_argument("--out_dir", default="", type=str)
    parser.add_argument("--seed", default=0, type=int)
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

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    safe_state(False)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    opt = optim.extract(args)

    # Resolve which iteration to load (avoid crashing when point_cloud is missing).
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
    n_views_total = len(views)

    eps_list = parse_float_list(args.eps_list)
    if not eps_list:
        eps_list = [float(args.epsilon)]

    view_indices = parse_int_list(args.view_indices)
    if view_indices:
        bad = [i for i in view_indices if i < 0 or i >= n_views_total]
        if bad:
            raise RuntimeError(f"Invalid view indices: {bad} (0..{n_views_total-1})")
    else:
        start = int(args.view_index)
        stride = max(1, int(args.view_stride))
        count = max(1, int(args.num_views))
        view_indices = [start + i * stride for i in range(count)]
        bad = [i for i in view_indices if i < 0 or i >= n_views_total]
        if bad:
            raise RuntimeError(f"View indices out of range: {bad} (0..{n_views_total-1})")

    out_dir = args.out_dir or os.path.join(dataset.model_path, "oracle_verify")
    os.makedirs(out_dir, exist_ok=True)

    orig_opacity = gaussians._opacity.data.clone()
    sweep_results = []

    def run_single(view_idx, eps):
        view = views[view_idx]
        clear_grads(gaussians)

        render_pkg = render(view, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp)
        image = render_pkg["render"]
        if view.alpha_mask is not None:
            image = image * view.alpha_mask.cuda()
        gt_image = view.original_image.cuda()

        loss = compute_loss(image, gt_image, opt.lambda_dssim)
        loss.backward()

        if gaussians._opacity.grad is None:
            raise RuntimeError("Opacity gradients are None. Ensure requires_grad and backward() ran.")

        base_loss = loss.item()
        alpha = gaussians.get_opacity.detach()
        dlogit = gaussians._opacity.grad.detach()
        dL_dalpha = dlogit / (alpha * (1.0 - alpha) + 1e-6)

        # Candidate indices
        if args.include_invisible:
            candidates = torch.arange(alpha.shape[0], device="cuda")
        else:
            vis = render_pkg["visibility_filter"].squeeze()
            if vis.numel() == 0:
                raise RuntimeError("No visible Gaussians in this view.")
            candidates = vis if vis.ndim == 1 else vis[:, 0]

        # Alpha filter to avoid saturation effects (especially for central diff)
        if args.alpha_min > 0.0 or args.alpha_max < 1.0:
            cand_alpha = alpha[candidates].squeeze(-1)
            mask = (cand_alpha >= args.alpha_min) & (cand_alpha <= args.alpha_max)
            candidates = candidates[mask]
            if candidates.numel() == 0:
                raise RuntimeError("No candidates left after alpha filter.")

        num_samples = min(args.num_samples, candidates.shape[0])
        perm = torch.randperm(candidates.shape[0], device="cuda")[:num_samples]
        sample_ids = candidates[perm].tolist()

        pred_changes = []
        actual_changes = []

        with torch.no_grad():
            for idx in sample_ids:
                pred = (eps * dL_dalpha[idx]).item()
                pred_changes.append(pred)

                cur_alpha = alpha[idx].item()
                if args.central_diff:
                    alpha_plus = min(max(cur_alpha + eps, 1e-4), 1.0 - 1e-4)
                    alpha_minus = min(max(cur_alpha - eps, 1e-4), 1.0 - 1e-4)

                    gaussians._opacity.data[idx] = inverse_sigmoid(torch.tensor(alpha_plus, device="cuda"))
                    render_pkg_p = render(view, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp)
                    image_p = render_pkg_p["render"]
                    if view.alpha_mask is not None:
                        image_p = image_p * view.alpha_mask.cuda()
                    loss_p = compute_loss(image_p, gt_image, opt.lambda_dssim).item()

                    gaussians._opacity.data[idx] = inverse_sigmoid(torch.tensor(alpha_minus, device="cuda"))
                    render_pkg_m = render(view, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp)
                    image_m = render_pkg_m["render"]
                    if view.alpha_mask is not None:
                        image_m = image_m * view.alpha_mask.cuda()
                    loss_m = compute_loss(image_m, gt_image, opt.lambda_dssim).item()

                    actual = 0.5 * (loss_p - loss_m)
                    actual_changes.append(actual)
                else:
                    new_alpha = min(max(cur_alpha + eps, 1e-4), 1.0 - 1e-4)
                    gaussians._opacity.data[idx] = inverse_sigmoid(torch.tensor(new_alpha, device="cuda"))

                    render_pkg2 = render(view, gaussians, pipe, background, use_trained_exp=dataset.train_test_exp)
                    image2 = render_pkg2["render"]
                    if view.alpha_mask is not None:
                        image2 = image2 * view.alpha_mask.cuda()
                    loss2 = compute_loss(image2, gt_image, opt.lambda_dssim).item()
                    actual = loss2 - base_loss
                    actual_changes.append(actual)

                gaussians._opacity.data[idx] = orig_opacity[idx]

        pred_arr = np.array(pred_changes, dtype=np.float64)
        actual_arr = np.array(actual_changes, dtype=np.float64)

        denom = np.sum((actual_arr - actual_arr.mean()) ** 2)
        r2 = 1.0 - (np.sum((actual_arr - pred_arr) ** 2) / denom) if denom > 0 else float("nan")
        corr = np.corrcoef(pred_arr, actual_arr)[0, 1] if actual_arr.size > 1 else float("nan")
        spear = spearman_corr(pred_arr, actual_arr)
        topk = topk_overlap(pred_arr, actual_arr, args.topk)

        return {
            "view_index": view_idx,
            "epsilon": eps,
            "num_samples": num_samples,
            "base_loss": base_loss,
            "r2": r2,
            "pearson": corr,
            "spearman": spear,
            "topk_overlap": topk,
        }

    multi = len(view_indices) > 1 or len(eps_list) > 1

    if not multi:
        eps = eps_list[0]
        view_idx = view_indices[0]
        result = run_single(view_idx, eps)
        summary_path = os.path.join(out_dir, "summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"num_samples: {result['num_samples']}\n")
            f.write(f"epsilon: {result['epsilon']}\n")
            f.write(f"base_loss: {result['base_loss']}\n")
            f.write(f"r2: {result['r2']}\n")
            f.write(f"pearson: {result['pearson']}\n")
            f.write(f"spearman: {result['spearman']}\n")
            f.write(f"topk_overlap: {result['topk_overlap']}\n")
            f.write(f"central_diff: {args.central_diff}\n")
            f.write(f"alpha_min: {args.alpha_min}\n")
            f.write(f"alpha_max: {args.alpha_max}\n")
            f.write(f"topk: {args.topk}\n")

        print("Directional-derivative verification finished.")
        print(f"Samples: {result['num_samples']}, epsilon: {result['epsilon']}")
        print(f"R^2: {result['r2']:.6f}, Pearson r: {result['pearson']:.6f}, Spearman: {result['spearman']:.6f}, TopK: {result['topk_overlap']:.6f}")
        print(f"Outputs: {out_dir}")
        return

    # Sweep
    for view_idx in view_indices:
        for eps in eps_list:
            res = run_single(view_idx, eps)
            sweep_results.append(res)

    # Aggregate by epsilon
    by_eps = {}
    for r in sweep_results:
        by_eps.setdefault(r["epsilon"], []).append(r)

    # Aggregate by view
    by_view = {}
    for r in sweep_results:
        by_view.setdefault(r["view_index"], []).append(r)

    # Single summary file with sections
    summary_path = os.path.join(out_dir, "sweep_summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# config central_diff={args.central_diff}, alpha_min={args.alpha_min}, alpha_max={args.alpha_max}, topk={args.topk}\n")
        f.write("# per_run\n")
        f.write("view_index,epsilon,num_samples,base_loss,r2,pearson,spearman,topk_overlap\n")
        for r in sweep_results:
            f.write(f"{r['view_index']},{r['epsilon']},{r['num_samples']},{r['base_loss']},{r['r2']},{r['pearson']},{r['spearman']},{r['topk_overlap']}\n")
        f.write("\n# by_epsilon\n")
        f.write("epsilon,mean_r2,std_r2,mean_pearson,std_pearson,mean_spearman,std_spearman,mean_topk,std_topk\n")
        for eps in eps_list:
            rows = by_eps.get(eps, [])
            r2_vals = np.array([x["r2"] for x in rows], dtype=np.float64)
            pr_vals = np.array([x["pearson"] for x in rows], dtype=np.float64)
            sp_vals = np.array([x["spearman"] for x in rows], dtype=np.float64)
            tk_vals = np.array([x["topk_overlap"] for x in rows], dtype=np.float64)
            mean_r2 = float(np.nanmean(r2_vals)) if r2_vals.size else float("nan")
            std_r2 = float(np.nanstd(r2_vals)) if r2_vals.size else float("nan")
            mean_pr = float(np.nanmean(pr_vals)) if pr_vals.size else float("nan")
            std_pr = float(np.nanstd(pr_vals)) if pr_vals.size else float("nan")
            mean_sp = float(np.nanmean(sp_vals)) if sp_vals.size else float("nan")
            std_sp = float(np.nanstd(sp_vals)) if sp_vals.size else float("nan")
            mean_tk = float(np.nanmean(tk_vals)) if tk_vals.size else float("nan")
            std_tk = float(np.nanstd(tk_vals)) if tk_vals.size else float("nan")
            f.write(f"{eps},{mean_r2},{std_r2},{mean_pr},{std_pr},{mean_sp},{std_sp},{mean_tk},{std_tk}\n")
        f.write("\n# by_view\n")
        f.write("view_index,mean_r2,std_r2,mean_pearson,std_pearson,mean_spearman,std_spearman,mean_topk,std_topk\n")
        for v in view_indices:
            rows = by_view.get(v, [])
            r2_vals = np.array([x["r2"] for x in rows], dtype=np.float64)
            pr_vals = np.array([x["pearson"] for x in rows], dtype=np.float64)
            sp_vals = np.array([x["spearman"] for x in rows], dtype=np.float64)
            tk_vals = np.array([x["topk_overlap"] for x in rows], dtype=np.float64)
            mean_r2 = float(np.nanmean(r2_vals)) if r2_vals.size else float("nan")
            std_r2 = float(np.nanstd(r2_vals)) if r2_vals.size else float("nan")
            mean_pr = float(np.nanmean(pr_vals)) if pr_vals.size else float("nan")
            std_pr = float(np.nanstd(pr_vals)) if pr_vals.size else float("nan")
            mean_sp = float(np.nanmean(sp_vals)) if sp_vals.size else float("nan")
            std_sp = float(np.nanstd(sp_vals)) if sp_vals.size else float("nan")
            mean_tk = float(np.nanmean(tk_vals)) if tk_vals.size else float("nan")
            std_tk = float(np.nanstd(tk_vals)) if tk_vals.size else float("nan")
            f.write(f"{v},{mean_r2},{std_r2},{mean_pr},{std_pr},{mean_sp},{std_sp},{mean_tk},{std_tk}\n")

    if not args.no_plot and HAS_MPL:
        mean_r2 = []
        std_r2 = []
        mean_pr = []
        std_pr = []
        mean_sp = []
        std_sp = []
        mean_tk = []
        std_tk = []
        for eps in eps_list:
            rows = by_eps.get(eps, [])
            r2_vals = np.array([x["r2"] for x in rows], dtype=np.float64)
            pr_vals = np.array([x["pearson"] for x in rows], dtype=np.float64)
            sp_vals = np.array([x["spearman"] for x in rows], dtype=np.float64)
            tk_vals = np.array([x["topk_overlap"] for x in rows], dtype=np.float64)
            mean_r2.append(float(np.nanmean(r2_vals)) if r2_vals.size else float("nan"))
            std_r2.append(float(np.nanstd(r2_vals)) if r2_vals.size else float("nan"))
            mean_pr.append(float(np.nanmean(pr_vals)) if pr_vals.size else float("nan"))
            std_pr.append(float(np.nanstd(pr_vals)) if pr_vals.size else float("nan"))
            mean_sp.append(float(np.nanmean(sp_vals)) if sp_vals.size else float("nan"))
            std_sp.append(float(np.nanstd(sp_vals)) if sp_vals.size else float("nan"))
            mean_tk.append(float(np.nanmean(tk_vals)) if tk_vals.size else float("nan"))
            std_tk.append(float(np.nanstd(tk_vals)) if tk_vals.size else float("nan"))

        has_topk = args.topk is not None and args.topk > 0
        if has_topk:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            ax_list = axes.reshape(-1)
        else:
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))
            ax_list = axes

        ax_list[0].errorbar(eps_list, mean_r2, yerr=std_r2, fmt="-o", capsize=3)
        ax_list[0].set_xlabel("epsilon")
        ax_list[0].set_ylabel("R²")
        ax_list[0].set_title("R² vs epsilon (mean ± std)")

        ax_list[1].errorbar(eps_list, mean_pr, yerr=std_pr, fmt="-o", capsize=3)
        ax_list[1].set_xlabel("epsilon")
        ax_list[1].set_ylabel("Pearson r")
        ax_list[1].set_title("Pearson r vs epsilon (mean ± std)")

        ax_list[2].errorbar(eps_list, mean_sp, yerr=std_sp, fmt="-o", capsize=3)
        ax_list[2].set_xlabel("epsilon")
        ax_list[2].set_ylabel("Spearman ρ")
        ax_list[2].set_title("Spearman ρ vs epsilon (mean ± std)")

        if has_topk:
            ax_list[3].errorbar(eps_list, mean_tk, yerr=std_tk, fmt="-o", capsize=3)
            ax_list[3].set_xlabel("epsilon")
            ax_list[3].set_ylabel("Top-K overlap")
            ax_list[3].set_title("Top-K overlap vs epsilon (mean ± std)")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "sweep_summary.png"), dpi=200)
        plt.close(fig)

    print("Directional-derivative sweep finished.")
    print(f"Views: {view_indices}")
    print(f"Epsilons: {eps_list}")
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
