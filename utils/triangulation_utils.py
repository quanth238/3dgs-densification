import torch
from utils.graphics_utils import fov2focal


def get_intrinsics(view):
    fx = fov2focal(view.FoVx, view.image_width)
    fy = fov2focal(view.FoVy, view.image_height)
    cx = view.image_width * 0.5
    cy = view.image_height * 0.5
    return fx, fy, cx, cy


def compute_fundamental(view_a, view_b):
    fx1, fy1, cx1, cy1 = get_intrinsics(view_a)
    fx2, fy2, cx2, cy2 = get_intrinsics(view_b)
    device = view_a.world_view_transform.device
    K1 = torch.tensor([[fx1, 0.0, cx1], [0.0, fy1, cy1], [0.0, 0.0, 1.0]], device=device)
    K2 = torch.tensor([[fx2, 0.0, cx2], [0.0, fy2, cy2], [0.0, 0.0, 1.0]], device=device)
    # world_view_transform is stored for row-vector math. Transpose to column-vector form.
    T1 = view_a.world_view_transform.transpose(0, 1)
    T2 = view_b.world_view_transform.transpose(0, 1)
    T = T2 @ torch.inverse(T1)
    R = T[:3, :3]
    t = T[:3, 3]
    tx = torch.tensor(
        [
            [0.0, -t[2], t[1]],
            [t[2], 0.0, -t[0]],
            [-t[1], t[0], 0.0],
        ],
        device=device,
    )
    F = torch.linalg.inv(K2).T @ tx @ R @ torch.linalg.inv(K1)
    return F


def project_world_points(view, pts):
    H = int(view.image_height)
    W = int(view.image_width)
    device = pts.device
    ones = torch.ones((pts.shape[0], 1), device=device, dtype=pts.dtype)
    pts_hom = torch.cat([pts, ones], dim=1)
    projmat = view.full_proj_transform
    p_clip = pts_hom @ projmat
    ndc = p_clip[:, :3] / (p_clip[:, 3:4] + 1e-8)
    finite = torch.isfinite(ndc).all(dim=1)
    x = ((ndc[:, 0] + 1.0) * W - 1.0) * 0.5
    y = ((ndc[:, 1] + 1.0) * H - 1.0) * 0.5
    xi = x.round().long()
    yi = y.round().long()
    valid = finite & (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
    return x, y, valid


def line_segment_from_fundamental(view_a, view_b, px, py, downscale):
    H = int(view_b.image_height)
    W = int(view_b.image_width)
    F = compute_fundamental(view_a, view_b)
    x1 = torch.tensor([px, py, 1.0], device="cuda")
    l = F @ x1
    a = float(l[0].item())
    b = float(l[1].item())
    c = float(l[2].item())
    if abs(a) + abs(b) < 1e-8:
        return None

    pts = []
    # Intersect with x=0 and x=W-1
    for x in (0.0, float(W - 1)):
        if abs(b) > 1e-8:
            y = -(a * x + c) / b
            if 0.0 <= y <= H - 1:
                pts.append((x, y))
    # Intersect with y=0 and y=H-1
    for y in (0.0, float(H - 1)):
        if abs(a) > 1e-8:
            x = -(b * y + c) / a
            if 0.0 <= x <= W - 1:
                pts.append((x, y))

    if len(pts) < 2:
        return None
    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    return x0 / downscale, y0 / downscale, x1 / downscale, y1 / downscale


def pixel_to_world(view, px, py, depth, fx, fy, cx, cy):
    x = (px + 0.5 - cx) / fx
    y = (py + 0.5 - cy) / fy
    cam = torch.tensor([x * depth, y * depth, depth, 1.0], device="cuda", dtype=torch.float32)
    view_inv = view.world_view_transform.inverse()
    world = cam @ view_inv
    return world[:3]


def ray_from_pixel(view, px, py):
    fx, fy, cx, cy = get_intrinsics(view)
    x_cam = (px + 0.5 - cx) / fx
    y_cam = (py + 0.5 - cy) / fy
    dir_cam = torch.tensor([x_cam, y_cam, 1.0, 0.0], device="cuda", dtype=torch.float32)
    origin_cam = torch.tensor([0.0, 0.0, 0.0, 1.0], device="cuda", dtype=torch.float32)
    view_inv = view.world_view_transform.inverse()
    origin = (origin_cam @ view_inv)[:3]
    direction = (dir_cam @ view_inv)[:3]
    direction = direction / (torch.norm(direction) + 1e-8)
    return origin, direction


def closest_point_between_rays(o1, d1, o2, d2, eps=1e-6):
    w0 = o1 - o2
    a = torch.dot(d1, d1)
    b = torch.dot(d1, d2)
    c = torch.dot(d2, d2)
    d = torch.dot(d1, w0)
    e = torch.dot(d2, w0)
    denom = a * c - b * b
    if denom.abs() < eps:
        return None
    s = (b * e - c * d) / denom
    t = (a * e - b * d) / denom
    p1 = o1 + s * d1
    p2 = o2 + t * d2
    return 0.5 * (p1 + p2), s.item(), t.item(), torch.norm(p1 - p2).item()


def extract_patch(img, x, y, r):
    H, W = img.shape[-2], img.shape[-1]
    x0 = x - r
    x1 = x + r + 1
    y0 = y - r
    y1 = y + r + 1
    if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
        return None
    patch = img[:, y0:y1, x0:x1].contiguous().view(-1)
    return patch


def match_pixel_epipolar(
    view_a,
    view_b,
    img_a,
    img_b,
    px,
    py,
    depth_min,
    depth_max,
    downscale,
    patch_r,
    max_samples,
    min_ncc,
    stats=None,
):
    # Downscale images for matching.
    if downscale > 1:
        img_a_ds = torch.nn.functional.interpolate(
            img_a.unsqueeze(0),
            scale_factor=1.0 / downscale,
            mode="bilinear",
            align_corners=False,
        )[0]
        img_b_ds = torch.nn.functional.interpolate(
            img_b.unsqueeze(0),
            scale_factor=1.0 / downscale,
            mode="bilinear",
            align_corners=False,
        )[0]
    else:
        img_a_ds = img_a
        img_b_ds = img_b

    px_ds = int(round(px / downscale))
    py_ds = int(round(py / downscale))
    patch_a = extract_patch(img_a_ds, px_ds, py_ds, patch_r)
    if patch_a is None:
        if stats is not None:
            stats["patch_a_fail"] += 1
        return None
    patch_a = patch_a - patch_a.mean()
    norm_a = torch.norm(patch_a) + 1e-8

    fx, fy, cx, cy = get_intrinsics(view_a)
    p_world_near = pixel_to_world(view_a, px, py, depth_min, fx, fy, cx, cy)
    p_world_far = pixel_to_world(view_a, px, py, depth_max, fx, fy, cx, cy)
    line_pts = torch.stack([p_world_near, p_world_far], dim=0)
    xs, ys, valid = project_world_points(view_b, line_pts)
    finite = torch.isfinite(xs) & torch.isfinite(ys)

    def try_depth_line():
        if not finite.all():
            return None
        x0 = xs[0].item() / downscale
        y0 = ys[0].item() / downscale
        x1 = xs[1].item() / downscale
        y1 = ys[1].item() / downscale

        dx = x1 - x0
        dy = y1 - y0
        if abs(dx) < 1e-8:
            if x0 < 0 or x0 > (img_b_ds.shape[-1] - 1):
                return None
            tx_min, tx_max = 0.0, 1.0
        else:
            tx0 = (0.0 - x0) / dx
            tx1 = ((img_b_ds.shape[-1] - 1) - x0) / dx
            tx_min, tx_max = min(tx0, tx1), max(tx0, tx1)

        if abs(dy) < 1e-8:
            if y0 < 0 or y0 > (img_b_ds.shape[-2] - 1):
                return None
            ty_min, ty_max = 0.0, 1.0
        else:
            ty0 = (0.0 - y0) / dy
            ty1 = ((img_b_ds.shape[-2] - 1) - y0) / dy
            ty_min, ty_max = min(ty0, ty1), max(ty0, ty1)

        t_min = max(0.0, tx_min, ty_min)
        t_max = min(1.0, tx_max, ty_max)
        if t_max <= t_min:
            return None
        return x0, y0, x1, y1, t_min, t_max

    line = try_depth_line()
    if line is None:
        fb = line_segment_from_fundamental(view_a, view_b, px, py, downscale)
        if fb is None:
            if stats is not None:
                stats["line_fail"] += 1
            return None
        x0, y0, x1, y1 = fb
        t_min, t_max = 0.0, 1.0
    else:
        x0, y0, x1, y1, t_min, t_max = line

    if max_samples < 2:
        max_samples = 2
    dx = x1 - x0
    dy = y1 - y0

    t_vals = torch.linspace(t_min, t_max, steps=max_samples, device="cuda")
    xs_line = x0 + dx * t_vals
    ys_line = y0 + dy * t_vals

    best_score = -1.0
    best_px = None
    best_py = None
    H, W = img_b_ds.shape[-2], img_b_ds.shape[-1]
    for x, y in zip(xs_line, ys_line):
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 0 or yi < 0 or xi >= W or yi >= H:
            if stats is not None:
                stats["sample_oob"] += 1
            continue
        patch_b = extract_patch(img_b_ds, xi, yi, patch_r)
        if patch_b is None:
            if stats is not None:
                stats["patch_b_fail"] += 1
            continue
        patch_b = patch_b - patch_b.mean()
        denom = (norm_a * (torch.norm(patch_b) + 1e-8)).item()
        if denom <= 0:
            continue
        ncc = float(torch.dot(patch_a, patch_b).item() / denom)
        if ncc > best_score:
            best_score = ncc
            best_px = int(round(xi * downscale))
            best_py = int(round(yi * downscale))

    if best_score < float(min_ncc) or best_px is None:
        if stats is not None:
            stats["ncc_fail"] += 1
        return None
    return best_px, best_py, best_score
