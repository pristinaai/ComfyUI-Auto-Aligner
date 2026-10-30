"""Align a source image onto a destination image.

Feature detection uses OpenCV SIFT. The fit, puppet warp, colour match, and
auto mask follow the same steps as the browser tool.
"""

import math

import cv2
import numpy as np

MIN_INLIERS = 8
THRESH_FRAC = 0.015
TIGHT_PX = 1.0
RATIO = 0.8
FAST = {
    "max_edge": 1600,
    "max_keypoints": 4000,
    "contrast": 0.015,
    "fusion": False,
    "iters": 500,
}
HIGH = {
    "max_edge": 3200,
    "max_keypoints": 16000,
    "contrast": 0.015,
    "fusion": False,
    "iters": 10000,
}
FUSED = {
    "max_edge": 3200,
    "max_keypoints": 16000,
    "contrast": 0.006,
    "fusion": True,
    "iters": 10000,
}
RIGOROUS = {
    "max_edge": 5120,
    "max_keypoints": 32000,
    "contrast": 0.008,
    "fusion": True,
    "iters": 12000,
}
DEFAULT_WARP_GRID = 12
BASE_LAMBDA = 1e3


def _gray(image):
    if image.ndim == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)


def _pyramid(gray):
    levels = [(gray, 1.0, 1.0)]
    current = gray
    while min(current.shape[:2]) >= 240 and len(levels) < 5:
        nxt = cv2.pyrDown(current)
        parent = levels[-1]
        levels.append(
            (
                nxt,
                parent[1] * (nxt.shape[1] / current.shape[1]),
                parent[2] * (nxt.shape[0] / current.shape[0]),
            )
        )
        current = nxt
    return levels


def _sample_grid(image, xs, ys):
    return cv2.remap(
        image,
        xs.astype(np.float32),
        ys.astype(np.float32),
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _refine_photometric(source_u8, dest_u8, sim, options):
    """Gauss-Newton fit of the warp to pixel values. Keeps the feature fit if this does not improve."""
    free = []
    if options["resize"]:
        free.append("s" if options.get("lock_aspect") else "sx")
        if not options.get("lock_aspect"):
            free.append("sy")
    if options["rotate"]:
        free.append("theta")
    if options["position"]:
        free.extend(("tx", "ty"))
    if not free:
        return sim

    src_levels = _pyramid(_gray(source_u8))
    dst_levels = _pyramid(_gray(dest_u8))
    sx, sy, c, si, tx, ty = [float(v) for v in sim]
    theta = math.atan2(si, c)
    if options["resize"] and options.get("lock_aspect"):
        scale = math.sqrt(max(sx, 1e-6) * max(sy, 1e-6))
        sx = sy = scale
    state = {"sx": sx, "sy": sy, "s": math.sqrt(max(sx, 1e-6) * max(sy, 1e-6)), "theta": theta, "tx": tx, "ty": ty}
    full_w = dest_u8.shape[1]
    full_h = dest_u8.shape[0]
    trans_cap = max(4.0, 0.02 * math.hypot(full_w, full_h))

    def pack(values):
        out_sx = values["s"] if "s" in free else values["sx"]
        out_sy = values["s"] if "s" in free else values["sy"]
        angle = values["theta"]
        return (out_sx, out_sy, math.cos(angle), math.sin(angle), values["tx"], values["ty"])

    def accumulate(level_s, level_d, values):
        image_s, fx_s, fy_s = level_s
        image_d, fx_d, fy_d = level_d
        sh, sw = image_s.shape[:2]
        dh, dw = image_d.shape[:2]
        step = max(1, int(math.sqrt((dh * dw) / 160_000)))
        ys, xs = np.mgrid[2:dh - 2:step, 2:dw - 2:step]
        full_x = (xs + 0.5) / fx_d - 0.5
        full_y = (ys + 0.5) / fy_d - 0.5
        cur = pack(values)
        dx = full_x - cur[4]
        dy = full_y - cur[5]
        rx = cur[2] * dx + cur[3] * dy
        ry = -cur[3] * dx + cur[2] * dy
        src_x = rx / cur[0]
        src_y = ry / cur[1]
        lsx = (src_x + 0.5) * fx_s - 0.5
        lsy = (src_y + 0.5) * fy_s - 0.5
        valid = (lsx > 1) & (lsy > 1) & (lsx < sw - 2) & (lsy < sh - 2)
        if int(valid.sum()) < 80:
            return None
        gx = np.zeros_like(image_s)
        gy = np.zeros_like(image_s)
        gx[:, 1:-1] = (image_s[:, 2:] - image_s[:, :-2]) * 0.5
        gy[1:-1, :] = (image_s[2:, :] - image_s[:-2, :]) * 0.5
        sampled = _sample_grid(image_s, lsx, lsy)[valid]
        grad_x = _sample_grid(gx, lsx, lsy)[valid] * fx_s
        grad_y = _sample_grid(gy, lsx, lsy)[valid] * fy_s
        target = image_d[ys, xs][valid]
        residual = sampled - target
        src_x = src_x[valid]
        src_y = src_y[valid]
        rx = rx[valid]
        ry = ry[valid]
        weight = np.ones_like(residual)
        heavy = np.abs(residual) > 12.0
        weight[heavy] = 12.0 / np.abs(residual[heavy])
        sw = np.sqrt(weight)
        columns = []
        inv_sx = 1.0 / cur[0]
        inv_sy = 1.0 / cur[1]
        for name in free:
            if name == "sx":
                dfull_x, dfull_y = -src_x * inv_sx, np.zeros_like(src_x)
            elif name == "sy":
                dfull_x, dfull_y = np.zeros_like(src_y), -src_y * inv_sy
            elif name == "s":
                dfull_x, dfull_y = -src_x * inv_sx, -src_y * inv_sy
            elif name == "theta":
                dfull_x, dfull_y = ry * inv_sx, -rx * inv_sy
            elif name == "tx":
                dfull_x, dfull_y = np.full_like(src_x, -cur[2] * inv_sx), np.full_like(src_x, cur[3] * inv_sy)
            else:
                dfull_x, dfull_y = np.full_like(src_x, -cur[3] * inv_sx), np.full_like(src_x, -cur[2] * inv_sy)
            columns.append(sw * (grad_x * dfull_x + grad_y * dfull_y))
        jac = np.stack(columns, axis=1)
        weighted_r = sw * residual
        return jac.T @ jac, -(jac.T @ weighted_r), float(np.mean(weight * residual * residual)), int(valid.sum())

    def cost_only(level_s, level_d, values):
        found = accumulate(level_s, level_d, values)
        if found is None:
            return None
        return found[2]

    for level in range(len(src_levels) - 1, -1, -1):
        lam = 1e-3
        for _ in range(8):
            found = accumulate(src_levels[level], dst_levels[level], state)
            if found is None:
                break
            matrix, rhs, cost, _count = found
            diag = np.maximum(np.diag(matrix), 1e-8)
            improved = False
            for _attempt in range(5):
                try:
                    step = np.linalg.solve(matrix + np.diag(lam * diag), rhs)
                except np.linalg.LinAlgError:
                    lam *= 8
                    continue
                trial = dict(state)
                for index, name in enumerate(free):
                    delta = float(step[index])
                    if name in ("sx", "sy", "s"):
                        delta = min(max(delta, -0.04 * trial[name]), 0.04 * trial[name])
                    elif name == "theta":
                        delta = min(max(delta, -0.03), 0.03)
                    else:
                        delta = min(max(delta, -trans_cap), trans_cap)
                    trial[name] = trial[name] + delta
                    if name == "s":
                        trial["sx"] = trial["sy"] = trial["s"]
                if trial["sx"] <= 1e-4 or trial["sy"] <= 1e-4:
                    lam *= 8
                    continue
                trial_cost = cost_only(src_levels[level], dst_levels[level], trial)
                if trial_cost is not None and trial_cost < cost:
                    state = trial
                    lam = max(lam * 0.3, 1e-6)
                    improved = True
                    break
                lam *= 8
            if not improved:
                break
    return pack(state)


def align_pair(source_u8, dest_u8, options):
    """Return float RGB (H, W, 3) and alpha (H, W) at the destination size."""
    height, width = dest_u8.shape[:2]
    mode = options.get("features", "high")
    if mode == "rigorous":
        order = (RIGOROUS, HIGH, FUSED)
    elif mode == "fast":
        order = (FAST, HIGH, FUSED)
    else:
        order = (HIGH, FUSED, FAST)
    pass_result = None
    for budget in order:
        pass_result = _fit_pass(source_u8, dest_u8, options, budget)
        if pass_result is not None:
            break
        print(
            f"Auto Aligner: {budget['max_keypoints']} features was not enough, trying the other pass"
        )
    if pass_result is None:
        raise RuntimeError("Too few inliers. Try images with more overlap.")

    sim, inliers, stats = pass_result
    if mode == "rigorous":
        sim = _refine_photometric(source_u8, dest_u8, sim, options)
    tps = None
    cells = 64
    if options["puppet_warp"]:
        tps = _fit_tps(inliers, sim, options["warp_grid"])
        cells = min(160, max(64, int(options["warp_grid"]) * 5))

    rgb, alpha = _render(source_u8, width, height, sim, tps, cells)
    if options["harmonize"] > 0:
        rgb = _harmonize(rgb, alpha, dest_u8, options["harmonize"])
    if options["auto_mask"]:
        alpha = alpha * _difference_mask(rgb, alpha, dest_u8, options["mask_sensitivity"])
    print(
        f"Auto Aligner: source {stats[0]} pts, destination {stats[1]} pts, "
        f"{stats[2]} matches, {stats[3]} inliers"
    )
    return rgb, alpha.astype(np.float32)


def _fit_pass(source_u8, dest_u8, options, budget):
    feat_a = _extract(source_u8, budget)
    feat_b = _extract(dest_u8, budget)
    if feat_a is None or feat_b is None:
        return None
    pts_a, desc_a = feat_a
    pts_b, desc_b = feat_b
    matches = _match(pts_a, desc_a, pts_b, desc_b)
    if len(matches) < MIN_INLIERS:
        return None
    fit = _ransac(
        matches,
        options,
        source_u8.shape[1],
        source_u8.shape[0],
        dest_u8.shape[1],
        dest_u8.shape[0],
        budget["iters"],
    )
    if fit is None:
        return None
    sim, inliers = fit
    return sim, inliers, (len(pts_a), len(pts_b), len(matches), len(inliers))


def _detect(rgb, nfeatures, contrast):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(
        nfeatures=int(nfeatures),
        nOctaveLayers=3,
        contrastThreshold=float(contrast),
        edgeThreshold=10,
        sigma=1.6,
    )
    keypoints, desc = sift.detectAndCompute(gray, None)
    if desc is None or len(keypoints) == 0:
        return None
    pts = np.array([kp.pt for kp in keypoints], np.float64)
    response = np.array([kp.response for kp in keypoints], np.float32)
    desc = desc.astype(np.float32)
    desc /= np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-8)
    return pts, desc, response


def _spatial_cap(pts, desc, response, limit, width, height):
    if len(pts) <= limit:
        return pts, desc
    cells = 24
    cx = np.clip(np.floor(pts[:, 0] / max(width, 1) * cells), 0, cells - 1).astype(np.int32)
    cy = np.clip(np.floor(pts[:, 1] / max(height, 1) * cells), 0, cells - 1).astype(np.int32)
    buckets = [[] for _ in range(cells * cells)]
    for idx in np.argsort(-response):
        buckets[int(cy[idx]) * cells + int(cx[idx])].append(int(idx))
    kept = []
    round_i = 0
    while len(kept) < limit:
        added = 0
        for bucket in buckets:
            if round_i < len(bucket):
                kept.append(bucket[round_i])
                added += 1
                if len(kept) >= limit:
                    break
        if added == 0:
            break
        round_i += 1
    kept = np.array(kept, np.int32)
    return pts[kept], desc[kept]


def _extract(rgb, budget):
    height, width = rgb.shape[:2]
    scale = min(1.0, budget["max_edge"] / max(height, width))
    work_w = max(32, int(round(width * scale)))
    work_h = max(32, int(round(height * scale)))
    work = cv2.resize(rgb, (work_w, work_h), interpolation=cv2.INTER_AREA)
    limit = budget["max_keypoints"]
    primary_n = int(limit * 0.88) if budget["fusion"] else limit
    found = _detect(work, max(primary_n, 8), budget["contrast"])
    if found is None:
        return None
    pts, desc, response = found
    pts = _to_original(pts, work_w, work_h, width, height)
    if budget["fusion"] and min(work_w, work_h) >= 512:
        half_w = max(32, work_w // 2)
        half_h = max(32, work_h // 2)
        half = cv2.resize(work, (half_w, half_h), interpolation=cv2.INTER_AREA)
        sec_n = max(256, min(2048, int(limit * 0.25)))
        second = _detect(half, sec_n, budget["contrast"])
        if second is not None:
            pts2, desc2, response2 = second
            pts2 = _to_original(pts2, half_w, half_h, width, height)
            pts = np.concatenate([pts, pts2], axis=0)
            desc = np.concatenate([desc, desc2], axis=0)
            response = np.concatenate([response, response2], axis=0)
    if len(pts) < 8:
        return None
    pts, desc = _spatial_cap(pts, desc, response, limit, width, height)
    if len(pts) < 8:
        return None
    return pts, desc


def _to_original(pts, work_w, work_h, width, height):
    """Map work-image pixel centers back to the original image."""
    out = pts.copy()
    out[:, 0] = (pts[:, 0] + 0.5) * (width / work_w) - 0.5
    out[:, 1] = (pts[:, 1] + 0.5) * (height / work_h) - 0.5
    return out


def _match(pts_a, desc_a, pts_b, desc_b):
    """Exact nearest neighbours. Ratio test is one way, then a mutual check."""
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    forward = matcher.knnMatch(desc_a, desc_b, k=2)
    backward = matcher.knnMatch(desc_b, desc_a, k=2)
    back = {}
    for pair in backward:
        if pair:
            back[pair[0].queryIdx] = pair[0].trainIdx
    pairs = []
    for pair in forward:
        if len(pair) < 2:
            continue
        best, second = pair
        if second.distance <= 0 or best.distance >= RATIO * second.distance:
            continue
        i = best.queryIdx
        j = best.trainIdx
        if back.get(j) != i:
            continue
        pairs.append((pts_a[i, 0], pts_a[i, 1], pts_b[j, 0], pts_b[j, 1]))
    return pairs


def _apply(sim, x, y):
    sx, sy, c, si, tx, ty = sim
    return c * sx * x - si * sy * y + tx, si * sx * x + c * sy * y + ty


def _apply_inv(sim, x, y):
    sx, sy, c, si, tx, ty = sim
    dx = x - tx
    dy = y - ty
    rx = c * dx + si * dy
    ry = -si * dx + c * dy
    return rx / sx, ry / sy


def _fit_uniform(src, dst, options, width_a, height_a, width_b, height_b):
    count = len(src)
    if count <= 0:
        return None
    mu_a = src.mean(axis=0)
    mu_b = dst.mean(axis=0)
    a = src - mu_a
    b = dst - mu_b
    sxx = float(np.sum(b[:, 0] * a[:, 0]))
    sxy = float(np.sum(b[:, 0] * a[:, 1]))
    syx = float(np.sum(b[:, 1] * a[:, 0]))
    syy = float(np.sum(b[:, 1] * a[:, 1]))
    var_a = float(np.sum(a * a))
    dot = float(np.sum(a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
    c, si = 1.0, 0.0
    if options["rotate"]:
        angle = math.atan2(syx - sxy, sxx + syy)
        c, si = math.cos(angle), math.sin(angle)
        if (sxx * syy - sxy * syx) < 0 and abs(sxx + syy) < 1e-12:
            c = -c
    scale = 1.0
    if options["resize"]:
        if var_a < 1e-12:
            return None
        if options["rotate"]:
            rx = c * a[:, 0] - si * a[:, 1]
            ry = si * a[:, 0] + c * a[:, 1]
            scale = float(np.sum(rx * b[:, 0] + ry * b[:, 1]) / var_a)
        else:
            scale = dot / var_a
        if not math.isfinite(scale) or scale <= 1e-6:
            return None
    if options["position"]:
        tx = mu_b[0] - scale * (c * mu_a[0] - si * mu_a[1])
        ty = mu_b[1] - scale * (si * mu_a[0] + c * mu_a[1])
    else:
        cx_a, cy_a = width_a * 0.5, height_a * 0.5
        cx_b, cy_b = width_b * 0.5, height_b * 0.5
        tx = cx_b - scale * (c * cx_a - si * cy_a)
        ty = cy_b - scale * (si * cx_a + c * cy_a)
    sim = (scale, scale, c, si, float(tx), float(ty))
    if not all(math.isfinite(v) for v in sim):
        return None
    return sim


def _fit_aniso(src, dst, options, width_a, height_a, width_b, height_b):
    """Independent x and y scale, one rotation, no shear."""
    if len(src) <= 0:
        return None
    if options["position"]:
        a0 = src.mean(axis=0)
        b0 = dst.mean(axis=0)
    else:
        a0 = np.array([width_a * 0.5, height_a * 0.5])
        b0 = np.array([width_b * 0.5, height_b * 0.5])
    a = src - a0
    b = dst - b0
    sxx = float(np.sum(b[:, 0] * a[:, 0]))
    sxy = float(np.sum(b[:, 0] * a[:, 1]))
    syx = float(np.sum(b[:, 1] * a[:, 0]))
    syy = float(np.sum(b[:, 1] * a[:, 1]))
    c, si = 1.0, 0.0
    if options["rotate"]:
        angle = math.atan2(syx - sxy, sxx + syy)
        c, si = math.cos(angle), math.sin(angle)
    sx, sy = 1.0, 1.0
    x = a[:, 0]
    y = a[:, 1]
    u = b[:, 0]
    v = b[:, 1]
    for _ in range(8):
        xx = float(np.sum(x * x))
        yy = float(np.sum(y * y))
        if xx < 1e-12 or yy < 1e-12:
            return None
        sx = float(np.sum(x * (c * u + si * v)) / xx)
        sy = float(np.sum(y * (-si * u + c * v)) / yy)
        if not (math.isfinite(sx) and math.isfinite(sy)) or sx <= 1e-6 or sy <= 1e-6:
            return None
        if not options["rotate"]:
            break
        xs = sx * x
        ys = sy * y
        sxx = float(np.sum(u * xs))
        sxy = float(np.sum(u * ys))
        syx = float(np.sum(v * xs))
        syy = float(np.sum(v * ys))
        angle = math.atan2(syx - sxy, sxx + syy)
        nc, nsi = math.cos(angle), math.sin(angle)
        if abs(nc - c) + abs(nsi - si) < 1e-12:
            c, si = nc, nsi
            break
        c, si = nc, nsi
    tx = float(b0[0] - (c * sx * a0[0] - si * sy * a0[1]))
    ty = float(b0[1] - (si * sx * a0[0] + c * sy * a0[1]))
    sim = (sx, sy, c, si, tx, ty)
    if not all(math.isfinite(v) for v in sim):
        return None
    return sim


def _fit_similarity(src, dst, options, width_a, height_a, width_b, height_b):
    if options["resize"] and not options.get("lock_aspect"):
        return _fit_aniso(src, dst, options, width_a, height_a, width_b, height_b)
    return _fit_uniform(src, dst, options, width_a, height_a, width_b, height_b)


def _inliers(pairs, sim, thresh):
    src = np.array([(p[0], p[1]) for p in pairs], np.float64)
    dst = np.array([(p[2], p[3]) for p in pairs], np.float64)
    mx, my = _apply(sim, src[:, 0], src[:, 1])
    keep = (mx - dst[:, 0]) ** 2 + (my - dst[:, 1]) ** 2 <= thresh * thresh
    return src[keep], dst[keep]


def _magsac(pairs, options, width_a, height_a, width_b, height_b):
    src = np.float32([(p[0], p[1]) for p in pairs])
    dst = np.float32([(p[2], p[3]) for p in pairs])
    thresh = THRESH_FRAC * math.hypot(width_b, height_b)
    affine, mask = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(thresh),
        maxIters=10000,
        confidence=0.999,
        refineIters=50,
    )
    if affine is None or mask is None:
        return None
    keep = mask.ravel().astype(bool)
    if int(keep.sum()) < MIN_INLIERS:
        return None
    sim = _fit_similarity(
        src[keep].astype(np.float64),
        dst[keep].astype(np.float64),
        options,
        width_a,
        height_a,
        width_b,
        height_b,
    )
    if sim is None:
        return None
    inl_src, inl_dst = _inliers(pairs, sim, thresh)
    if len(inl_src) >= MIN_INLIERS:
        final = _fit_similarity(inl_src, inl_dst, options, width_a, height_a, width_b, height_b)
        if final is not None:
            sim = final
        used_src, used_dst = inl_src, inl_dst
    else:
        used_src = src[keep].astype(np.float64)
        used_dst = dst[keep].astype(np.float64)
    sim, used_src, used_dst = _tighten(used_src, used_dst, sim, options, width_a, height_a, width_b, height_b)
    return sim, np.concatenate([used_src, used_dst], axis=1)


def _ransac(pairs, options, width_a, height_a, width_b, height_b, iterations):
    if len(pairs) < 2:
        return None
    aniso = options["resize"] and options["rotate"] and not options.get("lock_aspect")
    if options["resize"] and options["rotate"] and options["position"] and options.get("lock_aspect"):
        fitted = _magsac(pairs, options, width_a, height_a, width_b, height_b)
        if fitted is not None:
            return fitted
    thresh = THRESH_FRAC * math.hypot(width_b, height_b)
    iterations = max(iterations, 8000)
    rng = np.random.default_rng()
    best_src = None
    best_dst = None
    n = len(pairs)
    need = 3 if aniso else 2
    if n < need:
        return None
    for _ in range(iterations):
        pick = rng.choice(n, size=need, replace=False)
        sample = [pairs[int(i)] for i in pick]
        src = np.array([[p[0], p[1]] for p in sample], np.float64)
        dst = np.array([[p[2], p[3]] for p in sample], np.float64)
        model = _fit_similarity(src, dst, options, width_a, height_a, width_b, height_b)
        if model is None:
            continue
        inl_src, inl_dst = _inliers(pairs, model, thresh)
        if best_src is None or len(inl_src) > len(best_src):
            best_src, best_dst = inl_src, inl_dst
    if best_src is None or len(best_src) < MIN_INLIERS:
        return None
    sim = _fit_similarity(best_src, best_dst, options, width_a, height_a, width_b, height_b)
    if sim is None:
        return None
    refined_src, refined_dst = _inliers(pairs, sim, thresh)
    if len(refined_src) >= MIN_INLIERS:
        final = _fit_similarity(refined_src, refined_dst, options, width_a, height_a, width_b, height_b)
        if final is not None:
            sim = final
            best_src, best_dst = refined_src, refined_dst
    sim, best_src, best_dst = _tighten(
        best_src, best_dst, sim, options, width_a, height_a, width_b, height_b
    )
    packed = np.concatenate([best_src, best_dst], axis=1).astype(np.float64)
    return sim, packed


def _tighten(src, dst, sim, options, width_a, height_a, width_b, height_b):
    """Refit on matches that already agree to about one pixel."""
    mx, my = _apply(sim, src[:, 0], src[:, 1])
    keep = (mx - dst[:, 0]) ** 2 + (my - dst[:, 1]) ** 2 <= TIGHT_PX * TIGHT_PX
    if int(keep.sum()) < MIN_INLIERS:
        return sim, src, dst
    final = _fit_similarity(src[keep], dst[keep], options, width_a, height_a, width_b, height_b)
    if final is None:
        return sim, src, dst
    return final, src[keep], dst[keep]


def _u(r2):
    out = np.zeros_like(r2)
    ok = r2 >= 1e-12
    out[ok] = r2[ok] * np.log(r2[ok])
    return out


def _fit_tps(inliers, sim, warp_grid):
    count = len(inliers)
    cells = max(2, int(round(warp_grid)))
    xs = inliers[:, 2]
    ys = inliers[:, 3]
    span_x = max(1e-6, float(xs.max() - xs.min()))
    span_y = max(1e-6, float(ys.max() - ys.min()))
    cx = np.clip(np.floor((xs - xs.min()) / span_x * cells), 0, cells - 1).astype(np.int32)
    cy = np.clip(np.floor((ys - ys.min()) / span_y * cells), 0, cells - 1).astype(np.int32)
    bins = cy * cells + cx
    acc = np.zeros((cells * cells, 4), np.float64)
    hits = np.zeros(cells * cells, np.int32)
    np.add.at(acc, bins, inliers)
    np.add.at(hits, bins, 1)
    keep = hits > 0
    if int(keep.sum()) < 3:
        return None
    pts = acc[keep] / hits[keep, None]
    n = len(pts)
    lam = BASE_LAMBDA * (DEFAULT_WARP_GRID / cells) ** 2
    dim = n + 3
    k = np.zeros((dim, dim), np.float64)
    diff = pts[:, None, 2:4] - pts[None, :, 2:4]
    k[:n, :n] = _u(np.sum(diff * diff, axis=2))
    k[np.arange(n), np.arange(n)] += lam
    k[:n, n] = 1
    k[:n, n + 1] = pts[:, 2]
    k[:n, n + 2] = pts[:, 3]
    k[n, :n] = 1
    k[n + 1, :n] = pts[:, 2]
    k[n + 2, :n] = pts[:, 3]
    inv_x, inv_y = _apply_inv(sim, pts[:, 2], pts[:, 3])
    bx = np.zeros(dim)
    by = np.zeros(dim)
    bx[:n] = pts[:, 0] - inv_x
    by[:n] = pts[:, 1] - inv_y
    try:
        wx = np.linalg.solve(k, bx)
        wy = np.linalg.solve(k, by)
    except np.linalg.LinAlgError:
        return None
    return pts[:, 2], pts[:, 3], wx, wy


def _maps(width, height, sim, tps):
    """Source pixel coordinate for every destination pixel center."""
    xs = np.arange(width, dtype=np.float64)
    if tps is None:
        grid_x, grid_y = np.meshgrid(xs, np.arange(height, dtype=np.float64))
        mx, my = _apply_inv(sim, grid_x, grid_y)
        return np.ascontiguousarray(mx, np.float32), np.ascontiguousarray(my, np.float32)

    cx, cy, wx, wy = tps
    n = len(cx)
    map_x = np.empty((height, width), np.float32)
    map_y = np.empty((height, width), np.float32)
    for y0 in range(0, height, 24):
        y1 = min(height, y0 + 24)
        grid_x, grid_y = np.meshgrid(xs, np.arange(y0, y1, dtype=np.float64))
        x = grid_x.reshape(-1)
        y = grid_y.reshape(-1)
        mx, my = _apply_inv(sim, x, y)
        dx = x[:, None] - cx[None, :]
        dy = y[:, None] - cy[None, :]
        basis = _u(dx * dx + dy * dy)
        ox = wx[n] + wx[n + 1] * x + wx[n + 2] * y + basis @ wx[:n]
        oy = wy[n] + wy[n + 1] * x + wy[n + 2] * y + basis @ wy[:n]
        map_x[y0:y1] = (mx + ox).reshape(y1 - y0, width).astype(np.float32)
        map_y[y0:y1] = (my + oy).reshape(y1 - y0, width).astype(np.float32)
    return map_x, map_y


def _prefilter_source(source_u8, sim):
    """Blur before a shrink so the resample does not keep the original pixels."""
    sx = max(float(sim[0]), 1e-6)
    sy = max(float(sim[1]), 1e-6)
    sigma_x = 0.5 * max(0.0, (1.0 / sx) - 1.0)
    sigma_y = 0.5 * max(0.0, (1.0 / sy) - 1.0)
    if sigma_x < 0.2 and sigma_y < 0.2:
        return source_u8
    return cv2.GaussianBlur(
        source_u8,
        (0, 0),
        sigmaX=max(sigma_x, 1e-3),
        sigmaY=max(sigma_y, 1e-3),
    )


def _render(source_u8, width, height, sim, tps, cells):
    del cells
    map_x, map_y = _maps(width, height, sim, tps)
    source_u8 = _prefilter_source(source_u8, sim)
    warped = cv2.remap(
        source_u8,
        map_x,
        map_y,
        cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    coverage = np.ones(source_u8.shape[:2], np.float32)
    alpha = cv2.remap(
        coverage,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped.astype(np.float32) / 255.0, np.clip(alpha, 0.0, 1.0)


def _srgb_to_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def _lab_f(t):
    return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)


def _lab_f_inv(t):
    cube = t * t * t
    return np.where(cube > 0.008856, cube, (t - 16 / 116) / 7.787)


_TO_XYZ = np.array(
    [
        [0.4124, 0.3576, 0.1805],
        [0.2126, 0.7152, 0.0722],
        [0.0193, 0.1192, 0.9505],
    ],
    np.float64,
)
_FROM_XYZ = np.array(
    [
        [3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0415],
        [0.0557, -0.204, 1.057],
    ],
    np.float64,
)
_WHITE = np.array([0.95047, 1.0, 1.08883], np.float64)


def _rgb_to_lab(rgb):
    lin = _srgb_to_lin(np.clip(rgb, 0, 1))
    xyz = lin @ _TO_XYZ.T / _WHITE
    f = _lab_f(xyz)
    return np.stack(
        [116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])],
        axis=-1,
    )


def _lab_to_rgb(lab):
    fy = (lab[..., 0] + 16) / 116
    xyz = np.stack(
        [
            0.95047 * _lab_f_inv(fy + lab[..., 1] / 500),
            _lab_f_inv(fy),
            1.08883 * _lab_f_inv(fy - lab[..., 2] / 200),
        ],
        axis=-1,
    )
    return _lin_to_srgb(xyz @ _FROM_XYZ.T)


def _blur_grid(grid, weight):
    n = grid.shape[0]
    out = np.zeros_like(grid)
    for y in range(n):
        for x in range(n):
            total = 0.0
            w = 0.0
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    gx, gy = x + dx, y + dy
                    if gx < 0 or gy < 0 or gx >= n or gy >= n:
                        continue
                    kw = weight[gy, gx] / (1 + dx * dx + dy * dy)
                    total += grid[gy, gx] * kw
                    w += kw
            out[y, x] = total / w if w > 0 else 0
    return out


def _harmonize(rgb, alpha, dest_u8, strength):
    covered = alpha >= 0.98
    if int(covered.sum()) < 64:
        return rgb
    lab_a = _rgb_to_lab(rgb)
    lab_b = _rgb_to_lab(dest_u8.astype(np.float32) / 255.0)
    step = max(1, int(round(math.sqrt(rgb.shape[0] * rgb.shape[1] / 400_000))))
    sample = covered[::step, ::step]
    a_s = lab_a[::step, ::step][sample]
    b_s = lab_b[::step, ::step][sample]
    if len(a_s) < 64:
        return rgb
    mean_a = a_s.mean(axis=0)
    mean_b = b_s.mean(axis=0)
    std_a = np.maximum(a_s.std(axis=0), 1e-3)
    std_b = np.maximum(b_s.std(axis=0), 1e-3)
    gain = np.clip(std_b / std_a, 1 / 3, 3)

    g = 24
    cell_sum = np.zeros((g, g), np.float64)
    cell_w = np.zeros((g, g), np.float64)
    ys, xs = np.mgrid[0:rgb.shape[0]:step, 0:rgb.shape[1]:step]
    gy = np.clip((ys / rgb.shape[0] * g).astype(np.int32), 0, g - 1)
    gx = np.clip((xs / rgb.shape[1] * g).astype(np.int32), 0, g - 1)
    transferred_l = (lab_a[::step, ::step, 0] - mean_a[0]) * gain[0] + mean_b[0]
    delta = lab_b[::step, ::step, 0] - transferred_l
    mask = covered[::step, ::step]
    np.add.at(cell_sum, (gy[mask], gx[mask]), delta[mask])
    np.add.at(cell_w, (gy[mask], gx[mask]), 1)
    nz = cell_w > 0
    cell_sum[nz] /= cell_w[nz]
    light = _blur_grid(_blur_grid(cell_sum, cell_w), np.where(cell_w > 0, 1.0, 0.25))

    h, w = rgb.shape[:2]
    fy = np.clip((np.arange(h) / h) * g - 0.5, 0, g - 1.001)
    fx = np.clip((np.arange(w) / w) * g - 0.5, 0, g - 1.001)
    y0 = np.floor(fy).astype(np.int32)
    x0 = np.floor(fx).astype(np.int32)
    ty = fy - y0
    tx = fx - x0
    light_map = (
        (light[y0][:, x0] * (1 - tx)) * (1 - ty)[:, None]
        + (light[y0][:, x0 + 1] * tx) * (1 - ty)[:, None]
        + (light[y0 + 1][:, x0] * (1 - tx)) * ty[:, None]
        + (light[y0 + 1][:, x0 + 1] * tx) * ty[:, None]
    )
    target = (lab_a - mean_a) * gain + mean_b
    target[..., 0] += light_map
    mixed = lab_a + (target - lab_a) * strength
    out = _lab_to_rgb(mixed).astype(np.float32)
    out[~covered] = rgb[~covered]
    return np.clip(out, 0, 1)


def _smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1)
    return t * t * (3 - 2 * t)


def _difference_mask(rgb, alpha, dest_u8, sensitivity):
    h, w = rgb.shape[:2]
    scale = min(1.0, 1024 / max(h, w))
    mw = max(8, int(round(w * scale)))
    mh = max(8, int(round(h * scale)))
    small_a = cv2.resize((rgb * 255).astype(np.uint8), (mw, mh), interpolation=cv2.INTER_AREA).astype(np.float32)
    small_b = cv2.resize(dest_u8, (mw, mh), interpolation=cv2.INTER_AREA).astype(np.float32)
    small_alpha = cv2.resize(alpha, (mw, mh), interpolation=cv2.INTER_AREA)
    dr = small_a[..., 0] - small_b[..., 0]
    dg = small_a[..., 1] - small_b[..., 1]
    db = small_a[..., 2] - small_b[..., 2]
    diff = np.sqrt(0.3 * dr * dr + 0.59 * dg * dg + 0.11 * db * db) / 255.0
    diff[small_alpha < 8 / 255] = 0
    radius = max(1, int(round(mw / 400)))
    k = radius * 2 + 1
    smooth = cv2.blur(cv2.blur(diff, (k, k)), (k, k))
    thresh = 0.3 - 0.27 * min(1.0, max(0.0, sensitivity))
    mask = _smoothstep(thresh * 0.7, thresh * 1.3, smooth).astype(np.float32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 4 + 1, radius * 4 + 1))
    mask = cv2.dilate(mask, kernel)
    soft = radius * 4 + 1
    mask = cv2.blur(cv2.blur(mask, (soft, soft)), (soft, soft))
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
