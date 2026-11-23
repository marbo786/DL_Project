"""Image-only registration and inspectable reference difference maps."""

from dataclasses import dataclass

import cv2
import numpy as np

from research.reference_data import project


@dataclass
class Alignment:
    image: np.ndarray
    valid: np.ndarray
    matrix: np.ndarray
    accepted: bool
    inliers: int
    inlier_ratio: float
    reprojection_px: float | None


def grayscale(rgb):
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def register(reference, candidate):
    shape = reference.shape[:2]
    fallback = Alignment(candidate, np.ones(shape, bool), np.eye(3), False, 0, 0., None)
    detector = cv2.ORB_create(nfeatures=1200, fastThreshold=9, edgeThreshold=12)
    kp_ref, des_ref = detector.detectAndCompute(grayscale(reference), None)
    kp_can, des_can = detector.detectAndCompute(grayscale(candidate), None)
    if des_ref is None or des_can is None:
        return fallback
    matches = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des_can, des_ref, k=2)
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < .78 * pair[1].distance]
    if len(good) < 10:
        return fallback
    src = np.float32([kp_can[m.queryIdx].pt for m in good])
    dst = np.float32([kp_ref[m.trainIdx].pt for m in good])
    matrix, inlier = cv2.findHomography(src, dst, cv2.RANSAC, 2.5)
    if matrix is None or inlier is None or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-8:
        return fallback
    kept = inlier.ravel().astype(bool)
    count, ratio = int(kept.sum()), float(kept.mean())
    if count < 10 or ratio < .32:
        return fallback
    target = cv2.perspectiveTransform(src[kept, None, :], matrix)[:, 0]
    residual = float(np.median(np.linalg.norm(dst[kept] - target, axis=1)))
    height, width = shape
    area = cv2.contourArea(cv2.convexHull(dst[kept]))
    corners = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    warped_corners = cv2.perspectiveTransform(corners[:, None, :], matrix)[:, 0]
    valid = project(np.ones(shape, np.uint8), matrix, shape, nearest=True)
    valid = cv2.erode(valid, np.ones((5, 5), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    if (area < .03 * height * width or residual > 1.65 or valid.mean() < .75
            or np.max(np.abs(warped_corners - corners), axis=0).dot([1 / width, 1 / height]) > .45):
        return Alignment(candidate, fallback.valid, fallback.matrix, False, count, ratio, residual)
    return Alignment(project(candidate, matrix, shape), valid, matrix, True, count, ratio, residual)


def robust_luminance(reference, candidate, valid):
    a, b = grayscale(reference).astype(np.float32), grayscale(candidate).astype(np.float32)
    a_low, a_high = np.percentile(a[valid], [10, 90])
    b_low, b_high = np.percentile(b[valid], [10, 90])
    scale = np.clip((a_high - a_low) / max(b_high - b_low, 12), .65, 1.55)
    offset = np.median(a[valid]) - np.median(b[valid]) * scale
    return a / 255, np.clip(b * scale + offset, 0, 255) / 255


def difference(reference, candidate, valid, kind):
    if kind == "rgb":
        score = np.abs(reference.astype(np.float32) - candidate).mean(axis=2) / 255
    else:
        a, b = robust_luminance(reference, candidate, valid)
        if kind == "gray":
            score = np.abs(a - b)
        else:
            blur = lambda x: cv2.GaussianBlur(x, (7, 7), 1.35)
            ma, mb = blur(a), blur(b)
            va, vb = np.maximum(0, blur(a * a) - ma ** 2), np.maximum(0, blur(b * b) - mb ** 2)
            cov = blur(a * b) - ma * mb
            similarity = ((2 * ma * mb + .01 ** 2) * (2 * cov + .03 ** 2)
                          / ((ma ** 2 + mb ** 2 + .01 ** 2) * (va + vb + .03 ** 2)))
            score = np.clip((1 - similarity) / 2, 0, 1)
    score[~valid] = 0
    return score


def score_map(score, valid):
    values = score[valid]
    n = min(len(values), max(1, round(.005 * len(values))))
    return float(np.partition(values, -n)[-n:].mean()) if n else 0.


def regions(score, valid, threshold):
    mask = ((score > threshold) & valid).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros(count, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 5
    return keep[labels] & valid
