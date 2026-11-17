package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Bounding boxes for object detection. Boxes are rows {@code [x1, y1, x2, y2]} (corners,
 * {@code x2 >= x1}) unless a method says {@code cxcywh} (center and size). Overlaps and
 * losses are differentiable; non-maximum suppression picks indices on the host.
 */
public final class Boxes {

    private Boxes() {
    }

    /**
     * @param cxcywh {@code [n, 4]} center x, center y, width, height
     * @return the same boxes as corners
     */
    public static Var toCorners(Var cxcywh) {
        Var c = cxcywh.slice(-1, 0, 2);
        Var half = cxcywh.slice(-1, 2, 4).mul(0.5f);
        return Var.concat(-1, c.sub(half), c.add(half));
    }

    /**
     * @param corners {@code [n, 4]} corners
     * @return the same boxes as center and size
     */
    public static Var toCenters(Var corners) {
        Var lo = corners.slice(-1, 0, 2);
        Var hi = corners.slice(-1, 2, 4);
        return Var.concat(-1, lo.add(hi).mul(0.5f), hi.sub(lo));
    }

    /**
     * @param boxes {@code [n, 4]} corners
     * @return each box's area {@code [n]}
     */
    public static Var area(Var boxes) {
        Var size = boxes.slice(-1, 2, 4).sub(boxes.slice(-1, 0, 2)).maximum(0);
        int n = boxes.shape().dim(0);
        return size.slice(-1, 0, 1).mul(size.slice(-1, 1, 2)).reshape(n);
    }

    /**
     * The overlap of each box with each other box: intersection over union.
     *
     * @param a {@code [n, 4]} corners
     * @param b {@code [m, 4]} corners
     * @return {@code [n, m]}
     */
    public static Var iou(Var a, Var b) {
        Var[] r = pairwise(a, b);
        return r[0].div(r[1].maximum(1e-9f));
    }

    /**
     * Generalized IoU (Rezatofighi et al., 2019): IoU minus the share of the smallest box
     * around both that neither covers; in {@code [-1, 1]}, and informative even for boxes
     * that do not overlap.
     *
     * @param a {@code [n, 4]} corners
     * @param b {@code [m, 4]} corners
     * @return {@code [n, m]}
     */
    public static Var giou(Var a, Var b) {
        Var[] r = pairwise(a, b);
        Var union = r[1].maximum(1e-9f);
        Var iou = r[0].div(union);
        Var hull = r[2].maximum(1e-9f);
        return iou.sub(hull.sub(union).div(hull));
    }

    /**
     * {@code 1 - GIoU} of matching rows, averaged: a box regression loss.
     *
     * @param predicted {@code [n, 4]} corners
     * @param target {@code [n, 4]} corners
     * @return the loss, a scalar
     */
    public static Var giouLoss(Var predicted, Var target) {
        Var[] r = elementwise(predicted, target);
        Var union = r[1].maximum(1e-9f);
        Var hull = r[2].maximum(1e-9f);
        Var giou = r[0].div(union).sub(hull.sub(union).div(hull));
        return giou.neg().add(1).mean();
    }

    /** Intersection, union and hull area of every pair: {@code [n, m]} each. */
    private static Var[] pairwise(Var a, Var b) {
        int n = a.shape().dim(0);
        int m = b.shape().dim(0);
        Var ea = a.reshape(n, 1, 4).expand(Shape.of(n, m, 4));
        Var eb = b.reshape(1, m, 4).expand(Shape.of(n, m, 4));
        Var[] r = elementwise(ea, eb);
        return new Var[] {r[0].reshape(n, m), r[1].reshape(n, m), r[2].reshape(n, m)};
    }

    /** Intersection, union and hull area of matching boxes along the last axis. */
    private static Var[] elementwise(Var a, Var b) {
        Var aLo = a.slice(-1, 0, 2);
        Var aHi = a.slice(-1, 2, 4);
        Var bLo = b.slice(-1, 0, 2);
        Var bHi = b.slice(-1, 2, 4);
        Var inter = product(aHi.minimum(bHi).sub(aLo.maximum(bLo)).maximum(0));
        Var areaA = product(aHi.sub(aLo).maximum(0));
        Var areaB = product(bHi.sub(bLo).maximum(0));
        Var union = areaA.add(areaB).sub(inter);
        Var hull = product(aHi.maximum(bHi).sub(aLo.minimum(bLo)));
        return new Var[] {inter, union, hull};
    }

    /** Width times height along the last axis of size 2, keeping it as size 1. */
    private static Var product(Var wh) {
        return wh.slice(-1, 0, 1).mul(wh.slice(-1, 1, 2));
    }

    /**
     * Regression targets of boxes relative to anchors, as Faster R-CNN encodes them:
     * {@code (dx, dy) = (center - anchor center) / anchor size} and {@code (dw, dh) =
     * log(size / anchor size)}.
     *
     * @param boxes {@code [n, 4]} corners
     * @param anchors {@code [n, 4]} corners
     * @return {@code [n, 4]} deltas
     */
    public static Var encode(Var boxes, Var anchors) {
        Var b = toCenters(boxes);
        Var a = toCenters(anchors);
        Var aSize = a.slice(-1, 2, 4);
        Var d = b.slice(-1, 0, 2).sub(a.slice(-1, 0, 2)).div(aSize);
        Var s = b.slice(-1, 2, 4).div(aSize).log();
        return Var.concat(-1, d, s);
    }

    /**
     * The inverse of {@link #encode}.
     *
     * @param deltas {@code [n, 4]}
     * @param anchors {@code [n, 4]} corners
     * @return {@code [n, 4]} corners
     */
    public static Var decode(Var deltas, Var anchors) {
        Var a = toCenters(anchors);
        Var aSize = a.slice(-1, 2, 4);
        Var c = deltas.slice(-1, 0, 2).mul(aSize).add(a.slice(-1, 0, 2));
        Var s = deltas.slice(-1, 2, 4).exp().mul(aSize);
        return toCorners(Var.concat(-1, c, s));
    }

    /**
     * Anchor boxes on a grid, as a detection head's feature map lays them out: for every cell
     * (row-major) every size and aspect ratio, centered on the cell.
     *
     * @param rows the feature map's rows
     * @param cols its columns
     * @param stride image pixels per cell
     * @param sizes anchor side lengths in pixels (the square root of the area)
     * @param ratios height over width
     * @return {@code [rows * cols * sizes * ratios, 4]} corners
     */
    public static Tensor anchors(int rows, int cols, float stride, float[] sizes, float[] ratios) {
        int per = sizes.length * ratios.length;
        float[] out = new float[rows * cols * per * 4];
        int i = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                float cx = (c + 0.5f) * stride;
                float cy = (r + 0.5f) * stride;
                for (float s : sizes) {
                    for (float ratio : ratios) {
                        float w = s / (float) Math.sqrt(ratio);
                        float h = s * (float) Math.sqrt(ratio);
                        out[i++] = cx - w / 2;
                        out[i++] = cy - h / 2;
                        out[i++] = cx + w / 2;
                        out[i++] = cy + h / 2;
                    }
                }
            }
        }
        return Tensor.of(Shape.of(rows * cols * per, 4), out);
    }

    /**
     * Non-maximum suppression: boxes in order of score, each dropped if it overlaps a kept
     * box by more than {@code threshold}.
     *
     * @param boxes {@code [n, 4]} corners
     * @param scores {@code [n]}
     * @param threshold the IoU above which the lower-scored box goes
     * @param scoreThreshold boxes scoring below it are dropped first
     * @param max the most boxes to keep
     * @return the kept indices, best first
     */
    public static int[] nms(Tensor boxes, Tensor scores, float threshold, float scoreThreshold, int max) {
        return nms(boxes.toArray(), scores.toArray(), null, threshold, scoreThreshold, max);
    }

    /**
     * Non-maximum suppression per class: boxes of different classes never suppress each
     * other.
     *
     * @param boxes {@code [n, 4]} corners
     * @param scores {@code [n]}
     * @param classes the class of each box
     * @param threshold the IoU above which the lower-scored box goes
     * @param scoreThreshold boxes scoring below it are dropped first
     * @param max the most boxes to keep, over all classes
     * @return the kept indices, best first
     */
    public static int[] nms(Tensor boxes, Tensor scores, int[] classes, float threshold, float scoreThreshold,
            int max) {
        return nms(boxes.toArray(), scores.toArray(), classes, threshold, scoreThreshold, max);
    }

    /**
     * {@link #nms(Tensor, Tensor, int[], float, float, int)} on host arrays.
     *
     * @param boxes {@code 4n} corners
     * @param scores {@code n} scores
     * @param classes the class of each box, or {@code null} for one class
     * @param threshold the IoU above which the lower-scored box goes
     * @param scoreThreshold boxes scoring below it are dropped first
     * @param max the most boxes to keep
     * @return the kept indices, best first
     */
    public static int[] nms(float[] boxes, float[] scores, int[] classes, float threshold, float scoreThreshold,
            int max) {
        int n = scores.length;
        if (boxes.length != 4 * n || classes != null && classes.length != n) {
            throw new IllegalArgumentException(n + " scores need " + 4 * n + " box values, got " + boxes.length);
        }
        Integer[] order = new Integer[n];
        int candidates = 0;
        for (int i = 0; i < n; i++) {
            if (scores[i] >= scoreThreshold) {
                order[candidates++] = i;
            }
        }
        Integer[] sorted = Arrays.copyOf(order, candidates);
        Arrays.sort(sorted, Comparator.comparingDouble((Integer i) -> -scores[i]));
        List<Integer> kept = new ArrayList<>();
        for (int i : sorted) {
            if (kept.size() >= max) {
                break;
            }
            boolean keep = true;
            for (int k : kept) {
                if ((classes == null || classes[k] == classes[i]) && iou(boxes, i, k) > threshold) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                kept.add(i);
            }
        }
        return kept.stream().mapToInt(Integer::intValue).toArray();
    }

    private static float iou(float[] b, int i, int k) {
        float ix = Math.max(0, Math.min(b[4 * i + 2], b[4 * k + 2]) - Math.max(b[4 * i], b[4 * k]));
        float iy = Math.max(0, Math.min(b[4 * i + 3], b[4 * k + 3]) - Math.max(b[4 * i + 1], b[4 * k + 1]));
        float inter = ix * iy;
        float a = Math.max(0, b[4 * i + 2] - b[4 * i]) * Math.max(0, b[4 * i + 3] - b[4 * i + 1]);
        float c = Math.max(0, b[4 * k + 2] - b[4 * k]) * Math.max(0, b[4 * k + 3] - b[4 * k + 1]);
        float union = a + c - inter;
        return union > 0 ? inter / union : 0;
    }
}
