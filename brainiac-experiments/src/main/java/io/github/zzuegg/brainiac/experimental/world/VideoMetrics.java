package io.github.zzuegg.brainiac.experimental.world;

import java.util.ArrayList;
import java.util.List;

/**
 * Frozen scores for generated videos of bright objects on a dark background: a fixed blob
 * detector (threshold, 4-connected components, a minimum area) that counts objects and finds
 * their centroids, a nearest-neighbour tracker that measures how far objects deviate from
 * constant-velocity motion and how often a track breaks, and a kernel maximum mean discrepancy
 * between sets of feature vectors, the FVD-style distribution distance of the clip features
 * {@link #clipFeatures} or the frame features {@link #frameFeatures}. Nothing here is trained,
 * so the scores mean the same for every generator; compare them to the same scores on real
 * clips.
 */
public final class VideoMetrics {

    private VideoMetrics() {
    }

    /**
     * Object centroids in a frame.
     *
     * @param frame {@code side * side} pixels
     * @param side the frame side
     * @param threshold pixels above it are foreground
     * @param minPixels the smallest blob counted
     * @return {@code [blobs][2]} centroids {@code (x, y)}
     */
    public static double[][] blobs(float[] frame, int side, float threshold, int minPixels) {
        int[] label = new int[side * side];
        List<double[]> out = new ArrayList<>();
        int[] stack = new int[side * side];
        int next = 0;
        for (int start = 0; start < frame.length; start++) {
            if (frame[start] <= threshold || label[start] != 0) {
                continue;
            }
            next++;
            int top = 0;
            stack[top++] = start;
            label[start] = next;
            double sx = 0;
            double sy = 0;
            int n = 0;
            while (top > 0) {
                int p = stack[--top];
                int x = p % side;
                int y = p / side;
                sx += x;
                sy += y;
                n++;
                int[] nb = {x > 0 ? p - 1 : -1, x < side - 1 ? p + 1 : -1, y > 0 ? p - side : -1,
                    y < side - 1 ? p + side : -1};
                for (int q : nb) {
                    if (q >= 0 && label[q] == 0 && frame[q] > threshold) {
                        label[q] = next;
                        stack[top++] = q;
                    }
                }
            }
            if (n >= minPixels) {
                out.add(new double[] {sx / n, sy / n});
            }
        }
        return out.toArray(new double[0][]);
    }

    /**
     * Scores of one clip.
     *
     * @param countError the mean absolute difference between the detected and the true object
     *     count per frame
     * @param velocityError the mean distance between each object and its constant-velocity
     *     extrapolation from the two frames before (pixels)
     * @param trackBreaks the share of detections with no detection near them in the frame before
     * @param counts the detected count per frame
     */
    public record ClipScore(double countError, double velocityError, double trackBreaks, int[] counts) {
    }

    /**
     * Scores a clip with the frozen detector and tracker.
     *
     * @param clip {@code [T][side * side]} frames
     * @param side the frame side
     * @param trueCount the number of objects the clip should show
     * @param threshold the detector threshold
     * @param minPixels the smallest blob counted
     * @param radius the largest frame-to-frame motion that keeps a track
     * @return the scores
     */
    public static ClipScore score(float[][] clip, int side, int trueCount, float threshold, int minPixels,
            double radius) {
        int t = clip.length;
        double[][][] b = new double[t][][];
        int[] counts = new int[t];
        double countError = 0;
        for (int i = 0; i < t; i++) {
            b[i] = blobs(clip[i], side, threshold, minPixels);
            counts[i] = b[i].length;
            countError += Math.abs(counts[i] - trueCount);
        }
        double vel = 0;
        int velN = 0;
        int breaks = 0;
        int dets = 0;
        for (int i = 1; i < t; i++) {
            for (double[] a : b[i]) {
                dets++;
                double[] prev = nearest(b[i - 1], a);
                if (prev == null || dist(prev, a) > radius) {
                    breaks++;
                    continue;
                }
                if (i + 1 < t && b[i + 1].length > 0) {
                    double[] pred = {2 * a[0] - prev[0], 2 * a[1] - prev[1]};
                    double[] hit = nearest(b[i + 1], pred);
                    vel += Math.min(dist(hit, pred), radius);
                    velN++;
                }
            }
        }
        return new ClipScore(countError / t, velN == 0 ? radius : vel / velN, dets == 0 ? 1 : breaks / (double) dets,
                counts);
    }

    private static double[] nearest(double[][] set, double[] p) {
        double best = Double.MAX_VALUE;
        double[] out = null;
        for (double[] q : set) {
            double d = dist(q, p);
            if (d < best) {
                best = d;
                out = q;
            }
        }
        return out;
    }

    private static double dist(double[] a, double[] b) {
        return Math.hypot(a[0] - b[0], a[1] - b[1]);
    }

    /**
     * Averages {@link ClipScore}s.
     *
     * @param scores the per-clip scores
     * @return {@code {countError, velocityError, trackBreaks}}
     */
    public static double[] mean(List<ClipScore> scores) {
        double[] m = new double[3];
        for (ClipScore s : scores) {
            m[0] += s.countError() / scores.size();
            m[1] += s.velocityError() / scores.size();
            m[2] += s.trackBreaks() / scores.size();
        }
        return m;
    }

    private static float[] pool(float[] frame, int side, int cells) {
        int c = side / cells;
        float[] out = new float[cells * cells];
        for (int y = 0; y < side; y++) {
            for (int x = 0; x < side; x++) {
                out[(y / c) * cells + x / c] += frame[y * side + x] / (c * c);
            }
        }
        return out;
    }

    /**
     * A frame's features: the frame average-pooled to {@code 8 x 8}.
     *
     * @param frame the pixels
     * @param side the side, a multiple of 8
     * @return 64 features
     */
    public static float[] frameFeatures(float[] frame, int side) {
        return pool(frame, side, 8);
    }

    /**
     * A clip's features for a Fréchet-video-distance-style comparison: every frame pooled to
     * {@code 4 x 4}, and every frame difference pooled to {@code 4 x 4} (motion energy), so
     * both appearance and dynamics enter.
     *
     * @param clip {@code [T][side * side]}
     * @param side the side, a multiple of 4
     * @return {@code 32 T - 16} features
     */
    public static float[] clipFeatures(float[][] clip, int side) {
        int t = clip.length;
        float[] out = new float[16 * t + 16 * (t - 1)];
        for (int i = 0; i < t; i++) {
            System.arraycopy(pool(clip[i], side, 4), 0, out, 16 * i, 16);
        }
        for (int i = 1; i < t; i++) {
            float[] d = new float[side * side];
            for (int p = 0; p < d.length; p++) {
                d[p] = 2 * Math.abs(clip[i][p] - clip[i - 1][p]);
            }
            System.arraycopy(pool(d, side, 4), 0, out, 16 * t + 16 * (i - 1), 16);
        }
        return out;
    }

    /**
     * The squared maximum mean discrepancy between two sets with a Gaussian kernel whose
     * bandwidth is the median pairwise distance of the pooled set (the biased estimate, which is
     * 0 only for identical sets).
     *
     * @param a one set
     * @param b the other
     * @return MMD squared
     */
    public static double mmd(float[][] a, float[][] b) {
        List<float[]> all = new ArrayList<>(List.of(a));
        all.addAll(List.of(b));
        List<Double> d = new ArrayList<>();
        for (int i = 0; i < all.size(); i++) {
            for (int j = i + 1; j < all.size(); j++) {
                d.add(sq(all.get(i), all.get(j)));
            }
        }
        java.util.Collections.sort(d);
        double h = Math.max(1e-9, d.get(d.size() / 2));
        return kernelMean(a, a, h) + kernelMean(b, b, h) - 2 * kernelMean(a, b, h);
    }

    private static double kernelMean(float[][] a, float[][] b, double h) {
        double s = 0;
        for (float[] x : a) {
            for (float[] y : b) {
                s += Math.exp(-sq(x, y) / h);
            }
        }
        return s / (a.length * (double) b.length);
    }

    private static double sq(float[] a, float[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            double e = a[i] - b[i];
            s += e * e;
        }
        return s;
    }
}
