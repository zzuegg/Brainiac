package io.github.zzuegg.brainiac.experimental.vision;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Classical sparse matching and robust homography estimation.
 *
 * <ul>
 *   <li>{@link #harris}: the corner response {@code det M - k tr(M)^2} of the Gaussian-weighted
 *       structure tensor, and {@link #keypoints}: its strongest local maxima;
 *   <li>{@link #patches}: zero-mean unit-norm patch descriptors; {@link #brief}: BRIEF-like
 *       binary descriptors from random pixel-pair comparisons on the smoothed image;
 *   <li>{@link #match}/{@link #matchBinary}: mutual nearest neighbours with Lowe's ratio test;
 *   <li>{@link #dlt}: the normalised direct linear transform homography (least squares on any
 *       number of matches), {@link #ransac}: four-point RANSAC with adaptive iteration count
 *       and a least-squares refit on the inliers.
 * </ul>
 *
 * <p>Points are {@code {x, y}} in pixels; homographies are row-major {@code 3 x 3} arrays
 * mapping the first image to the second.
 *
 * <p>Papers: Harris, Stephens, "A Combined Corner and Edge Detector", AVC 1988; Calonder et
 * al., "BRIEF", ECCV 2010; Fischler, Bolles, "Random Sample Consensus", CACM 1981.
 */
public final class FeatureMatching {

    private FeatureMatching() {
    }

    /**
     * @param img row-major {@code h x w}
     * @param h rows
     * @param w columns
     * @param sigma the integration scale
     * @param k the trace weight, such as 0.05
     * @return the response
     */
    public static float[] harris(float[] img, int h, int w, double sigma, double k) {
        float[][] g = Conv2dOps.sobel(Conv2dOps.gaussianBlur(img, h, w, 0.7), h, w);
        float[] xx = new float[h * w];
        float[] yy = new float[h * w];
        float[] xy = new float[h * w];
        for (int i = 0; i < xx.length; i++) {
            xx[i] = g[0][i] * g[0][i];
            yy[i] = g[1][i] * g[1][i];
            xy[i] = g[0][i] * g[1][i];
        }
        xx = Conv2dOps.gaussianBlur(xx, h, w, sigma);
        yy = Conv2dOps.gaussianBlur(yy, h, w, sigma);
        xy = Conv2dOps.gaussianBlur(xy, h, w, sigma);
        float[] r = new float[h * w];
        for (int i = 0; i < r.length; i++) {
            double tr = xx[i] + yy[i];
            r[i] = (float) (xx[i] * yy[i] - xy[i] * xy[i] - k * tr * tr);
        }
        return r;
    }

    /**
     * The strongest local maxima of a response map.
     *
     * @param response the map
     * @param h rows
     * @param w columns
     * @param count the most to return
     * @param radius the suppression radius
     * @param border pixels skipped at the border
     * @return points {@code {x, y}}, strongest first
     */
    public static List<double[]> keypoints(float[] response, int h, int w, int count, int radius, int border) {
        List<int[]> cand = new ArrayList<>();
        for (int y = border; y < h - border; y++) {
            for (int x = border; x < w - border; x++) {
                float v = response[y * w + x];
                if (v <= 0) {
                    continue;
                }
                boolean max = true;
                for (int dy = -radius; dy <= radius && max; dy++) {
                    for (int dx = -radius; dx <= radius; dx++) {
                        int yy = y + dy;
                        int xx = x + dx;
                        if ((dx != 0 || dy != 0) && yy >= 0 && xx >= 0 && yy < h && xx < w
                                && response[yy * w + xx] > v) {
                            max = false;
                            break;
                        }
                    }
                }
                if (max) {
                    cand.add(new int[] {x, y});
                }
            }
        }
        cand.sort((a, b) -> Float.compare(response[b[1] * w + b[0]], response[a[1] * w + a[0]]));
        List<double[]> out = new ArrayList<>();
        for (int i = 0; i < Math.min(count, cand.size()); i++) {
            out.add(new double[] {cand.get(i)[0], cand.get(i)[1]});
        }
        return out;
    }

    /**
     * Normalised patch descriptors.
     *
     * @param img row-major {@code h x w}
     * @param h rows
     * @param w columns
     * @param points the points
     * @param radius the patch radius
     * @return per point its descriptor
     */
    public static float[][] patches(float[] img, int h, int w, List<double[]> points, int radius) {
        float[] s = Conv2dOps.gaussianBlur(img, h, w, 0.8);
        int side = 2 * radius + 1;
        float[][] out = new float[points.size()][side * side];
        for (int p = 0; p < points.size(); p++) {
            double[] pt = points.get(p);
            double mean = 0;
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    float v = Conv2dOps.bilinear(s, h, w, pt[0] + dx, pt[1] + dy, 0);
                    out[p][(dy + radius) * side + dx + radius] = v;
                    mean += v;
                }
            }
            mean /= side * side;
            double norm = 0;
            for (int i = 0; i < out[p].length; i++) {
                out[p][i] -= (float) mean;
                norm += out[p][i] * out[p][i];
            }
            float inv = (float) (1 / Math.sqrt(norm + 1e-12));
            for (int i = 0; i < out[p].length; i++) {
                out[p][i] *= inv;
            }
        }
        return out;
    }

    /**
     * BRIEF-like binary descriptors.
     *
     * @param img row-major {@code h x w}
     * @param h rows
     * @param w columns
     * @param points the points
     * @param radius the sampling radius
     * @param bits the descriptor length (a multiple of 64)
     * @param seed the seed of the comparison pattern (use the same for both images)
     * @return per point its packed bits
     */
    public static long[][] brief(float[] img, int h, int w, List<double[]> points, int radius, int bits, long seed) {
        float[] s = Conv2dOps.gaussianBlur(img, h, w, 1.2);
        SplittableRandom r = new SplittableRandom(seed);
        double[][] pairs = new double[bits][4];
        for (double[] q : pairs) {
            for (int k = 0; k < 4; k++) {
                q[k] = Math.max(-radius, Math.min(radius, r.nextGaussian() * radius / 2.0));
            }
        }
        long[][] out = new long[points.size()][bits / 64];
        for (int p = 0; p < points.size(); p++) {
            double[] pt = points.get(p);
            for (int b = 0; b < bits; b++) {
                float a = Conv2dOps.bilinear(s, h, w, pt[0] + pairs[b][0], pt[1] + pairs[b][1], 0);
                float c = Conv2dOps.bilinear(s, h, w, pt[0] + pairs[b][2], pt[1] + pairs[b][3], 0);
                if (a < c) {
                    out[p][b >> 6] |= 1L << (b & 63);
                }
            }
        }
        return out;
    }

    /**
     * Mutual nearest neighbours by Euclidean distance with a ratio test.
     *
     * @param a descriptors of the first image
     * @param b descriptors of the second
     * @param ratio the largest ratio of best to second-best distance
     * @return matches {@code {i, j}}
     */
    public static List<int[]> match(float[][] a, float[][] b, double ratio) {
        double[][] d = new double[a.length][b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                double s = 0;
                for (int k = 0; k < a[i].length; k++) {
                    s += (a[i][k] - b[j][k]) * (a[i][k] - b[j][k]);
                }
                d[i][j] = Math.sqrt(s);
            }
        }
        return mutual(d, ratio);
    }

    /**
     * Mutual nearest neighbours by Hamming distance with a ratio test.
     *
     * @param a descriptors of the first image
     * @param b descriptors of the second
     * @param ratio the largest ratio of best to second-best distance
     * @return matches {@code {i, j}}
     */
    public static List<int[]> matchBinary(long[][] a, long[][] b, double ratio) {
        double[][] d = new double[a.length][b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                int s = 0;
                for (int k = 0; k < a[i].length; k++) {
                    s += Long.bitCount(a[i][k] ^ b[j][k]);
                }
                d[i][j] = s;
            }
        }
        return mutual(d, ratio);
    }

    private static List<int[]> mutual(double[][] d, double ratio) {
        List<int[]> out = new ArrayList<>();
        if (d.length == 0 || d[0].length == 0) {
            return out;
        }
        int m = d[0].length;
        int[] bestOfB = new int[m];
        for (int j = 0; j < m; j++) {
            int best = 0;
            for (int i = 1; i < d.length; i++) {
                if (d[i][j] < d[best][j]) {
                    best = i;
                }
            }
            bestOfB[j] = best;
        }
        for (int i = 0; i < d.length; i++) {
            int best = 0;
            double second = Double.MAX_VALUE;
            for (int j = 1; j < m; j++) {
                if (d[i][j] < d[i][best]) {
                    second = d[i][best];
                    best = j;
                } else if (d[i][j] < second) {
                    second = d[i][j];
                }
            }
            if (bestOfB[best] == i && (m == 1 || d[i][best] <= ratio * second)) {
                out.add(new int[] {i, best});
            }
        }
        return out;
    }

    // ---------------------------------------------------------------- geometry

    /**
     * @param hm a homography
     * @param p a point
     * @return the mapped point
     */
    public static double[] apply(double[] hm, double[] p) {
        double z = hm[6] * p[0] + hm[7] * p[1] + hm[8];
        return new double[] {(hm[0] * p[0] + hm[1] * p[1] + hm[2]) / z, (hm[3] * p[0] + hm[4] * p[1] + hm[5]) / z};
    }

    /**
     * @param hm a homography
     * @return its inverse (scaled so the last entry is 1 where possible)
     */
    public static double[] invert(double[] hm) {
        double a = hm[0];
        double b = hm[1];
        double c = hm[2];
        double d = hm[3];
        double e = hm[4];
        double f = hm[5];
        double g = hm[6];
        double h = hm[7];
        double i = hm[8];
        double[] inv = {e * i - f * h, c * h - b * i, b * f - c * e, f * g - d * i, a * i - c * g, c * d - a * f,
                d * h - e * g, b * g - a * h, a * e - b * d};
        double s = Math.abs(inv[8]) > 1e-12 ? inv[8] : 1;
        for (int k = 0; k < 9; k++) {
            inv[k] /= s;
        }
        return inv;
    }

    private static double[] normaliser(List<double[]> pts) {
        double mx = 0;
        double my = 0;
        for (double[] p : pts) {
            mx += p[0] / pts.size();
            my += p[1] / pts.size();
        }
        double d = 0;
        for (double[] p : pts) {
            d += Math.hypot(p[0] - mx, p[1] - my) / pts.size();
        }
        double s = Math.sqrt(2) / Math.max(d, 1e-9);
        return new double[] {s, 0, -s * mx, 0, s, -s * my, 0, 0, 1};
    }

    private static double[] mul(double[] a, double[] b) {
        double[] c = new double[9];
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                for (int k = 0; k < 3; k++) {
                    c[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
                }
            }
        }
        return c;
    }

    /**
     * The normalised DLT homography (least squares with {@code h33 = 1}).
     *
     * @param src points in the first image (at least four)
     * @param dst corresponding points in the second
     * @return the homography, or {@code null} if degenerate
     */
    public static double[] dlt(List<double[]> src, List<double[]> dst) {
        double[] ts = normaliser(src);
        double[] td = normaliser(dst);
        double[][] ata = new double[8][8];
        double[] atb = new double[8];
        for (int n = 0; n < src.size(); n++) {
            double[] p = apply(ts, src.get(n));
            double[] q = apply(td, dst.get(n));
            double[][] rows = {{p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]},
                {0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]}};
            double[] rhs = {q[0], q[1]};
            for (int r = 0; r < 2; r++) {
                for (int i = 0; i < 8; i++) {
                    atb[i] += rows[r][i] * rhs[r];
                    for (int j = 0; j < 8; j++) {
                        ata[i][j] += rows[r][i] * rows[r][j];
                    }
                }
            }
        }
        double[] x = solve(ata, atb);
        if (x == null) {
            return null;
        }
        double[] hn = {x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], 1};
        double[] hm = mul(invert(td), mul(hn, ts));
        double s = Math.abs(hm[8]) > 1e-12 ? hm[8] : 1;
        for (int k = 0; k < 9; k++) {
            hm[k] /= s;
        }
        return hm;
    }

    private static double[] solve(double[][] a0, double[] b0) {
        int n = b0.length;
        double[][] a = new double[n][n + 1];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a0[i], 0, a[i], 0, n);
            a[i][n] = b0[i];
        }
        for (int c = 0; c < n; c++) {
            int piv = c;
            for (int i = c + 1; i < n; i++) {
                if (Math.abs(a[i][c]) > Math.abs(a[piv][c])) {
                    piv = i;
                }
            }
            if (Math.abs(a[piv][c]) < 1e-12) {
                return null;
            }
            double[] t = a[c];
            a[c] = a[piv];
            a[piv] = t;
            for (int i = c + 1; i < n; i++) {
                double f = a[i][c] / a[c][c];
                for (int j = c; j <= n; j++) {
                    a[i][j] -= f * a[c][j];
                }
            }
        }
        double[] x = new double[n];
        for (int i = n - 1; i >= 0; i--) {
            double s = a[i][n];
            for (int j = i + 1; j < n; j++) {
                s -= a[i][j] * x[j];
            }
            x[i] = s / a[i][i];
        }
        return x;
    }

    /**
     * A RANSAC fit.
     *
     * @param homography the refitted model, or {@code null}
     * @param inliers per match whether it is an inlier
     * @param iterations the iterations run
     */
    public record Fit(double[] homography, boolean[] inliers, int iterations) {

        /** @return the number of inliers */
        public int count() {
            int c = 0;
            for (boolean b : inliers) {
                c += b ? 1 : 0;
            }
            return c;
        }
    }

    /**
     * Four-point RANSAC.
     *
     * @param src matched points in the first image
     * @param dst matched points in the second
     * @param threshold the inlier reprojection error in pixels
     * @param maxIterations the most iterations
     * @param confidence the stopping confidence, such as 0.99
     * @param seed the seed
     * @return the fit
     */
    public static Fit ransac(List<double[]> src, List<double[]> dst, double threshold, int maxIterations,
            double confidence, long seed) {
        int n = src.size();
        boolean[] best = new boolean[n];
        int bestCount = -1;
        if (n < 4) {
            return new Fit(null, best, 0);
        }
        SplittableRandom r = new SplittableRandom(seed);
        int needed = maxIterations;
        int it = 0;
        for (; it < Math.min(needed, maxIterations); it++) {
            int[] pick = new int[4];
            for (int k = 0; k < 4; k++) {
                boolean dup;
                do {
                    pick[k] = r.nextInt(n);
                    dup = false;
                    for (int q = 0; q < k; q++) {
                        dup |= pick[q] == pick[k];
                    }
                } while (dup);
            }
            List<double[]> s = new ArrayList<>();
            List<double[]> d = new ArrayList<>();
            for (int k : pick) {
                s.add(src.get(k));
                d.add(dst.get(k));
            }
            double[] hm = dlt(s, d);
            if (hm == null) {
                continue;
            }
            boolean[] in = new boolean[n];
            int count = 0;
            for (int k = 0; k < n; k++) {
                double[] p = apply(hm, src.get(k));
                in[k] = Math.hypot(p[0] - dst.get(k)[0], p[1] - dst.get(k)[1]) < threshold;
                count += in[k] ? 1 : 0;
            }
            if (count > bestCount) {
                bestCount = count;
                best = in;
                double w = count / (double) n;
                double p4 = Math.pow(w, 4);
                needed = p4 >= 1 ? 1 : (int) Math.ceil(Math.log(1 - confidence) / Math.log(1 - Math.max(p4, 1e-9)));
            }
        }
        if (bestCount < 4) {
            return new Fit(null, best, it);
        }
        List<double[]> s = new ArrayList<>();
        List<double[]> d = new ArrayList<>();
        for (int k = 0; k < n; k++) {
            if (best[k]) {
                s.add(src.get(k));
                d.add(dst.get(k));
            }
        }
        return new Fit(dlt(s, d), best, it);
    }

    /**
     * The mean distance between the image corners mapped by an estimate and by the truth.
     *
     * @param estimate the estimate ({@code null} counts as infinitely wrong)
     * @param truth the true homography
     * @param w the width
     * @param h the height
     * @return the corner error in pixels
     */
    public static double cornerError(double[] estimate, double[] truth, int w, int h) {
        if (estimate == null) {
            return Double.POSITIVE_INFINITY;
        }
        double[][] corners = {{0, 0}, {w - 1, 0}, {0, h - 1}, {w - 1, h - 1}};
        double e = 0;
        for (double[] c : corners) {
            double[] a = apply(estimate, c);
            double[] b = apply(truth, c);
            e += Math.hypot(a[0] - b[0], a[1] - b[1]) / 4;
        }
        return Double.isNaN(e) ? Double.POSITIVE_INFINITY : e;
    }

    /**
     * Warps an image: {@code out(x) = img(H^-1 x)}.
     *
     * @param img row-major {@code h x w}
     * @param h rows
     * @param w columns
     * @param hm the homography from the source to the output
     * @param outside the value outside the source
     * @return the warped image
     */
    public static float[] warp(float[] img, int h, int w, double[] hm, float outside) {
        double[] inv = invert(hm);
        float[] out = new float[h * w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double[] p = apply(inv, new double[] {x, y});
                out[y * w + x] = Conv2dOps.bilinear(img, h, w, p[0], p[1], outside);
            }
        }
        return out;
    }
}
