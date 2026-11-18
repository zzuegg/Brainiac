package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Edge detectors and boundary metrics.
 *
 * <ul>
 *   <li>{@link #sobel}: Gaussian-smoothed Sobel gradient magnitude; {@link #laplacian}: the
 *       magnitude of the Laplacian of Gaussian;
 *   <li>{@link #canny}: smoothing, gradient magnitude and direction, non-maximum suppression
 *       along the gradient, and hysteresis linking between a low and a high threshold;
 *   <li>{@link Hed}: holistically-nested edge detection, a three-stage fully convolutional
 *       network whose side outputs (each upsampled to full size) are deeply supervised with a
 *       class-balanced cross-entropy and fused by a learned {@code 1 x 1} convolution; with
 *       {@code deepSupervision = false} only the fused output is trained, with plain
 *       cross-entropy (the single-output FCN baseline).
 * </ul>
 *
 * <p>{@link #evaluate} computes ODS and OIS F-measures and average precision over thresholds,
 * matching predicted to true boundary pixels one-to-one within a distance tolerance; score
 * maps are thinned by non-maximum suppression first, as in the BSDS benchmark.
 *
 * <p>Papers: Canny, "A Computational Approach to Edge Detection", TPAMI 1986; Xie, Tu,
 * "Holistically-Nested Edge Detection", ICCV 2015,
 * <a href="https://arxiv.org/abs/1504.06375">arXiv:1504.06375</a>.
 */
public final class EdgeDetectors {

    private EdgeDetectors() {
    }

    /**
     * @param image row-major {@code h x w}
     * @param h rows
     * @param w columns
     * @param sigma the smoothing scale (0 for none)
     * @return {@code {magnitude, gx, gy}}
     */
    public static float[][] gradient(float[] image, int h, int w, double sigma) {
        float[] s = sigma > 0 ? Conv2dOps.gaussianBlur(image, h, w, sigma) : image;
        float[][] g = Conv2dOps.sobel(s, h, w);
        float[] m = new float[h * w];
        for (int i = 0; i < m.length; i++) {
            m[i] = (float) Math.hypot(g[0][i], g[1][i]);
        }
        return new float[][] {m, g[0], g[1]};
    }

    /**
     * @param image row-major {@code h x w}
     * @param h rows
     * @param w columns
     * @param sigma the smoothing scale
     * @return the gradient magnitude
     */
    public static float[] sobel(float[] image, int h, int w, double sigma) {
        return gradient(image, h, w, sigma)[0];
    }

    /**
     * @param image row-major {@code h x w}
     * @param h rows
     * @param w columns
     * @param sigma the smoothing scale
     * @return {@code |LoG|}
     */
    public static float[] laplacian(float[] image, int h, int w, double sigma) {
        float[] l = Conv2dOps.filter(Conv2dOps.gaussianBlur(image, h, w, sigma), h, w, Conv2dOps.LAPLACIAN, 3, 3);
        for (int i = 0; i < l.length; i++) {
            l[i] = Math.abs(l[i]);
        }
        return l;
    }

    /**
     * Non-maximum suppression along the gradient direction (four sectors).
     *
     * @param m the magnitude
     * @param gx horizontal gradient
     * @param gy vertical gradient
     * @param h rows
     * @param w columns
     * @return the thinned magnitude
     */
    public static float[] suppress(float[] m, float[] gx, float[] gy, int h, int w) {
        float[] out = new float[m.length];
        for (int y = 1; y + 1 < h; y++) {
            for (int x = 1; x + 1 < w; x++) {
                int i = y * w + x;
                double a = Math.toDegrees(Math.atan2(gy[i], gx[i]));
                a = a < 0 ? a + 180 : a;
                int dx;
                int dy;
                if (a < 22.5 || a >= 157.5) {
                    dx = 1;
                    dy = 0;
                } else if (a < 67.5) {
                    dx = 1;
                    dy = 1;
                } else if (a < 112.5) {
                    dx = 0;
                    dy = 1;
                } else {
                    dx = -1;
                    dy = 1;
                }
                float p = m[(y + dy) * w + x + dx];
                float q = m[(y - dy) * w + x - dx];
                if (m[i] >= p && m[i] >= q) {
                    out[i] = m[i];
                }
            }
        }
        return out;
    }

    /**
     * Thins any score map by suppression along its own gradient.
     *
     * @param score the map
     * @param h rows
     * @param w columns
     * @return the thinned map
     */
    public static float[] thin(float[] score, int h, int w) {
        float[][] g = gradient(score, h, w, 1.0);
        return suppress(score, g[1], g[2], h, w);
    }

    /**
     * Canny edges.
     *
     * @param image row-major {@code h x w}
     * @param h rows
     * @param w columns
     * @param sigma the smoothing scale
     * @param low the low threshold, relative to the largest magnitude
     * @param high the high threshold, relative to the largest magnitude
     * @return the edge pixels
     */
    public static boolean[] canny(float[] image, int h, int w, double sigma, double low, double high) {
        float[][] g = gradient(image, h, w, sigma);
        float[] nms = suppress(g[0], g[1], g[2], h, w);
        return hysteresis(nms, h, w, low, high);
    }

    /**
     * Hysteresis: pixels above {@code high} and those above {@code low} connected to them.
     *
     * @param nms the thinned magnitude
     * @param h rows
     * @param w columns
     * @param low relative low threshold
     * @param high relative high threshold
     * @return the edge pixels
     */
    public static boolean[] hysteresis(float[] nms, int h, int w, double low, double high) {
        float max = 1e-12f;
        for (float v : nms) {
            max = Math.max(max, v);
        }
        boolean[] e = new boolean[nms.length];
        java.util.ArrayDeque<Integer> stack = new java.util.ArrayDeque<>();
        for (int i = 0; i < nms.length; i++) {
            if (nms[i] >= high * max) {
                e[i] = true;
                stack.push(i);
            }
        }
        while (!stack.isEmpty()) {
            int i = stack.pop();
            int y = i / w;
            int x = i % w;
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int yy = y + dy;
                    int xx = x + dx;
                    if (yy >= 0 && xx >= 0 && yy < h && xx < w) {
                        int j = yy * w + xx;
                        if (!e[j] && nms[j] >= low * max) {
                            e[j] = true;
                            stack.push(j);
                        }
                    }
                }
            }
        }
        return e;
    }

    // ---------------------------------------------------------------- metrics

    /**
     * Boundary scores.
     *
     * @param ods F at the best dataset-wide threshold
     * @param ois mean of per-image best F
     * @param ap average precision over thresholds
     */
    public record Scores(double ods, double ois, double ap) {
    }

    /**
     * One-to-one matching within a radius (greedy, closest first).
     *
     * @param pred predicted edge pixels
     * @param gold true boundary pixels
     * @param h rows
     * @param w columns
     * @param radius the tolerance in pixels
     * @return {@code {matched predictions, predictions, matched truths, truths}}
     */
    public static int[] match(boolean[] pred, boolean[] gold, int h, int w, int radius) {
        boolean[] used = new boolean[gold.length];
        int mp = 0;
        int np = 0;
        int ng = 0;
        for (boolean g : gold) {
            ng += g ? 1 : 0;
        }
        for (int i = 0; i < pred.length; i++) {
            if (!pred[i]) {
                continue;
            }
            np++;
            int y = i / w;
            int x = i % w;
            int best = -1;
            int bestD = Integer.MAX_VALUE;
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    int yy = y + dy;
                    int xx = x + dx;
                    int d = dx * dx + dy * dy;
                    if (yy >= 0 && xx >= 0 && yy < h && xx < w && d <= radius * radius && d < bestD) {
                        int j = yy * w + xx;
                        if (gold[j] && !used[j]) {
                            best = j;
                            bestD = d;
                        }
                    }
                }
            }
            if (best >= 0) {
                used[best] = true;
                mp++;
            }
        }
        return new int[] {mp, np, mp, ng};
    }

    private static double f(double p, double r) {
        return p + r <= 0 ? 0 : 2 * p * r / (p + r);
    }

    /**
     * Evaluates per-threshold binary maps.
     *
     * @param binaries {@code [threshold][image]} edge maps
     * @param gold per image its boundary
     * @param h rows
     * @param w columns
     * @param radius the matching tolerance
     * @return the scores
     */
    public static Scores evaluate(List<List<boolean[]>> binaries, List<boolean[]> gold, int h, int w, int radius) {
        int t = binaries.size();
        int n = gold.size();
        double[] precision = new double[t];
        double[] recall = new double[t];
        double[][] perImage = new double[n][t];
        double ods = 0;
        for (int k = 0; k < t; k++) {
            long mp = 0;
            long np = 0;
            long mg = 0;
            long ng = 0;
            for (int i = 0; i < n; i++) {
                int[] c = match(binaries.get(k).get(i), gold.get(i), h, w, radius);
                mp += c[0];
                np += c[1];
                mg += c[2];
                ng += c[3];
                perImage[i][k] = f(c[1] == 0 ? 0 : c[0] / (double) c[1], c[3] == 0 ? 0 : c[2] / (double) c[3]);
            }
            precision[k] = np == 0 ? 1 : mp / (double) np;
            recall[k] = ng == 0 ? 0 : mg / (double) ng;
            ods = Math.max(ods, f(precision[k], recall[k]));
        }
        double ois = 0;
        for (double[] row : perImage) {
            double b = 0;
            for (double v : row) {
                b = Math.max(b, v);
            }
            ois += b / n;
        }
        Integer[] idx = new Integer[t];
        for (int k = 0; k < t; k++) {
            idx[k] = k;
        }
        java.util.Arrays.sort(idx, (a, b) -> Double.compare(recall[a], recall[b]));
        double ap = 0;
        double prevR = 0;
        double prevP = 1;
        for (int k : idx) {
            ap += (recall[k] - prevR) * (precision[k] + prevP) / 2;
            prevR = recall[k];
            prevP = precision[k];
        }
        return new Scores(ods, ois, ap);
    }

    /**
     * Evaluates score maps after thinning, at {@code thresholds} evenly spaced quantile levels
     * relative to each map's maximum.
     *
     * @param scores per image its score map
     * @param gold per image its boundary
     * @param h rows
     * @param w columns
     * @param thresholds how many thresholds
     * @param radius the matching tolerance
     * @return the scores
     */
    public static Scores evaluateScores(List<float[]> scores, List<boolean[]> gold, int h, int w, int thresholds,
            int radius) {
        List<float[]> thin = new ArrayList<>();
        float max = 1e-12f;
        for (float[] s : scores) {
            float[] t = thin(s, h, w);
            thin.add(t);
            for (float v : t) {
                max = Math.max(max, v);
            }
        }
        List<List<boolean[]>> bins = new ArrayList<>();
        for (int k = 1; k <= thresholds; k++) {
            float thr = max * k / (thresholds + 1);
            List<boolean[]> b = new ArrayList<>();
            for (float[] t : thin) {
                boolean[] e = new boolean[t.length];
                for (int i = 0; i < t.length; i++) {
                    e[i] = t[i] > thr;
                }
                b.add(e);
            }
            bins.add(b);
        }
        return evaluate(bins, gold, h, w, radius);
    }

    // ---------------------------------------------------------------- HED

    /** Holistically-nested edge detection. */
    public static final class Hed {
        private final Var[] conv;
        private final Var[] bias;
        private final Var[] side;
        private final Var fuse;
        private final boolean deep;
        private final List<Var> params = new ArrayList<>();

        /**
         * Creates a network.
         *
         * @param channels the widths of the three stages
         * @param deepSupervision deep supervision and balanced loss, or a single-output FCN
         * @param seed the seed
         */
        public Hed(int[] channels, boolean deepSupervision, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            this.deep = deepSupervision;
            conv = new Var[6];
            bias = new Var[6];
            int in = 1;
            for (int s = 0; s < 3; s++) {
                for (int k = 0; k < 2; k++) {
                    conv[2 * s + k] = VisionOps.convWeight(r, channels[s], in, 3);
                    bias[2 * s + k] = VisionOps.zeros(channels[s]);
                    in = channels[s];
                }
            }
            side = new Var[3];
            for (int s = 0; s < 3; s++) {
                side[s] = VisionOps.convWeight(r, 1, channels[s], 1);
            }
            fuse = Var.param(Tensor.full(Shape.of(1, 3, 1, 1), 1f / 3, Tensor.zeros(Shape.of(1)).device()));
            params.addAll(List.of(conv));
            params.addAll(List.of(bias));
            params.addAll(List.of(side));
            params.add(fuse);
        }

        /** @return {@code {side1, side2, side3, fused}} logits {@code [b, 1, h, w]} */
        private Var[] forward(Var x) {
            int h = x.shape().dim(2);
            int w = x.shape().dim(3);
            Var[] out = new Var[4];
            Var a = x;
            for (int s = 0; s < 3; s++) {
                if (s > 0) {
                    a = VisionOps.maxPool2(a);
                }
                a = VisionOps.conv(a, conv[2 * s], bias[2 * s]).relu();
                a = VisionOps.conv(a, conv[2 * s + 1], bias[2 * s + 1]).relu();
                Var o = VisionOps.conv(a, side[s], null);
                out[s] = s == 0 ? o : Functional.resize(o, new int[] {h, w}, Functional.Interpolation.LINEAR, false);
            }
            out[3] = VisionOps.conv(Var.concat(1, out[0], out[1], out[2]), fuse, null);
            return out;
        }

        private static Var bce(Var logits, Tensor y, boolean balanced) {
            float[] t = y.toArray();
            float pos = 0;
            for (float v : t) {
                pos += v;
            }
            float beta = balanced ? 1 - pos / t.length : 0.5f;
            float[] wt = new float[t.length];
            for (int i = 0; i < t.length; i++) {
                wt[i] = 2 * (t[i] > 0.5f ? beta : 1 - beta);
            }
            Var p = logits.sigmoid();
            Var l = p.add(1e-6f).log().mul(y).add(p.mul(-1f).add(1f + 1e-6f).log().mul(Var.of(y).mul(-1f).add(1f)));
            return l.mul(Tensor.of(y.shape(), wt)).mean().mul(-1f);
        }

        /**
         * Trains.
         *
         * @param images {@code [n][h * w]}
         * @param edges {@code [n][h * w]} boundary labels
         * @param h rows
         * @param w columns
         * @param steps steps
         * @param batch batch size
         * @param lr the learning rate
         * @param seed the seed
         */
        public void train(List<float[]> images, List<boolean[]> edges, int h, int w, int steps, int batch, float lr,
                long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            Optimizer opt = Optimizer.adam(lr);
            for (int s = 0; s < steps; s++) {
                float[] x = new float[batch * h * w];
                float[] y = new float[batch * h * w];
                for (int b = 0; b < batch; b++) {
                    int i = r.nextInt(images.size());
                    System.arraycopy(images.get(i), 0, x, b * h * w, h * w);
                    for (int p = 0; p < h * w; p++) {
                        y[b * h * w + p] = edges.get(i)[p] ? 1 : 0;
                    }
                }
                Var[] o = forward(Var.of(Tensor.of(Shape.of(batch, 1, h, w), x)));
                Tensor yt = Tensor.of(Shape.of(batch, 1, h, w), y);
                Var loss = bce(o[3], yt, deep);
                if (deep) {
                    for (int k = 0; k < 3; k++) {
                        loss = loss.add(bce(o[k], yt, true));
                    }
                }
                params.forEach(Var::zeroGrad);
                loss.backward();
                opt.step(params);
            }
            params.forEach(Var::zeroGrad);
        }

        /**
         * @param image {@code h * w}
         * @param h rows
         * @param w columns
         * @return the fused edge probability
         */
        public float[] predict(float[] image, int h, int w) {
            return Var.noGrad(() -> forward(Var.of(Tensor.of(Shape.of(1, 1, h, w), image)))[3].sigmoid())
                    .value().toArray();
        }
    }
}
