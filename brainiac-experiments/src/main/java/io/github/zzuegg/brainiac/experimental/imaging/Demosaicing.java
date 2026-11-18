package io.github.zzuegg.brainiac.experimental.imaging;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.InverseProblems;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Demosaicing of RGGB Bayer mosaics ({@link InverseProblems#bayer}).
 *
 * <ul>
 *   <li>{@link Method#NEAREST}: each missing value copied from the nearest sample of its colour
 *       (the crudest baseline).</li>
 *   <li>{@link Method#BILINEAR}: per-channel normalised convolution with the bilinear kernel;
 *       zippers and false colour at edges.</li>
 *   <li>{@link Method#MALVAR} (Malvar, He &amp; Cutler, ICASSP 2004): bilinear interpolation
 *       corrected by the Laplacian of the channel actually sampled at the pixel, as fixed 5x5
 *       filters.</li>
 *   <li>{@link Method#RESIDUAL_INTERPOLATION} (Kiku et al., ICIP 2013): green by Malvar, then red
 *       and blue as a guided-filter estimate from green ({@link EdgePreservingFilters#guidedMasked})
 *       plus the bilinearly interpolated residual between the samples and that estimate.</li>
 *   <li>{@link Method#LEARNED} (after Gharbi et al., SIGGRAPH Asia 2016): a small residual CNN
 *       ({@link Learned}) on the bilinear estimate and the mosaic, trained with hard-patch mining
 *       (half of every batch drawn from the hardest quarter of the training images by their last
 *       loss), which concentrates on moiré-prone stripes.</li>
 * </ul>
 * Images are {@code [3][side * side]} with an even side.
 */
public final class Demosaicing {

    /** The methods. */
    public enum Method {
        /** Nearest-sample fill. */
        NEAREST,
        /** Bilinear interpolation. */
        BILINEAR,
        /** Malvar-He-Cutler gradient-corrected interpolation. */
        MALVAR,
        /** Residual interpolation with a guided filter. */
        RESIDUAL_INTERPOLATION,
        /** A learned CNN (use {@link Learned}). */
        LEARNED
    }

    private static final double[] G_AT_RB = {0, 0, -1, 0, 0, 0, 0, 2, 0, 0, -1, 2, 4, 2, -1, 0, 0, 2, 0, 0, 0, 0, -1,
        0, 0};
    private static final double[] RB_AT_G_ROW = {0, 0, 0.5, 0, 0, 0, -1, 0, -1, 0, -1, 4, 5, 4, -1, 0, -1, 0, -1, 0,
        0, 0, 0.5, 0, 0};
    private static final double[] RB_AT_G_COL = {0, 0, -1, 0, 0, 0, -1, 4, -1, 0, 0.5, 0, 5, 0, 0.5, 0, -1, 4, -1, 0,
        0, 0, -1, 0, 0};
    private static final double[] RB_AT_BR = {0, 0, -1.5, 0, 0, 0, 2, 0, 2, 0, -1.5, 0, 6, 0, -1.5, 0, 2, 0, 2, 0, 0,
        0, -1.5, 0, 0};

    private Demosaicing() {
    }

    /**
     * Demosaics with a classical method.
     *
     * @param method any method but {@link Method#LEARNED}
     * @param mosaic the mosaic
     * @param side the side
     * @return {@code [3][side * side]}
     */
    public static double[][] demosaic(Method method, double[] mosaic, int side) {
        return switch (method) {
            case NEAREST -> nearest(mosaic, side);
            case BILINEAR -> bilinear(mosaic, side);
            case MALVAR -> malvar(mosaic, side);
            case RESIDUAL_INTERPOLATION -> residual(mosaic, side);
            case LEARNED -> throw new IllegalArgumentException("use Demosaicing.Learned");
        };
    }

    private static int clamp(int v, int side) {
        return Math.max(0, Math.min(side - 1, v));
    }

    private static double[] maskedInterpolation(double[] values, boolean[] mask, int side) {
        double[] k = {1, 2, 1, 2, 4, 2, 1, 2, 1};
        double[] out = new double[side * side];
        for (int r = 0; r < side; r++) {
            for (int c = 0; c < side; c++) {
                int i = r * side + c;
                if (mask[i]) {
                    out[i] = values[i];
                    continue;
                }
                double s = 0;
                double w = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int rr = r + dy;
                        int cc = c + dx;
                        if (rr < 0 || cc < 0 || rr >= side || cc >= side) {
                            continue;
                        }
                        int j = rr * side + cc;
                        if (mask[j]) {
                            double kv = k[(dy + 1) * 3 + dx + 1];
                            s += kv * values[j];
                            w += kv;
                        }
                    }
                }
                out[i] = w > 0 ? s / w : 0;
            }
        }
        return out;
    }

    private static boolean[] channelMask(int ch, int side) {
        boolean[] m = new boolean[side * side];
        for (int r = 0; r < side; r++) {
            for (int c = 0; c < side; c++) {
                m[r * side + c] = InverseProblems.bayerChannel(r, c) == ch;
            }
        }
        return m;
    }

    private static double[][] nearest(double[] mosaic, int side) {
        double[][] out = new double[3][side * side];
        for (int r = 0; r < side; r++) {
            for (int c = 0; c < side; c++) {
                int r0 = r & ~1;
                int c0 = c & ~1;
                out[0][r * side + c] = mosaic[r0 * side + c0];
                out[1][r * side + c] = InverseProblems.bayerChannel(r, c) == 1 ? mosaic[r * side + c]
                        : mosaic[r0 * side + c0 + 1];
                out[2][r * side + c] = mosaic[(r0 + 1) * side + c0 + 1];
            }
        }
        return out;
    }

    private static double[][] bilinear(double[] mosaic, int side) {
        double[][] out = new double[3][];
        for (int ch = 0; ch < 3; ch++) {
            out[ch] = maskedInterpolation(mosaic, channelMask(ch, side), side);
        }
        return out;
    }

    private static double filter(double[] mosaic, int side, int r, int c, double[] k) {
        double s = 0;
        for (int dy = -2; dy <= 2; dy++) {
            for (int dx = -2; dx <= 2; dx++) {
                double kv = k[(dy + 2) * 5 + dx + 2];
                if (kv != 0) {
                    int rr = reflect(r + dy, side);
                    int cc = reflect(c + dx, side);
                    s += kv * mosaic[rr * side + cc];
                }
            }
        }
        return s / 8;
    }

    private static int reflect(int i, int side) {
        // whole-sample reflection keeps the Bayer phase on an even-sided image
        if (i < 0) {
            return clamp(-i, side);
        }
        if (i >= side) {
            return clamp(2 * (side - 1) - i, side);
        }
        return i;
    }

    private static double[][] malvar(double[] mosaic, int side) {
        double[][] out = new double[3][side * side];
        for (int r = 0; r < side; r++) {
            for (int c = 0; c < side; c++) {
                int i = r * side + c;
                int ch = InverseProblems.bayerChannel(r, c);
                double v = mosaic[i];
                if (ch == 0) {
                    out[0][i] = v;
                    out[1][i] = filter(mosaic, side, r, c, G_AT_RB);
                    out[2][i] = filter(mosaic, side, r, c, RB_AT_BR);
                } else if (ch == 2) {
                    out[2][i] = v;
                    out[1][i] = filter(mosaic, side, r, c, G_AT_RB);
                    out[0][i] = filter(mosaic, side, r, c, RB_AT_BR);
                } else if (r % 2 == 0) {
                    // green in a red row: red neighbours left and right, blue above and below
                    out[1][i] = v;
                    out[0][i] = filter(mosaic, side, r, c, RB_AT_G_ROW);
                    out[2][i] = filter(mosaic, side, r, c, RB_AT_G_COL);
                } else {
                    out[1][i] = v;
                    out[2][i] = filter(mosaic, side, r, c, RB_AT_G_ROW);
                    out[0][i] = filter(mosaic, side, r, c, RB_AT_G_COL);
                }
            }
        }
        return out;
    }

    private static double[][] residual(double[] mosaic, int side) {
        double[][] out = new double[3][];
        double[] g = malvar(mosaic, side)[1];
        out[1] = g;
        for (int ch : new int[] {0, 2}) {
            boolean[] m = channelMask(ch, side);
            double[] tentative = EdgePreservingFilters.guidedMasked(g, mosaic, m, side, 2, 1e-4);
            double[] res = new double[side * side];
            for (int i = 0; i < res.length; i++) {
                res[i] = m[i] ? mosaic[i] - tentative[i] : 0;
            }
            double[] ri = maskedInterpolation(res, m, side);
            out[ch] = new double[side * side];
            for (int i = 0; i < res.length; i++) {
                out[ch][i] = tentative[i] + ri[i];
            }
        }
        return out;
    }

    /**
     * The colour PSNR over all three channels, ignoring a border.
     *
     * @param estimate the estimate
     * @param truth the truth
     * @param side the side
     * @param border pixels ignored at each edge
     * @return CPSNR in dB
     */
    public static double cpsnr(double[][] estimate, double[][] truth, int side, int border) {
        double s = 0;
        int n = 0;
        for (int ch = 0; ch < 3; ch++) {
            for (int r = border; r < side - border; r++) {
                for (int c = border; c < side - border; c++) {
                    int i = r * side + c;
                    double d = Math.max(0, Math.min(1, estimate[ch][i])) - truth[ch][i];
                    s += d * d;
                    n++;
                }
            }
        }
        return 10 * Math.log10(n / Math.max(s, 1e-12));
    }

    /**
     * Colour error on edge pixels: the mean Euclidean RGB error where the true green gradient
     * exceeds a threshold (zipper and false-colour artefacts concentrate there).
     *
     * @param estimate the estimate
     * @param truth the truth
     * @param side the side
     * @return the mean error on edge pixels
     */
    public static double edgeColourError(double[][] estimate, double[][] truth, int side) {
        double s = 0;
        int n = 0;
        for (int r = 2; r < side - 2; r++) {
            for (int c = 2; c < side - 2; c++) {
                int i = r * side + c;
                double gx = truth[1][i + 1] - truth[1][i - 1];
                double gy = truth[1][i + side] - truth[1][i - side];
                if (Math.hypot(gx, gy) > 0.1) {
                    double e = 0;
                    for (int ch = 0; ch < 3; ch++) {
                        double d = Math.max(0, Math.min(1, estimate[ch][i])) - truth[ch][i];
                        e += d * d;
                    }
                    s += Math.sqrt(e);
                    n++;
                }
            }
        }
        return n == 0 ? 0 : s / n;
    }

    /** The learned demosaicker: a residual CNN on the bilinear estimate and the mosaic. */
    public static final class Learned {
        private final ImagingCnn net;

        /**
         * Creates an untrained network.
         *
         * @param hidden hidden channels
         * @param depth convolutions
         * @param seed the seed
         */
        public Learned(int hidden, int depth, long seed) {
            this.net = new ImagingCnn(4, hidden, 3, depth, true, new SplittableRandom(seed));
        }

        /** @return the trainable parameters */
        public List<Var> parameters() {
            return net.parameters();
        }

        private static Tensor inputs(List<double[]> mosaics, int side) {
            int b = mosaics.size();
            int n = side * side;
            float[] v = new float[b * 4 * n];
            for (int i = 0; i < b; i++) {
                double[][] bl = bilinear(mosaics.get(i), side);
                for (int ch = 0; ch < 3; ch++) {
                    for (int p = 0; p < n; p++) {
                        v[(i * 4 + ch) * n + p] = (float) bl[ch][p];
                    }
                }
                for (int p = 0; p < n; p++) {
                    v[(i * 4 + 3) * n + p] = (float) mosaics.get(i)[p];
                }
            }
            return Tensor.of(Shape.of(b, 4, side, side), v);
        }

        private static Tensor targets(List<double[][]> images, int side) {
            int b = images.size();
            int n = side * side;
            float[] v = new float[b * 3 * n];
            for (int i = 0; i < b; i++) {
                for (int ch = 0; ch < 3; ch++) {
                    for (int p = 0; p < n; p++) {
                        v[(i * 3 + ch) * n + p] = (float) images.get(i)[ch][p];
                    }
                }
            }
            return Tensor.of(Shape.of(b, 3, side, side), v);
        }

        /**
         * Trains on mosaic and image pairs.
         *
         * @param mosaics the mosaics
         * @param images the true images
         * @param side the side
         * @param steps gradient steps
         * @param batch images per step
         * @param hardMining whether to draw half of each batch from the hardest quarter
         * @param learningRate Adam's rate
         * @param seed the seed
         */
        public void fit(List<double[]> mosaics, List<double[][]> images, int side, int steps, int batch,
                boolean hardMining, float learningRate, long seed) {
            Tensor x = inputs(mosaics, side);
            Tensor y = targets(images, side);
            int n = mosaics.size();
            double[] lastLoss = new double[n];
            java.util.Arrays.fill(lastLoss, Double.MAX_VALUE);
            SplittableRandom r = new SplittableRandom(seed);
            Optimizer opt = Optimizer.adam(learningRate);
            for (int s = 0; s < steps; s++) {
                int[] idx = new int[batch];
                Integer[] order = null;
                if (hardMining && s >= n / batch) {
                    order = new Integer[n];
                    for (int i = 0; i < n; i++) {
                        order[i] = i;
                    }
                    java.util.Arrays.sort(order, (a, b) -> Double.compare(lastLoss[b], lastLoss[a]));
                }
                for (int i = 0; i < batch; i++) {
                    if (order != null && i < batch / 2) {
                        idx[i] = order[r.nextInt(Math.max(1, n / 4))];
                    } else {
                        idx[i] = r.nextInt(n);
                    }
                }
                Tensor[] xs = new Tensor[batch];
                Tensor[] ys = new Tensor[batch];
                for (int i = 0; i < batch; i++) {
                    xs[i] = x.slice(0, idx[i], idx[i] + 1);
                    ys[i] = y.slice(0, idx[i], idx[i] + 1);
                }
                Var err = net.forward(Var.of(Tensor.concat(0, xs))).sub(Var.of(Tensor.concat(0, ys))).square();
                float[] per = Var.noGrad(() -> err.mean(3, false).mean(2, false).mean(1, false)).value().toArray();
                for (int i = 0; i < batch; i++) {
                    lastLoss[idx[i]] = per[i];
                }
                err.mean().backward();
                opt.step(net.parameters());
            }
        }

        /**
         * Demosaics one mosaic.
         *
         * @param mosaic the mosaic
         * @param side the side
         * @return {@code [3][side * side]}
         */
        public double[][] demosaic(double[] mosaic, int side) {
            float[] o = Var.noGrad(() -> net.forward(Var.of(inputs(List.of(mosaic), side)))).value().toArray();
            int n = side * side;
            double[][] out = new double[3][n];
            for (int ch = 0; ch < 3; ch++) {
                for (int p = 0; p < n; p++) {
                    out[ch][p] = o[ch * n + p];
                }
            }
            return out;
        }
    }
}
