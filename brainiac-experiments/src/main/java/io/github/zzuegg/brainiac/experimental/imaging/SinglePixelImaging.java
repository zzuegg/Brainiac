package io.github.zzuegg.brainiac.experimental.imaging;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.signal.SparseRecovery;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Single-pixel imaging (Duarte et al., IEEE SPM 2008; Higham et al., Sci. Rep. 2018): a scene of
 * {@code n = side^2} pixels is measured by {@code m} inner products {@code y_i = <p_i, x> + noise}
 * with patterns {@code p_i}, all scaled to unit norm so the detector noise is in image units.
 * <ul>
 *   <li>Hadamard patterns (the separable Walsh functions {@code h_u(row) h_v(col)}), measured and
 *       inverted with the fast Walsh-Hadamard transform, keeping the first {@code m} in
 *       {@link Ordering#SEQUENCY} (by {@code seq(u) + seq(v)}, the number of sign changes) or
 *       {@link Ordering#CAKE_CUTTING} order (Yu et al., Sensors 2019: by the number of connected
 *       pieces {@code (seq(u) + 1)(seq(v) + 1)}); missing coefficients are zero;</li>
 *   <li>{@link Ordering#DCT}: the {@code m} lowest-frequency orthonormal DCT-II coefficients, the
 *       fixed-transform reference;</li>
 *   <li>random {@code +-1} patterns with compressed-sensing recovery by wave 12's
 *       {@link SparseRecovery} FISTA on orthonormal Haar coefficients ({@link #compressedHaar}) or
 *       TV ({@link #compressedTv});</li>
 *   <li>{@link Learned}: binarised {@code +-1} patterns trained jointly with a decoder (a linear map
 *       and a residual CNN) through a straight-through sign.</li>
 * </ul>
 */
public final class SinglePixelImaging {

    /** The deterministic pattern orderings. */
    public enum Ordering {
        /** Hadamard by total sequency. */
        SEQUENCY,
        /** Hadamard by number of connected pieces. */
        CAKE_CUTTING,
        /** Low-frequency DCT (zig-zag by {@code u + v}). */
        DCT
    }

    private SinglePixelImaging() {
    }

    /**
     * The fast Walsh-Hadamard transform in natural order, in place, unnormalised.
     *
     * @param v the vector, of power-of-two length
     */
    public static void fwht(double[] v) {
        int n = v.length;
        if (Integer.bitCount(n) != 1) {
            throw new IllegalArgumentException("length must be a power of two, got " + n);
        }
        for (int h = 1; h < n; h <<= 1) {
            for (int i = 0; i < n; i += h << 1) {
                for (int j = i; j < i + h; j++) {
                    double a = v[j];
                    double b = v[j + h];
                    v[j] = a + b;
                    v[j + h] = a - b;
                }
            }
        }
    }

    /**
     * The orthonormal 2-D Walsh-Hadamard transform (its own inverse).
     *
     * @param x the image, {@code side x side}
     * @param side the side, a power of two
     * @return the coefficients, indexed {@code u * side + v}
     */
    public static double[] hadamard2d(double[] x, int side) {
        double[] c = x.clone();
        double[] row = new double[side];
        for (int r = 0; r < side; r++) {
            System.arraycopy(c, r * side, row, 0, side);
            fwht(row);
            System.arraycopy(row, 0, c, r * side, side);
        }
        for (int col = 0; col < side; col++) {
            for (int r = 0; r < side; r++) {
                row[r] = c[r * side + col];
            }
            fwht(row);
            for (int r = 0; r < side; r++) {
                c[r * side + col] = row[r] / side;
            }
        }
        return c;
    }

    /**
     * The sequency (number of sign changes) of each natural-order Walsh function.
     *
     * @param side the length
     * @return the sequencies
     */
    public static int[] sequency(int side) {
        int[] s = new int[side];
        double[] e = new double[side];
        for (int u = 0; u < side; u++) {
            Arrays.fill(e, 0);
            e[u] = 1;
            fwht(e);
            for (int i = 1; i < side; i++) {
                if (e[i] != e[i - 1]) {
                    s[u]++;
                }
            }
        }
        return s;
    }

    /**
     * The order in which coefficients are measured.
     *
     * @param ordering the ordering
     * @param side the image side
     * @return the 2-D coefficient indices {@code u * side + v}, first measured first
     */
    public static int[] order(Ordering ordering, int side) {
        int[] seq = ordering == Ordering.DCT ? identity(side) : sequency(side);
        Integer[] idx = new Integer[side * side];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Comparator<Integer> bySum = Comparator.comparingInt(i -> seq[i / side] + seq[i % side]);
        Comparator<Integer> c = switch (ordering) {
            case SEQUENCY, DCT -> bySum.thenComparingInt(i -> Math.max(seq[i / side], seq[i % side]));
            case CAKE_CUTTING -> Comparator.<Integer>comparingInt(i -> (seq[i / side] + 1) * (seq[i % side] + 1))
                    .thenComparing(bySum);
        };
        Arrays.sort(idx, c.thenComparingInt(i -> i));
        int[] out = new int[idx.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = idx[i];
        }
        return out;
    }

    private static int[] identity(int n) {
        int[] s = new int[n];
        for (int i = 0; i < n; i++) {
            s[i] = i;
        }
        return s;
    }

    /**
     * The orthonormal 2-D DCT-II, or its inverse.
     *
     * @param x the input
     * @param side the side
     * @param inverse whether to invert
     * @return the output
     */
    public static double[] dct2d(double[] x, int side, boolean inverse) {
        double[][] c = new double[side][side];
        for (int k = 0; k < side; k++) {
            double a = Math.sqrt((k == 0 ? 1.0 : 2.0) / side);
            for (int i = 0; i < side; i++) {
                c[k][i] = a * Math.cos(Math.PI * (i + 0.5) * k / side);
            }
        }
        double[] t = new double[x.length];
        double[] out = new double[x.length];
        for (int r = 0; r < side; r++) {
            for (int k = 0; k < side; k++) {
                double s = 0;
                for (int i = 0; i < side; i++) {
                    s += (inverse ? c[i][k] : c[k][i]) * x[r * side + i];
                }
                t[r * side + k] = s;
            }
        }
        for (int col = 0; col < side; col++) {
            for (int k = 0; k < side; k++) {
                double s = 0;
                for (int i = 0; i < side; i++) {
                    s += (inverse ? c[i][k] : c[k][i]) * t[i * side + col];
                }
                out[k * side + col] = s;
            }
        }
        return out;
    }

    /**
     * Measures and reconstructs with the first {@code m} deterministic patterns (Hadamard or DCT):
     * the transform's inverse with the unmeasured coefficients set to zero.
     *
     * @param ordering the ordering
     * @param x the scene
     * @param side its side
     * @param m the number of measurements
     * @param sigma the detector noise
     * @param r the noise source
     * @return the reconstruction
     */
    public static double[] truncated(Ordering ordering, double[] x, int side, int m, double sigma,
            SplittableRandom r) {
        boolean dct = ordering == Ordering.DCT;
        double[] c = dct ? dct2d(x, side, false) : hadamard2d(x, side);
        double[] kept = new double[c.length];
        int[] ord = order(ordering, side);
        for (int i = 0; i < m; i++) {
            kept[ord[i]] = c[ord[i]] + sigma * gaussian(r);
        }
        return dct ? dct2d(kept, side, true) : hadamard2d(kept, side);
    }

    /**
     * Random {@code +-1 / sqrt(n)} patterns.
     *
     * @param m the number of patterns
     * @param n the number of pixels
     * @param r the source
     * @return the patterns {@code [m][n]}
     */
    public static double[][] randomPatterns(int m, int n, SplittableRandom r) {
        double[][] a = new double[m][n];
        double s = 1 / Math.sqrt(n);
        for (double[] row : a) {
            for (int j = 0; j < n; j++) {
                row[j] = r.nextBoolean() ? s : -s;
            }
        }
        return a;
    }

    /**
     * Measures with explicit patterns.
     *
     * @param a the patterns {@code [m][n]}
     * @param x the scene
     * @param sigma the detector noise
     * @param r the noise source
     * @return the measurements
     */
    public static double[] measure(double[][] a, double[] x, double sigma, SplittableRandom r) {
        double[] y = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            y[i] = ImageMath.dot(a[i], x) + sigma * gaussian(r);
        }
        return y;
    }

    /**
     * Compressed-sensing recovery on orthonormal Haar coefficients by wave 12's
     * {@link SparseRecovery} (FISTA or AMP) on the effective matrix {@code A W^T}.
     *
     * @param method the sparse solver
     * @param a the patterns
     * @param y the measurements
     * @param side the image side
     * @param lambda the lasso penalty (FISTA) or threshold multiplier (AMP)
     * @param iterations the iterations
     * @return the image
     */
    public static double[] compressedHaar(SparseRecovery.Method method, double[][] a, double[] y, int side,
            double lambda, int iterations) {
        int levels = Integer.numberOfTrailingZeros(side) - 2;
        double[][] aw = new double[a.length][];
        for (int i = 0; i < a.length; i++) {
            aw[i] = Wavelets.forward2d(a[i], side, Wavelets.Wavelet.HAAR, levels);
        }
        double[] c = SparseRecovery.solve(method, aw, y, 0, lambda, iterations, null, null);
        return Wavelets.inverse2d(c, side, Wavelets.Wavelet.HAAR, levels);
    }

    /**
     * Compressed-sensing recovery with a TV prior by Chambolle-Pock ({@link PrimalDual#linear}).
     *
     * @param a the patterns
     * @param y the measurements
     * @param side the side
     * @param lambda the TV weight
     * @param iterations the iterations
     * @return the image
     */
    public static double[] compressedTv(double[][] a, double[] y, int side, double lambda, int iterations) {
        return PrimalDual.linear(LinearOperator.matrix(a), y, side, lambda, true, iterations, null);
    }

    private static double gaussian(SplittableRandom r) {
        double u = 1 - r.nextDouble();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r.nextDouble());
    }

    /**
     * Learned binarised patterns with a decoder (a deep compressed-sensing autoencoder): the
     * patterns are {@code sign(W) / sqrt(n)}, with a straight-through gradient (identity inside
     * {@code |W| <= 1}); the decoder is a linear map {@code m -> n} (initialised as the patterns'
     * transpose) followed by a residual CNN.
     */
    public static final class Learned {

        private final int m;
        private final int side;
        private final Var w;
        private final Var decoder;
        private final ImagingCnn cnn;

        /**
         * Creates the model.
         *
         * @param m the number of patterns
         * @param side the image side
         * @param hidden the CNN's channels
         * @param depth the CNN's convolutions
         * @param seed the seed
         */
        public Learned(int m, int side, int hidden, int depth, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            int n = side * side;
            this.m = m;
            this.side = side;
            this.w = Var.param(Tensor.normal(Shape.of(m, n), 0, 0.5f, r));
            this.decoder = Var.param(binary(w.value()));
            this.cnn = new ImagingCnn(1, hidden, 1, depth, true, r);
        }

        private Tensor binary(Tensor t) {
            float[] v = t.toArray();
            float s = (float) (1 / Math.sqrt((double) side * side));
            float[] o = new float[v.length];
            for (int i = 0; i < v.length; i++) {
                o[i] = v[i] >= 0 ? s : -s;
            }
            return Tensor.of(t.shape(), o);
        }

        private Var patterns() {
            return Var.custom(in -> binary(in[0]), (in, out, g) -> {
                float[] v = in[0].toArray();
                float[] gv = g.toArray();
                float s = (float) (1 / Math.sqrt((double) side * side));
                float[] o = new float[v.length];
                for (int i = 0; i < v.length; i++) {
                    o[i] = Math.abs(v[i]) <= 1 ? gv[i] * s : 0;
                }
                return new Tensor[] {Tensor.of(in[0].shape(), o)};
            }, w);
        }

        private Var decode(Var y) {
            int b = y.shape().dim(0);
            Var x0 = y.matmul(decoder).reshape(b, 1, side, side);
            return cnn.forward(x0);
        }

        /**
         * Trains patterns and decoder on clean images with noisy simulated measurements.
         *
         * @param images the training images
         * @param steps the Adam steps
         * @param batch the batch size
         * @param lr the learning rate
         * @param sigma the detector noise
         * @param seed the seed
         */
        public void fit(List<double[]> images, int steps, int batch, float lr, double sigma, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            int n = side * side;
            Optimizer opt = Optimizer.adam(lr);
            List<Var> params = parameters();
            for (int s = 0; s < steps; s++) {
                float[] xb = new float[batch * n];
                for (int i = 0; i < batch; i++) {
                    double[] img = images.get(r.nextInt(images.size()));
                    for (int j = 0; j < n; j++) {
                        xb[i * n + j] = (float) img[j];
                    }
                }
                Tensor x = Tensor.of(Shape.of(batch, n), xb);
                Var y = Var.of(x).matmul(patterns().transpose())
                        .add(Var.of(Tensor.normal(Shape.of(batch, m), 0, (float) sigma, r)));
                Var loss = decode(y).sub(Var.of(x.reshape(batch, 1, side, side))).square().mean();
                loss.backward();
                opt.step(params);
            }
        }

        /**
         * Measures with the learned (binary) patterns and decodes.
         *
         * @param x the scene
         * @param sigma the detector noise
         * @param r the noise source
         * @return the reconstruction
         */
        public double[] reconstruct(double[] x, double sigma, SplittableRandom r) {
            float[] p = binary(w.value()).toArray();
            int n = side * side;
            float[] y = new float[m];
            for (int i = 0; i < m; i++) {
                double s = 0;
                for (int j = 0; j < n; j++) {
                    s += p[i * n + j] * x[j];
                }
                y[i] = (float) (s + sigma * gaussian(r));
            }
            Var out = Var.noGrad(() -> decode(Var.of(Tensor.of(Shape.of(1, m), y))));
            return ImageMath.toDouble(out.value().toArray());
        }

        /** @return the decoder's trainable values (linear map and CNN), excluding the patterns */
        public long decoderParameters() {
            return decoder.value().size() + cnn.parameterCount();
        }

        private List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(w, decoder));
            p.addAll(cnn.parameters());
            return p;
        }
    }
}
