package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * One-bit neural networks, from binary weights to binary weights and activations:
 *
 * <ul>
 *   <li>{@link Mode#BINARY_CONNECT} (Courbariaux et al., NeurIPS 2015,
 *       <a href="https://arxiv.org/abs/1511.00363">arXiv:1511.00363</a>): real-valued latent
 *       weights, their signs in the forward pass, the gradient passed straight through to the
 *       latent weights (cancelled where {@code |w| > 1}); activations stay real;</li>
 *   <li>{@link Mode#BNN} (Hubara/Courbariaux et al., NeurIPS 2016,
 *       <a href="https://arxiv.org/abs/1602.02830">arXiv:1602.02830</a>): activations binarised
 *       too, with the clipped ("hard tanh") straight-through estimator, so a layer is XNOR and
 *       popcount;</li>
 *   <li>{@link Mode#XNOR} (Rastegari et al., ECCV 2016,
 *       <a href="https://arxiv.org/abs/1603.05279">arXiv:1603.05279</a>): binary weights scaled per
 *       output unit by their mean absolute value {@code α};</li>
 *   <li>{@link Mode#BI_REAL} (Liu et al., ECCV 2018,
 *       <a href="https://arxiv.org/abs/1808.00278">arXiv:1808.00278</a>): XNOR layers with a
 *       real-valued shortcut around each square binary layer and the piecewise-polynomial
 *       ApproxSign derivative {@code 2 - 2|x|} on {@code |x| < 1}.</li>
 * </ul>
 *
 * {@link BinaryDense} is the drop-in {@link Layer}; {@link PackedXnor} is the inference kernel
 * with weights and inputs packed 64 to a {@code long}: {@code x·w = n - 2 popcount(x XOR w)}.
 * After each optimizer step, {@link #clipLatent} keeps the latent weights in {@code [-1, 1]}.
 */
public final class BinaryLayers {

    /** The binarisation scheme. */
    public enum Mode {
        /** binary weights, real activations */
        BINARY_CONNECT,
        /** binary weights and activations */
        BNN,
        /** binary weights with per-unit scale, binary activations */
        XNOR,
        /** XNOR with real shortcuts and ApproxSign */
        BI_REAL
    }

    private BinaryLayers() {
    }

    /**
     * Signs with {@code sign(0) = +1}.
     *
     * @param x the values
     * @return {@code ±1}
     */
    public static Tensor signs(Tensor x) {
        float[] v = x.toArray();
        for (int i = 0; i < v.length; i++) {
            v[i] = v[i] >= 0 ? 1f : -1f;
        }
        return Tensor.of(x.shape(), v);
    }

    /**
     * The sign with the clipped straight-through estimator: gradient 1 on {@code |x| <= 1}.
     *
     * @param x the input
     * @return {@code ±1} with the hard-tanh gradient
     */
    public static Var sign(Var x) {
        return Functional.straightThrough(Var.of(signs(x.value())), x.clamp(-1f, 1f));
    }

    /**
     * The sign with Bi-Real's ApproxSign gradient {@code 2 - 2|x|} on {@code |x| < 1}.
     *
     * @param x the input
     * @return {@code ±1} with the polynomial gradient
     */
    public static Var approxSign(Var x) {
        Var c = x.clamp(-1f, 1f);
        Var poly = c.mul(2f).sub(c.mul(c.abs()));
        return Functional.straightThrough(Var.of(signs(x.value())), poly);
    }

    /**
     * The binary weights a mode uses.
     *
     * @param w latent weights {@code [in, out]}
     * @param mode the mode
     * @return {@code sign(w)}, scaled by the per-column mean {@code |w|} for XNOR and Bi-Real
     */
    public static Var weights(Var w, Mode mode) {
        Var b = Functional.straightThrough(Var.of(signs(w.value())), w.clamp(-1f, 1f));
        if (mode == Mode.XNOR || mode == Mode.BI_REAL) {
            return b.mul(w.abs().mean(0, true));
        }
        return b;
    }

    /**
     * The activation a mode uses.
     *
     * @param x pre-activations
     * @param mode the mode
     * @return ReLU for BinaryConnect, ApproxSign for Bi-Real, the clipped-STE sign otherwise
     */
    public static Var activation(Var x, Mode mode) {
        return switch (mode) {
            case BINARY_CONNECT -> x.relu();
            case BI_REAL -> approxSign(x);
            default -> sign(x);
        };
    }

    /**
     * Clamps latent weights to {@code [-1, 1]}, as BinaryConnect does after every update.
     *
     * @param weights the latent weights
     */
    public static void clipLatent(List<Var> weights) {
        for (Var w : weights) {
            w.assign(w.value().maximum(Tensor.full(Shape.of(), -1f)).minimum(Tensor.full(Shape.of(), 1f)));
        }
    }

    /**
     * A binary linear layer: {@code activation(x) · weights(W) + b}, plus {@code x} for Bi-Real
     * when the layer is square. The input is binarised for every mode but BinaryConnect, so put a
     * real layer first (as the papers keep the first and last layers real).
     */
    public static final class BinaryDense implements Layer {

        private final Mode mode;
        private final int in;
        private final int out;
        private final Var weight;
        private final Var bias;

        /**
         * Creates the layer with latent weights uniform in {@code ±sqrt(3 / in)}.
         *
         * @param in the input size
         * @param out the output size
         * @param mode the mode
         * @param random the initializer
         */
        public BinaryDense(int in, int out, Mode mode, SplittableRandom random) {
            this.mode = mode;
            this.in = in;
            this.out = out;
            float[] w = new float[in * out];
            double bound = Math.sqrt(3.0 / in);
            for (int i = 0; i < w.length; i++) {
                w[i] = (float) ((2 * random.nextDouble() - 1) * bound);
            }
            this.weight = Var.param(Tensor.of(Shape.of(in, out), w));
            this.bias = Var.param(Tensor.zeros(Shape.of(out)));
        }

        /**
         * @param units the output size
         * @param mode the mode
         * @return a spec for model builders
         */
        public static LayerSpec spec(int units, Mode mode) {
            return new LayerSpec() {
                @Override
                public Shape outputShape(Shape input) {
                    int[] d = input.toArray();
                    d[d.length - 1] = units;
                    return Shape.of(d);
                }

                @Override
                public Layer create(Shape input, SplittableRandom random) {
                    return new BinaryDense(input.dim(-1), units, mode, random);
                }
            };
        }

        @Override
        public Var forward(Var input) {
            Var a = mode == Mode.BINARY_CONNECT ? input : activation(input, mode);
            Var z = a.matmul(weights(weight, mode)).add(bias);
            return mode == Mode.BI_REAL && in == out ? z.add(input) : z;
        }

        @Override
        public List<Var> parameters() {
            return List.of(weight, bias);
        }

        @Override
        public int outputSize(int inputSize) {
            return out;
        }

        /** @return the latent weights {@code [in, out]} */
        public Var weight() {
            return weight;
        }

        /** @return the bias */
        public Var bias() {
            return bias;
        }

        /** @return the mode */
        public Mode mode() {
            return mode;
        }

        @Override
        public String toString() {
            return "BinaryDense(" + in + "->" + out + ", " + mode + ")";
        }
    }

    /**
     * Bit-packed XNOR/popcount inference of a binary layer: inputs and weights are signs packed
     * 64 to a {@code long}, and {@code Σ_i x_i w_i = n - 2 popcount(x XOR w)}.
     */
    public static final class PackedXnor {

        private final int in;
        private final int out;
        private final long[][] rows;
        private final float[] scale;
        private final float[] bias;

        private PackedXnor(int in, int out, long[][] rows, float[] scale, float[] bias) {
            this.in = in;
            this.out = out;
            this.rows = rows;
            this.scale = scale;
            this.bias = bias;
        }

        /**
         * Packs a layer's weight signs.
         *
         * @param w latent weights {@code [in, out]}
         * @param b the bias {@code [out]}
         * @param scaled whether to keep XNOR-Net's per-unit scale {@code α}
         * @return the packed layer
         */
        public static PackedXnor pack(Tensor w, Tensor b, boolean scaled) {
            int in = w.shape().dim(0);
            int out = w.shape().dim(1);
            float[] v = w.toArray();
            int words = (in + 63) / 64;
            long[][] rows = new long[out][words];
            float[] scale = new float[out];
            for (int o = 0; o < out; o++) {
                double s = 0;
                for (int i = 0; i < in; i++) {
                    float x = v[i * out + o];
                    s += Math.abs(x);
                    if (x >= 0) {
                        rows[o][i >> 6] |= 1L << (i & 63);
                    }
                }
                scale[o] = scaled ? (float) (s / in) : 1f;
            }
            return new PackedXnor(in, out, rows, scale, b.toArray());
        }

        /**
         * Packs the signs of an input vector.
         *
         * @param x the input
         * @return the packed signs
         */
        public static long[] packSigns(float[] x) {
            long[] p = new long[(x.length + 63) / 64];
            for (int i = 0; i < x.length; i++) {
                if (x[i] >= 0) {
                    p[i >> 6] |= 1L << (i & 63);
                }
            }
            return p;
        }

        /**
         * The layer's output for packed input signs.
         *
         * @param x packed input signs
         * @return {@code α (n - 2 popcount(x XOR w)) + b}
         */
        public float[] apply(long[] x) {
            float[] y = new float[out];
            for (int o = 0; o < out; o++) {
                long[] r = rows[o];
                int diff = 0;
                for (int j = 0; j < r.length; j++) {
                    diff += Long.bitCount(x[j] ^ r[j]);
                }
                // padding bits are 0 in both, so they never differ
                y[o] = scale[o] * (in - 2 * diff) + bias[o];
            }
            return y;
        }

        /** @return the stored bits of the weights (1 per weight plus 32 per scale and bias) */
        public long bits() {
            return (long) in * out + 64L * out;
        }
    }

    /**
     * Plain float matrix-vector product, the speed baseline of {@link PackedXnor}.
     *
     * @param x the input {@code [in]}
     * @param w the weights {@code [in, out]} row-major
     * @param out the output size
     * @return {@code x · w}
     */
    public static float[] floatMatVec(float[] x, float[] w, int out) {
        float[] y = new float[out];
        for (int i = 0; i < x.length; i++) {
            float xi = x[i];
            int base = i * out;
            for (int o = 0; o < out; o++) {
                y[o] += xi * w[base + o];
            }
        }
        return y;
    }
}
