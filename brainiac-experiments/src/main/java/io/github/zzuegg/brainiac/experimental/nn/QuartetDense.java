package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Op;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A dense layer trained with low-precision matrix multiplications in both passes, after
 * Quartet (Castro et al. 2025, "Quartet: Native FP4 Training Can Be Optimal for Large
 * Language Models", <a href="https://arxiv.org/abs/2505.14669">arXiv:2505.14669</a>). All
 * three products of a training step ({@code y = x w}, {@code dx = dy w^T},
 * {@code dw = x^T dy}) run on MXFP4 operands ({@link MicroscalingFormat#MXFP4}: E2M1 values,
 * one power-of-two scale per 32 along the reduction axis):
 *
 * <ul>
 *   <li><b>Forward</b>: both operands are rotated by a block Hadamard transform along the
 *       reduction axis (with fixed random signs; {@code x H H^T w = x w}), which spreads
 *       outlier features over each block of 32 before rounding to nearest.</li>
 *   <li><b>Backward</b>: the operands of both gradient products are rotated by a
 *       <em>fresh</em> random-sign Hadamard transform along their reduction axis (the output
 *       features for {@code dx}, the batch for {@code dw}) and rounded
 *       <em>stochastically</em>, with scales chosen so nothing saturates. Stochastic
 *       rounding makes each gradient product unbiased, so errors average out over steps
 *       instead of piling up, and the rotation shrinks its variance.</li>
 * </ul>
 * Gradients pass through the forward quantiser unchanged (straight-through). The other
 * {@link Precision}s are the baselines: float32, naive FP4 (round to nearest, no rotation,
 * in both passes) and FP8 (E4M3 forward, E5M2 gradients, a scale per tensor). Values are
 * simulated in float32 ("fake quantisation"); a dimension that is not a multiple of 32 is
 * zero-padded inside the rotation.
 */
public final class QuartetDense implements Layer {

    /** How the layer's products are computed. */
    public enum Precision {
        /** float32, the reference. */
        FP32,
        /** FP8 with a scale per tensor: E4M3 operands forward, E5M2 gradients backward. */
        FP8,
        /** MXFP4 round-to-nearest in both passes, no rotation. */
        FP4_NAIVE,
        /** Quartet: Hadamard + round-to-nearest forward, random Hadamard + stochastic rounding backward. */
        QUARTET
    }

    private static final int BLOCK = 32;
    private static final float[] GRID = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6};

    private final int units;
    private final Precision precision;
    private final Activation activation;
    private final Var weight;
    private final Var bias;
    private final float[] forwardSigns;
    private final SplittableRandom random;

    /**
     * @param weight the initial weight {@code [inputs, units]}
     * @param activation the activation
     * @param precision how the products are computed
     * @param seed the seed of the rotations and stochastic rounding
     */
    public QuartetDense(Tensor weight, Activation activation, Precision precision, long seed) {
        int inputs = weight.shape().dim(0);
        this.units = weight.shape().dim(1);
        this.precision = precision;
        this.activation = activation.bind(units);
        this.weight = Var.param(weight);
        this.bias = Var.param(Tensor.zeros(Shape.of(units)));
        this.random = new SplittableRandom(seed);
        this.forwardSigns = signs(padded(inputs), random);
    }

    /**
     * Describes a fresh layer for {@code Model.sequential().add(...)}.
     *
     * @param units the output width
     * @param activation the activation
     * @param precision how the products are computed
     * @return the spec
     */
    public static LayerSpec spec(int units, Activation activation, Precision precision) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] d = input.toArray();
                d[d.length - 1] = units;
                return Shape.of(d);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                int in = input.dim(input.rank() - 1);
                Tensor w = Tensor.normal(Shape.of(in, units), 0, (float) Math.sqrt(2.0 / in), random);
                return new QuartetDense(w, activation, precision, random.nextLong());
            }
        };
    }

    /** @return the weight {@code [inputs, units]} */
    public Var weight() {
        return weight;
    }

    @Override
    public Var forward(Var input) {
        Var y = precision == Precision.FP32 ? input.matmul(weight)
                : Var.apply(new LowPrecisionMatmul(this), input, weight);
        return activation.forward(y.add(bias));
    }

    @Override
    public int outputSize(int inputSize) {
        return units;
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, bias);
    }

    @Override
    public String toString() {
        return "QuartetDense(" + weight.shape().dim(0) + " -> " + units + ", " + precision + ")";
    }

    private static int padded(int n) {
        return (n + BLOCK - 1) / BLOCK * BLOCK;
    }

    private static float[] signs(int n, SplittableRandom r) {
        float[] s = new float[n];
        for (int i = 0; i < n; i++) {
            s[i] = r.nextBoolean() ? 1 : -1;
        }
        return s;
    }

    /**
     * Rounds a magnitude (already divided by its block scale) to an E2M1 grid point,
     * stochastically: up with probability equal to the fraction of the gap.
     */
    private static float roundStochastic(float v, SplittableRandom r) {
        float a = Math.min(Math.abs(v), 6);
        int i = 0;
        while (i < GRID.length - 1 && GRID[i + 1] <= a) {
            i++;
        }
        if (i == GRID.length - 1) {
            return Math.copySign(6, v);
        }
        float lo = GRID[i];
        float hi = GRID[i + 1];
        float up = (a - lo) / (hi - lo);
        return Math.copySign(r.nextFloat() < up ? hi : lo, v);
    }

    /**
     * Quantises a {@code [rows, cols]} row-major matrix to MXFP4 in blocks of 32 along one
     * axis (padded with zeros), optionally after a block Hadamard rotation along it.
     *
     * @param a the values, {@code rows * cols}
     * @param rows the rows
     * @param cols the columns
     * @param alongRows whether the blocks run down the rows (axis 0) instead of along a row
     * @param signs the random signs of the rotation, or {@code null} for none
     * @param random the source for stochastic rounding, or {@code null} for round to nearest
     * @return the quantised (rotated) matrix with the padded axis, {@code [rows', cols']}
     */
    static float[] mxfp4(float[] a, int rows, int cols, boolean alongRows, float[] signs, SplittableRandom random) {
        int len = alongRows ? rows : cols;
        int lanes = alongRows ? cols : rows;
        int plen = padded(len);
        float[] out = new float[plen * lanes];
        float[] block = new float[BLOCK];
        for (int lane = 0; lane < lanes; lane++) {
            for (int b0 = 0; b0 < plen; b0 += BLOCK) {
                for (int k = 0; k < BLOCK; k++) {
                    int i = b0 + k;
                    float v = i < len ? (alongRows ? a[i * cols + lane] : a[lane * cols + i]) : 0;
                    block[k] = signs == null ? v : v * signs[i];
                }
                if (signs != null) {
                    RotatedQuant.walshHadamard(block);
                }
                float amax = 0;
                for (float v : block) {
                    amax = Math.max(amax, Math.abs(v));
                }
                float scale = amax == 0 ? 0 : random == null
                        ? (float) Math.scalb(1.0, Math.getExponent(amax) - 2)   // the MX scale, may saturate
                        : (float) Math.scalb(1.0, Math.getExponent(amax / 6) + 1);   // never saturates
                for (int k = 0; k < BLOCK; k++) {
                    float q = scale == 0 ? 0 : (random == null ? MicroscalingFormat.roundE2m1(block[k] / scale)
                            : roundStochastic(block[k] / scale, random)) * scale;
                    int i = b0 + k;
                    if (alongRows) {
                        out[i * lanes + lane] = q;
                    } else {
                        out[lane * plen + i] = q;
                    }
                }
            }
        }
        return out;
    }

    private static float[] fp8(float[] a, Fp8Format format) {
        float amax = 0;
        for (float v : a) {
            amax = Math.max(amax, Math.abs(v));
        }
        float s = amax == 0 ? 1 : amax / format.max();
        float[] out = new float[a.length];
        for (int i = 0; i < a.length; i++) {
            out[i] = format.round(a[i] / s) * s;
        }
        return out;
    }

    /** The low-precision {@code x w} with its two low-precision gradient products. */
    private static final class LowPrecisionMatmul implements Op {

        private final QuartetDense layer;

        LowPrecisionMatmul(QuartetDense layer) {
            this.layer = layer;
        }

        /** {@code a [n, k] x b [k, m]} with the reduction axis {@code k} quantised in both. */
        private Tensor product(Tensor a, Tensor b, boolean backward) {
            int n = a.shape().dim(0);
            int k = a.shape().dim(1);
            int m = b.shape().dim(1);
            float[] av = a.toArray();
            float[] bv = b.toArray();
            if (layer.precision == Precision.FP8) {
                Fp8Format f = backward ? Fp8Format.E5M2 : Fp8Format.E4M3;
                return Tensor.of(Shape.of(n, k), fp8(av, f)).matmul(Tensor.of(Shape.of(k, m), fp8(bv, f)));
            }
            boolean quartet = layer.precision == Precision.QUARTET;
            float[] signs = !quartet ? null : backward ? signs(padded(k), layer.random) : layer.forwardSigns;
            SplittableRandom sr = quartet && backward ? layer.random : null;
            int pk = padded(k);
            float[] aq = mxfp4(av, n, k, false, signs, sr);
            float[] bq = mxfp4(bv, k, m, true, signs, sr);
            return Tensor.of(Shape.of(n, pk), aq).matmul(Tensor.of(Shape.of(pk, m), bq));
        }

        @Override
        public Tensor forward(Tensor... in) {
            return product(in[0], in[1], false);
        }

        @Override
        public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
            Tensor dx = product(grad, in[1].transpose(), true);   // [n, out] x [out, in]
            Tensor dw = product(in[0].transpose(), grad, true);   // [in, n] x [n, out]
            return new Tensor[] {dx, dw};
        }

        @Override
        public String name() {
            return "quartetMatmul";
        }
    }
}
