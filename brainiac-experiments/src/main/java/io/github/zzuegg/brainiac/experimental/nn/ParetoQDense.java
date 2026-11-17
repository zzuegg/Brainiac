package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A dense layer with ParetoQ's bit-specific weight quantizers for quantization-aware training
 * at 1, 1.58 (ternary), 2, 3 and 4 bits. ParetoQ found that the best quantizer function changes
 * with the bit width, and that low-bit training does best as a short QAT phase after mostly
 * full-precision training (see {@code experimental.train.ParetoQSchedule}):
 * <ul>
 * <li>1 bit: {@code alpha sign(w)};</li>
 * <li>1.58 bits: ternary {@code alpha round(clip(w / alpha, -1, 1))};</li>
 * <li>2 bits: <em>stretched elastic quantization</em> (SEQ), four levels without zero spread
 *     evenly over {@code [-alpha, alpha]}: {@code alpha (round(clip(w / alpha) * k / 2 - 1/2) +
 *     1/2) 2 / k}, {@code k = 4} (a zero level wastes one of only four levels);</li>
 * <li>3 and 4 bits: LSQ, {@code alpha round(clip(w / alpha, -2^(b-1), 2^(b-1) - 1))}.</li>
 * </ul>
 * The scale {@code alpha} is learned per output unit (straight-through rounding, gradients to
 * {@code alpha} through the quantizer as in LSQ); it is set from the weights when QAT starts
 * (the largest magnitude for SEQ, ternary and sign; {@code 2 mean|w| / sqrt(Q_p)} for LSQ).
 * Before {@link #quantize(boolean) quantization is switched on} the layer is an ordinary dense
 * layer. Activations stay in full precision (weight-only QAT, as in the paper's main results).
 *
 * <p>Liu et al., "ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization" (2025),
 * <a href="https://arxiv.org/abs/2502.02631">arXiv:2502.02631</a>.
 */
public final class ParetoQDense implements Layer {

    private final int inputs;
    private final int units;
    private final double bits;
    private final Var weight;
    private final Var bias;
    private final Var alpha;
    private final Activation activation;
    private boolean quantized;

    /**
     * @param weight the initial weight {@code [inputs, units]}
     * @param bias the initial bias {@code [units]}, or {@code null}
     * @param activation the activation
     * @param bits 1, 1.58, 2, 3 or 4
     */
    public ParetoQDense(Tensor weight, Tensor bias, Activation activation, double bits) {
        if (bits != 1 && bits != 1.58 && bits != 2 && bits != 3 && bits != 4) {
            throw new IllegalArgumentException("bits must be 1, 1.58, 2, 3 or 4, got " + bits);
        }
        this.inputs = weight.shape().dim(0);
        this.units = weight.shape().dim(1);
        this.bits = bits;
        this.weight = Var.param(weight);
        this.bias = bias == null ? null : Var.param(bias);
        this.alpha = Var.param(Tensor.full(Shape.of(1, units), 1f));
        this.activation = activation;
    }

    /**
     * A layer with the weights of a dense layer.
     *
     * @param dense the layer
     * @param bits the bit width
     * @return the layer, quantization off
     */
    public static ParetoQDense from(Dense dense, double bits) {
        return new ParetoQDense(dense.weight().value(), dense.bias() == null ? null : dense.bias().value(),
                dense.activation(), bits);
    }

    /**
     * Describes the layer for model builders.
     *
     * @param units the output size
     * @param activation the activation
     * @param bits the bit width
     * @return the spec
     */
    public static LayerSpec spec(int units, Activation activation, double bits) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] d = input.toArray();
                d[d.length - 1] = units;
                return Shape.of(d);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                int in = input.dim(-1);
                return new ParetoQDense(Init.XAVIER.create(Shape.of(in, units), in, units, random),
                        Tensor.zeros(Shape.of(units)), activation, bits);
            }
        };
    }

    /** @return the bit width */
    public double bits() {
        return bits;
    }

    /** @return whether the forward pass quantizes the weights */
    public boolean quantized() {
        return quantized;
    }

    /**
     * Switches quantization on (initializing the scales from the current weights) or off.
     *
     * @param on whether to quantize
     */
    public void quantize(boolean on) {
        if (on && !quantized) {
            float[] w = weight.value().toArray();
            float[] a = new float[units];
            for (int u = 0; u < units; u++) {
                double max = 0;
                double mean = 0;
                for (int i = 0; i < inputs; i++) {
                    double v = Math.abs(w[i * units + u]);
                    max = Math.max(max, v);
                    mean += v / inputs;
                }
                a[u] = (float) Math.max(1e-6, bits >= 3 ? 2 * mean / Math.sqrt((1 << ((int) bits - 1)) - 1) : max);
            }
            alpha.assign(Tensor.of(Shape.of(1, units), a, alpha.value().device()));
        }
        this.quantized = on;
    }

    /** Rounding with an identity gradient. */
    private static Var roundSte(Var x) {
        return x.add(Var.of(x.value().round().sub(x.value())));
    }

    /** @return the quantized weight as a differentiable function of the latent weight and the scales */
    public Var quantizedWeight() {
        Var u = weight.div(alpha);
        Var r;
        if (bits == 1) {
            // sign with sign(0) = +1, so there are exactly two levels; identity gradient inside [-1, 1]
            Tensor sign = u.value().mul(-1f).step().mul(-2f).add(1f);
            Var c = u.clamp(-1, 1);
            r = c.add(Var.of(sign.sub(c.value())));
        } else if (bits == 1.58) {
            r = roundSte(u.clamp(-1, 1));
        } else if (bits == 2) {
            int k = 4;
            Var s = roundSte(u.clamp(-1, 1).mul(k / 2f).sub(0.5f));
            s = s.maximum(-k / 2f).minimum(k / 2f - 1);
            r = s.add(0.5f).mul(2f / k);
        } else {
            int qp = (1 << ((int) bits - 1)) - 1;
            r = roundSte(u.clamp(-qp - 1, qp));
        }
        return r.mul(alpha);
    }

    @Override
    public Var forward(Var input) {
        Var w = quantized ? quantizedWeight() : weight;
        Var y = input.matmul(w);
        if (bias != null) {
            y = y.add(bias);
        }
        return activation.forward(y);
    }

    @Override
    public List<Var> parameters() {
        return bias == null ? List.of(weight, alpha) : List.of(weight, bias, alpha);
    }

    @Override
    public int outputSize(int inputSize) {
        return units;
    }

    @Override
    public String toString() {
        return "ParetoQDense[" + inputs + " -> " + units + ", " + bits + " bits" + (quantized ? ", quantized" : "")
                + "]";
    }
}
