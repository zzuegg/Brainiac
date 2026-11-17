package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Op;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A dense layer trained for low-bit inference with LSQ, learned step size quantization: the
 * weights (and optionally the inputs) are fake-quantized to {@code bits}-bit integers times a
 * step size, and the step size is itself a parameter trained by gradient descent. Rounding
 * passes gradients straight through inside the clipping range; the step size gets the
 * gradient {@code round(v) - v} inside and the clip level outside, scaled by
 * {@code 1 / sqrt(N * Qp)} so it learns at the pace of the weights. Learning where to clip
 * lets 2-4-bit networks reach full-precision accuracy, beating fixed-scale QAT
 * ({@code Trainer.quantizationAware}).
 *
 * <p>Paper: Esser et al., "Learned Step Size Quantization" (2019),
 * <a href="https://arxiv.org/abs/1902.08153">arXiv:1902.08153</a>.
 */
public final class LsqDense implements Layer {

    private final int inputs;
    private final int units;
    private final int bits;
    private final int inputBits;
    private final Activation activation;
    private final Var weight;
    private final Var bias;
    private final Var weightStep;
    private final Var inputStep;

    /**
     * Creates the layer from existing weights.
     *
     * @param weight the weight {@code [inputs, units]}, copied
     * @param bias the bias {@code [units]}, copied, or {@code null}
     * @param activation the activation
     * @param bits weight bits, at least 2
     * @param inputBits input bits, or 0 to keep inputs in floating point
     */
    public LsqDense(Tensor weight, Tensor bias, Activation activation, int bits, int inputBits) {
        this.inputs = weight.shape().dim(0);
        this.units = weight.shape().dim(1);
        this.bits = bits;
        this.inputBits = inputBits;
        this.activation = activation.bind(units);
        this.weight = Var.param(weight);
        this.bias = Var.param(bias != null ? bias : Tensor.zeros(Shape.of(units)));
        int qp = QuantMath.levels(bits);
        float meanAbs = weight.abs().mean().item();
        this.weightStep = Var.param(Tensor.full(Shape.of(1), Math.max(1e-4f, 2 * meanAbs / (float) Math.sqrt(qp))));
        this.inputStep = inputBits > 0
                ? Var.param(Tensor.full(Shape.of(1), 2f / (float) Math.sqrt(QuantMath.levels(inputBits)))) : null;
    }

    /**
     * Turns a trained dense layer into an LSQ layer for quantization-aware fine-tuning.
     *
     * @param dense the layer, whose weights are copied
     * @param bits weight bits
     * @param inputBits input bits, or 0 for none
     * @return the layer
     */
    public static LsqDense from(Dense dense, int bits, int inputBits) {
        return new LsqDense(dense.weight().value(), dense.bias() == null ? null : dense.bias().value(),
                dense.activation(), bits, inputBits);
    }

    /**
     * Describes a fresh layer for {@code Model.sequential().add(...)}.
     *
     * @param units the output width
     * @param activation the activation
     * @param bits weight bits
     * @return the spec
     */
    public static LayerSpec spec(int units, Activation activation, int bits) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] d = input.toArray();
                d[d.length - 1] = units;
                return Shape.of(d);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return create(input.dim(input.rank() - 1), random);
            }

            private Layer create(int in, RandomGenerator random) {
                return new LsqDense(TechniqueOps.weight(random, in, units).value(), null, activation, bits, 0);
            }
        };
    }

    /**
     * LSQ fake quantization of any values.
     *
     * @param x the values
     * @param step the learned step size {@code [1]}
     * @param bits the bits (symmetric, {@code Qn = Qp = 2^(bits-1) - 1})
     * @return {@code round(clamp(x / step)) * step} with LSQ gradients
     */
    public static Var quantize(Var x, Var step, int bits) {
        int qp = QuantMath.levels(bits);
        return Var.apply(new LsqOp(qp, (float) (1 / Math.sqrt(x.shape().size() * (double) qp))), x, step);
    }

    @Override
    public Var forward(Var input) {
        Var x = inputStep != null ? quantize(input, inputStep, inputBits) : input;
        return activation.forward(x.matmul(quantize(weight, weightStep, bits)).add(bias));
    }

    /** @return the weights as the layer uses them, on the integer grid */
    public Tensor quantizedWeight() {
        return quantize(Var.of(weight.value()), Var.of(weightStep.value()), bits).value();
    }

    /** @return the learned weight step size */
    public Var weightStep() {
        return weightStep;
    }

    @Override
    public int outputSize(int inputSize) {
        return units;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(weight, bias, weightStep));
        if (inputStep != null) {
            p.add(inputStep);
        }
        p.addAll(activation.parameters());
        return p;
    }

    @Override
    public String toString() {
        return "LsqDense(" + inputs + " -> " + units + ", " + bits + " bits)";
    }

    /** Fake quantization with LSQ's straight-through and step-size gradients. */
    static final class LsqOp implements Op {

        private final int qp;
        private final float gradScale;

        LsqOp(int qp, float gradScale) {
            this.qp = qp;
            this.gradScale = gradScale;
        }

        @Override
        public Tensor forward(Tensor... in) {
            Tensor v = in[0].div(in[1]);
            return v.round().maximum(Tensor.scalar(-qp).to(v.device())).minimum(Tensor.scalar(qp).to(v.device()))
                    .mul(in[1]);
        }

        @Override
        public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
            Tensor v = in[0].div(in[1]);
            Tensor lo = Tensor.scalar(-qp).to(v.device());
            Tensor hi = Tensor.scalar(qp).to(v.device());
            Tensor inside = v.greater(lo).mul(hi.greater(v));
            Tensor clipped = v.round().maximum(lo).minimum(hi);
            Tensor dsElem = inside.mul(v.round().sub(v)).add(inside.neg().add(1f).mul(clipped));
            Tensor ds = grad.mul(dsElem).sum().mul(gradScale).reshape(in[1].shape().toArray());
            return new Tensor[] {grad.mul(inside), ds};
        }

        @Override
        public String name() {
            return "lsq";
        }
    }
}
