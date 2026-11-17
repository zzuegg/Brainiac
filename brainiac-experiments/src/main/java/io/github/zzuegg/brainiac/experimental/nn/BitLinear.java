package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.tensor.TernaryMatrix;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * The BitNet b1.58 linear layer (Ma et al., <a href="https://arxiv.org/abs/2402.17764">2402.17764</a>;
 * BitNet, <a href="https://arxiv.org/abs/2310.11453">2310.11453</a>): a drop-in replacement for
 * a bias-free {@code Dense} whose weights are ternary {-1, 0, +1} times one scale and whose
 * input activations are 8-bit, trained with quantization-aware training.
 *
 * <p>Training keeps full-precision "latent" weights. The forward pass quantizes them with
 * the absmean rule ({@code scale = mean|W|}, {@code W_q = clip(round(W / scale), -1, 1) * scale})
 * and the activations per row with absmax to 8 bits, and uses the straight-through
 * estimator ({@code W + stopgrad(W_q - W)}) so gradients reach the latent weights as if no
 * rounding happened. After training, {@link #pack} produces a {@link TernaryMatrix} whose
 * product needs only additions.
 */
public final class BitLinear implements Layer {
    private final int inputs;
    private final int units;
    private final Var weight;

    private BitLinear(int inputs, int units, SplittableRandom random) {
        this.inputs = inputs;
        this.units = units;
        this.weight = Var.param(Init.XAVIER.create(Shape.of(inputs, units), inputs, units, random));
    }

    /**
     * Creates the layer.
     *
     * @param inputs the input size
     * @param units the output size
     * @param seed the initialization seed
     * @return the layer
     */
    public static BitLinear of(int inputs, int units, long seed) {
        return new BitLinear(inputs, units, new SplittableRandom(seed));
    }

    /**
     * Describes the layer for model builders.
     *
     * @param units the output size
     * @return the spec
     */
    public static LayerSpec spec(int units) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] d = input.toArray();
                d[d.length - 1] = units;
                return Shape.of(d);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new BitLinear(input.dim(-1), units, random);
            }
        };
    }

    /**
     * Ternary-quantizes weights (absmean).
     *
     * @param w the latent weights
     * @return the dequantized ternary weights, same shape
     */
    public static Tensor quantizeWeights(Tensor w) {
        float scale = Math.max(w.abs().mean().item(), 1e-8f);
        return w.div(scale).round().maximum(Tensor.full(Shape.of(), -1f)).minimum(Tensor.full(Shape.of(), 1f))
                .mul(scale);
    }

    /**
     * Quantizes activations to 8 bits per row (absmax) and back.
     *
     * @param x activations, last axis the features
     * @return the fake-quantized activations
     */
    public static Tensor quantizeActivations(Tensor x) {
        int last = x.shape().rank() - 1;
        Tensor max = x.abs().max(last, true).maximum(Tensor.full(Shape.of(), 1e-5f));
        Tensor s = Tensor.full(Shape.of(), 127f).div(max);
        return x.mul(s).round().maximum(Tensor.full(Shape.of(), -128f)).minimum(Tensor.full(Shape.of(), 127f)).div(s);
    }

    @Override
    public Var forward(Var input) {
        if (input.shape().dim(-1) != inputs) {
            throw new IllegalArgumentException("BitLinear expects [*, " + inputs + "], got " + input.shape());
        }
        Var w = weight.add(Var.of(quantizeWeights(weight.value()).sub(weight.value())));   // straight-through
        Var x = input.add(Var.of(quantizeActivations(input.value()).sub(input.value())));
        return x.matmul(w);
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight);
    }

    @Override
    public int outputSize(int inputSize) {
        return units;
    }

    /** @return the latent full-precision weights {@code [inputs, units]} */
    public Var weight() {
        return weight;
    }

    /**
     * Packs the quantized weights for addition-only inference.
     *
     * @return the ternary matrix
     */
    public TernaryMatrix pack() {
        return TernaryMatrix.quantize(weight.value().toArray(), inputs, units);
    }
}
