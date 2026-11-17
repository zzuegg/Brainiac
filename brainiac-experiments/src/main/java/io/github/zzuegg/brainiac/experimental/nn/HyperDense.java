package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A dense layer whose weights are generated per example by a hypernetwork (Ha, Dai and Le,
 * <a href="https://arxiv.org/abs/1609.09106">1609.09106</a>): a small MLP maps a context
 * vector {@code z} to the weight matrix and bias, {@code [W(z), b(z)] = H(z)}, and the layer
 * computes {@code y = x W(z) + b(z)}. Only the hypernetwork is trained, so one set of
 * parameters encodes a whole family of target networks, indexed by the context (a task
 * description, a layer embedding, …).
 *
 * <p>{@link #forward(Var, Var)} takes context and input separately. As a single-input
 * {@link Layer} the input packs {@code [batch, context + inputs]} and the output is
 * {@code [batch, context + outputs]}, the context passed through, so generated layers stack in
 * a sequential model.
 */
public final class HyperDense implements Layer {
    private final int context;
    private final int inputs;
    private final int outputs;
    private final Var hidden;
    private final Var hiddenBias;
    private final Var head;
    private final Var headBias;

    private HyperDense(int context, int inputs, int outputs, int width, SplittableRandom random) {
        this.context = context;
        this.inputs = inputs;
        this.outputs = outputs;
        this.hidden = Var.param(Init.XAVIER.create(Shape.of(context, width), context, width, random));
        this.hiddenBias = Var.param(Tensor.zeros(Shape.of(width)));
        int generated = inputs * outputs + outputs;
        // scale so generated weights start with roughly He variance for the target layer
        float scale = (float) Math.sqrt(4.0 / inputs / width);
        this.head = Var.param(Tensor.normal(Shape.of(width, generated), 0, scale, random));
        this.headBias = Var.param(Tensor.zeros(Shape.of(generated)));
    }

    /**
     * Creates a layer.
     *
     * @param context the context width
     * @param inputs the target layer's input width
     * @param outputs the target layer's output width
     * @param width the hypernetwork's hidden width
     * @param seed the seed
     * @return the layer
     */
    public static HyperDense of(int context, int inputs, int outputs, int width, long seed) {
        return new HyperDense(context, inputs, outputs, width, new SplittableRandom(seed));
    }

    /**
     * Describes the layer for model builders; input {@code [context + inputs]}, output
     * {@code [context + outputs]}.
     *
     * @param context the context width
     * @param outputs the target output width
     * @param width the hypernetwork width
     * @return the spec
     */
    public static LayerSpec spec(int context, int outputs, int width) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(context + outputs);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new HyperDense(context, input.dim(-1) - context, outputs, width, random);
            }
        };
    }

    /**
     * Generates the target weights.
     *
     * @param z the context {@code [batch, context]}
     * @return {@code [batch, inputs * outputs + outputs]}: the row-major weight matrix, then the bias
     */
    public Var generate(Var z) {
        return z.matmul(hidden).add(hiddenBias).tanh().matmul(head).add(headBias);
    }

    /**
     * Applies the generated layer (no activation).
     *
     * @param z the context {@code [batch, context]}
     * @param x the input {@code [batch, inputs]}
     * @return {@code [batch, outputs]}
     */
    public Var forward(Var z, Var x) {
        int b = x.shape().dim(0);
        Var g = generate(z);
        Var w = g.slice(1, 0, inputs * outputs).reshape(b, inputs, outputs);
        Var bias = g.slice(1, inputs * outputs, inputs * outputs + outputs);
        return x.reshape(b, 1, inputs).matmul(w).reshape(b, outputs).add(bias);
    }

    @Override
    public Var forward(Var input) {
        if (input.shape().rank() != 2 || input.shape().dim(1) != context + inputs) {
            throw new IllegalArgumentException("HyperDense expects [batch, " + (context + inputs) + "], got "
                    + input.shape());
        }
        Var z = input.slice(1, 0, context);
        return Var.concat(1, z, forward(z, input.slice(1, context, context + inputs)));
    }

    @Override
    public List<Var> parameters() {
        return List.of(hidden, hiddenBias, head, headBias);
    }

    @Override
    public int outputSize(int inputSize) {
        return context + outputs;
    }
}
