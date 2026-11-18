package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.PadMode;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * SNAIL ("simple neural attentive learner"): a meta-learner that reads a whole episode as a
 * sequence and predicts the label of each step from everything before it. It interleaves two
 * kinds of causal block, each of which appends its output to its input as new features
 * (dense-net style):
 *
 * <ul>
 *   <li>a <b>temporal-convolution block</b>: a stack of gated dilated causal convolutions
 *       (kernel 2, dilations 1, 2, 4, ... up to the sequence length), so every step sees a
 *       summary of the whole past with logarithmic depth;</li>
 *   <li>a <b>causal attention block</b>: single-head soft attention over all earlier steps,
 *       which can pinpoint one specific earlier example, such as the support item that
 *       matches the query.</li>
 * </ul>
 *
 * <p>For few-shot classification each step is an input concatenated with its one-hot label
 * (zeros for the query, the last step); for regression it is {@code (x, y)}. There is no
 * inner loop: adaptation is one forward pass.
 *
 * <p>Input {@code [batch, length, inputs]}, output {@code [batch, length, outputs]}.
 *
 * <p>Paper: Mishra, Rohaninejad, Chen, Abbeel, "A Simple Neural Attentive Meta-Learner"
 * (2018), <a href="https://arxiv.org/abs/1707.03141">arXiv:1707.03141</a>.
 */
public final class SnailLearner implements Layer {

    private interface Block {
        Var forward(Var x, int batch, int length);

        int outputs();

        List<Var> parameters();
    }

    /** A gated dilated causal convolution of kernel 2 that appends {@code filters} features. */
    private record DenseBlock(int dilation, int in, int filters, Var now, Var past, Var bias) implements Block {

        @Override
        public Var forward(Var x, int batch, int length) {
            Var shifted = dilation >= length ? Var.of(Tensor.zeros(x.shape()).to(x.value().device()))
                    : Functional.padAxis(x.slice(1, 0, length - dilation), 1, dilation, 0, PadMode.CONSTANT, 0f);
            Var h = x.reshape(batch * length, in).matmul(now)
                    .add(shifted.reshape(batch * length, in).matmul(past)).add(bias);
            Var gated = h.slice(1, 0, filters).tanh().mul(h.slice(1, filters, 2 * filters).sigmoid());
            return Var.concat(2, x, gated.reshape(batch, length, filters));
        }

        @Override
        public int outputs() {
            return in + filters;
        }

        @Override
        public List<Var> parameters() {
            return List.of(now, past, bias);
        }
    }

    /** Single-head causal attention that appends {@code values} features. */
    private record AttentionBlock(int in, int keys, int values, Var wq, Var wk, Var wv) implements Block {

        @Override
        public Var forward(Var x, int batch, int length) {
            Var flat = x.reshape(batch * length, in);
            Var q = flat.matmul(wq).reshape(batch, 1, length, keys);
            Var k = flat.matmul(wk).reshape(batch, 1, length, keys);
            Var v = flat.matmul(wv).reshape(batch, 1, length, values);
            Var read = Functional.attention(q, k, v, true, (float) (1 / Math.sqrt(keys)));
            return Var.concat(2, x, read.reshape(batch, length, values));
        }

        @Override
        public int outputs() {
            return in + values;
        }

        @Override
        public List<Var> parameters() {
            return List.of(wq, wk, wv);
        }
    }

    private final int inputs;
    private final int outputs;
    private final List<Block> blocks;
    private final Var outWeight;
    private final Var outBias;

    /**
     * Creates the learner: temporal-convolution, attention, temporal-convolution, attention,
     * then a linear read-out per step.
     *
     * @param inputs features per step (input plus label encoding)
     * @param outputs predictions per step
     * @param length the episode length, which sets the dilations
     * @param filters features each dilated convolution appends
     * @param keys the attention key size
     * @param values features each attention block appends
     * @param random the weight initializer
     */
    public SnailLearner(int inputs, int outputs, int length, int filters, int keys, int values,
            RandomGenerator random) {
        this(inputs, outputs, length, filters, keys, values, 2, random);
    }

    /**
     * Creates the learner with a chosen number of (temporal-convolution, attention) rounds.
     *
     * @param inputs features per step
     * @param outputs predictions per step
     * @param length the episode length, which sets the dilations
     * @param filters features each dilated convolution appends
     * @param keys the attention key size
     * @param values features each attention block appends
     * @param rounds how many temporal-convolution plus attention pairs
     * @param random the weight initializer
     */
    public SnailLearner(int inputs, int outputs, int length, int filters, int keys, int values, int rounds,
            RandomGenerator random) {
        this.inputs = inputs;
        this.outputs = outputs;
        List<Block> list = new ArrayList<>();
        int width = inputs;
        for (int round = 0; round < rounds; round++) {
            for (int d = 1; d < length; d *= 2) {
                Block b = new DenseBlock(d, width, filters, weight(width, 2 * filters, random),
                        weight(width, 2 * filters, random), Var.param(Tensor.zeros(Shape.of(2 * filters))));
                list.add(b);
                width = b.outputs();
            }
            Block a = new AttentionBlock(width, keys, values, weight(width, keys, random), weight(width, keys, random),
                    weight(width, values, random));
            list.add(a);
            width = a.outputs();
        }
        this.blocks = List.copyOf(list);
        this.outWeight = weight(width, outputs, random);
        this.outBias = Var.param(Tensor.zeros(Shape.of(outputs)));
    }

    /**
     * Describes a SNAIL learner for {@code Model.sequential().add(...)}; the input shape is
     * {@code [length, inputs]}.
     *
     * @param outputs predictions per step
     * @param filters features per dilated convolution
     * @param keys the attention key size
     * @param values features per attention block
     * @return the spec
     */
    public static LayerSpec spec(int outputs, int filters, int keys, int values) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                if (input.rank() != 2) {
                    throw new IllegalArgumentException("SNAIL takes [length, features] samples, got " + input);
                }
                return Shape.of(input.dim(0), outputs);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new SnailLearner(input.dim(1), outputs, input.dim(0), filters, keys, values, random);
            }
        };
    }

    private static Var weight(int in, int out, RandomGenerator random) {
        return Var.param(Tensor.normal(Shape.of(in, out), 0f, (float) Math.sqrt(1.0 / in), random));
    }

    @Override
    public Var forward(Var input) {
        if (input.shape().rank() != 3 || input.shape().dim(2) != inputs) {
            throw new IllegalArgumentException("expected [batch, length, " + inputs + "], got " + input.shape());
        }
        int batch = input.shape().dim(0);
        int length = input.shape().dim(1);
        Var x = input;
        for (Block b : blocks) {
            x = b.forward(x, batch, length);
        }
        int width = x.shape().dim(2);
        return x.reshape(batch * length, width).matmul(outWeight).add(outBias).reshape(batch, length, outputs);
    }

    /**
     * The prediction at the last step of every sequence, the query in an episode.
     *
     * @param input {@code [batch, length, inputs]}
     * @return {@code [batch, outputs]}
     */
    public Var last(Var input) {
        Var out = forward(input);
        int length = out.shape().dim(1);
        return out.slice(1, length - 1, length).reshape(out.shape().dim(0), outputs);
    }

    @Override
    public List<Var> parameters() {
        List<Var> out = new ArrayList<>();
        for (Block b : blocks) {
            out.addAll(b.parameters());
        }
        out.add(outWeight);
        out.add(outBias);
        return out;
    }

    @Override
    public int outputSize(int inputSize) {
        return outputs;
    }

    @Override
    public String toString() {
        return "SnailLearner(" + inputs + " -> " + outputs + ", " + blocks.size() + " blocks)";
    }
}
