package io.github.zzuegg.brainiac.experimental.parallel;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.random.RandomGenerator;

/**
 * Splits the backward pass into its two halves, as zero-bubble pipeline schedules need: B, the
 * gradient with respect to a stage's input (which the previous stage waits for), and W, the
 * gradient with respect to its weights (which nobody waits for and can run later to fill a
 * bubble).
 *
 * <p>The core autograd computes both at once, so this class provides linear layers whose
 * matrix product is a custom autograd operation that treats its weight as a constant: during
 * the backward pass it returns only the input gradient {@code g W^T} and records the pair
 * {@code (x, g)} under the micro-batch being processed; {@link #backwardWeight} later computes
 * {@code x^T g} and the bias gradient and adds them to the parameters' gradients. Everything
 * else (activations, attention, residuals) is ordinary autograd, so a transformer block built
 * from {@link #linear} layers and {@link #block} splits exactly. Without deferral
 * ({@link #deferring(boolean)} false) the weight gradients are computed inside the backward pass,
 * which is the ordinary combined backward.
 *
 * <p>One tape belongs to one pipeline stage and is used by one thread at a time.
 *
 * <p>Paper: Qi et al., "Zero Bubble Pipeline Parallelism", ICLR 2024,
 * <a href="https://arxiv.org/abs/2401.10241">arXiv:2401.10241</a>.
 */
public final class SplitBackward {

    private final Map<Integer, List<Runnable>> deferred = new HashMap<>();
    private int current = -1;
    private boolean deferring = true;

    /** Creates an empty tape. */
    public SplitBackward() {
    }

    /**
     * @param on whether weight gradients wait for {@link #backwardWeight} (true, the default) or
     *     are computed inside the backward pass
     * @return this tape
     */
    public SplitBackward deferring(boolean on) {
        this.deferring = on;
        return this;
    }

    /**
     * Runs the B half for one micro-batch: back-propagates {@code seed} from {@code output} and
     * returns the gradient of {@code input}, recording the weight-gradient work for later.
     *
     * @param microBatch the micro-batch
     * @param output the stage's output
     * @param seed the gradient of the output
     * @param input the stage's input, created with {@link Var#param}
     * @return the input's gradient
     */
    public Tensor backwardInput(int microBatch, Var output, Tensor seed, Var input) {
        current = microBatch;
        try {
            output.backward(seed);
        } finally {
            current = -1;
        }
        return input.grad();
    }

    /**
     * Runs the W half for one micro-batch: the weight gradients recorded by its B half.
     *
     * @param microBatch the micro-batch
     */
    public void backwardWeight(int microBatch) {
        List<Runnable> work = deferred.remove(microBatch);
        if (work != null) {
            work.forEach(Runnable::run);
        }
    }

    /** @return micro-batches whose W half has not run yet */
    public int pending() {
        return deferred.size();
    }

    private void defer(Runnable weightGradient) {
        if (!deferring || current < 0) {
            weightGradient.run();
        } else {
            deferred.computeIfAbsent(current, k -> new ArrayList<>()).add(weightGradient);
        }
    }

    private static void accumulate(Var p, Tensor g) {
        p.setGrad(p.grad() == null ? g : p.grad().add(g));
    }

    /**
     * A linear layer {@code act(x W + b)} whose weight gradient is deferred to the W half.
     *
     * @param inputs input width
     * @param outputs output width
     * @param activation the activation
     * @param random initialises the weight (scaled normal, {@code 1 / sqrt(inputs)})
     * @return the layer
     */
    public Linear linear(int inputs, int outputs, Activation activation, RandomGenerator random) {
        Tensor w = Tensor.normal(Shape.of(inputs, outputs), 0, (float) (1 / Math.sqrt(inputs)), random);
        return new Linear(this, Var.param(w), Var.param(Tensor.zeros(Shape.of(outputs))), activation);
    }

    /**
     * A pre-norm transformer block {@code x + attn(norm(x))}, then {@code x + mlp(norm(x))}, with
     * every matrix a {@link #linear}; the norm is a parameter-free RMS normalisation.
     *
     * @param width the model width
     * @param heads attention heads
     * @param hidden the MLP width
     * @param causal causal attention
     * @param random initialises the weights
     * @return the block, mapping {@code [batch, sequence, width]} to the same shape
     */
    public Block block(int width, int heads, int hidden, boolean causal, RandomGenerator random) {
        return new Block(this, width, heads, hidden, causal, random);
    }

    /** A linear layer whose weight gradient runs in the W half. */
    public static final class Linear implements Layer {

        private final SplitBackward tape;
        private final Var weight;
        private final Var bias;
        private final Activation activation;

        /**
         * Creates the layer around existing parameters.
         *
         * @param tape the tape that defers its weight gradients
         * @param weight {@code [inputs, outputs]}
         * @param bias {@code [outputs]}
         * @param activation the activation
         */
        public Linear(SplitBackward tape, Var weight, Var bias, Activation activation) {
            this.tape = tape;
            this.weight = weight;
            this.bias = bias;
            this.activation = activation;
        }

        @Override
        public Var forward(Var input) {
            Shape s = input.shape();
            int in = weight.shape().dim(0);
            int out = weight.shape().dim(1);
            int rows = Math.toIntExact(s.size() / in);
            Var x2 = input.reshape(rows, in);
            Var y = Var.custom(v -> v[0].matmul(weight.value()).add(bias.value()), (inputs, output, g) -> {
                Tensor x = inputs[0];
                tape.defer(() -> {
                    accumulate(weight, x.transpose().matmul(g));
                    accumulate(bias, g.sum(0));
                });
                return new Tensor[] {g.matmul(weight.value().transpose())};
            }, x2);
            int[] dims = s.toArray();
            dims[dims.length - 1] = out;
            Var r = y.reshape(dims);
            return activation == Activation.IDENTITY ? r : activation.forward(r);
        }

        @Override
        public List<Var> parameters() {
            return List.of(weight, bias);
        }

        @Override
        public int outputSize(int inputSize) {
            return weight.shape().dim(1);
        }

        /** @return the weight {@code [inputs, outputs]} */
        public Var weight() {
            return weight;
        }

        /** @return the bias */
        public Var bias() {
            return bias;
        }
    }

    /** A pre-norm transformer block built from deferred-weight-gradient linear layers. */
    public static final class Block implements Layer {

        private final int heads;
        private final boolean causal;
        private final Linear q;
        private final Linear k;
        private final Linear v;
        private final Linear o;
        private final Linear up;
        private final Linear down;

        Block(SplitBackward tape, int width, int heads, int hidden, boolean causal, RandomGenerator random) {
            if (width % heads != 0) {
                throw new IllegalArgumentException("width must divide into heads");
            }
            this.heads = heads;
            this.causal = causal;
            this.q = tape.linear(width, width, Activation.IDENTITY, random);
            this.k = tape.linear(width, width, Activation.IDENTITY, random);
            this.v = tape.linear(width, width, Activation.IDENTITY, random);
            this.o = tape.linear(width, width, Activation.IDENTITY, random);
            this.up = tape.linear(width, hidden, Activation.RELU, random);
            this.down = tape.linear(hidden, width, Activation.IDENTITY, random);
        }

        /**
         * Parameter-free RMS normalisation over the last axis.
         *
         * @param x the input
         * @return {@code x / sqrt(mean(x^2) + 1e-5)}
         */
        public static Var rms(Var x) {
            int last = x.shape().rank() - 1;
            return x.div(x.square().mean(last, true).add(1e-5f).sqrt());
        }

        @Override
        public Var forward(Var x) {
            int b = x.shape().dim(0);
            int s = x.shape().dim(1);
            int d = x.shape().dim(2);
            int dh = d / heads;
            Var n = rms(x);
            Var qh = q.forward(n).reshape(b, s, heads, dh).permute(0, 2, 1, 3);
            Var kh = k.forward(n).reshape(b, s, heads, dh).permute(0, 2, 1, 3);
            Var vh = v.forward(n).reshape(b, s, heads, dh).permute(0, 2, 1, 3);
            Var a = Functional.attention(qh, kh, vh, causal, (float) (1 / Math.sqrt(dh)));
            Var h = x.add(o.forward(a.permute(0, 2, 1, 3).reshape(b, s, d)));
            return h.add(down.forward(up.forward(rms(h))));
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>();
            for (Linear l : List.of(q, k, v, o, up, down)) {
                p.addAll(l.parameters());
            }
            return p;
        }
    }
}
