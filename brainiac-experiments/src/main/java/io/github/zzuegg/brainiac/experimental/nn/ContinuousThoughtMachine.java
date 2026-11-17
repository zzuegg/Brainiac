package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Continuous Thought Machine (Darlow et al., <a href="https://arxiv.org/abs/2505.05522">2505.05522</a>):
 * a network that thinks over internal ticks, decoupled from the input length, with two
 * unusual ideas:
 * <ul>
 *   <li><b>Neuron-level models.</b> Every neuron keeps a private history of its last
 *       {@code memory} pre-activations and has its own tiny MLP (private weights) that turns
 *       that history into its post-activation. Neurons are little temporal processors, not
 *       point non-linearities.</li>
 *   <li><b>Synchronization as the representation.</b> The output is read from how pairs of
 *       neurons fire together over the ticks: for sampled pairs {@code (i, j)} the decayed
 *       inner product {@code S_ij = sum_t r^(T-t) z_i(t) z_j(t) / sqrt(sum_t r^(T-t))},
 *       with a learned decay {@code r} per pair.</li>
 * </ul>
 * Each tick, a synapse MLP mixes the post-activations with the input into new
 * pre-activations. Every tick produces logits; {@link #tickLoss} trains, as in the paper,
 * on the tick with the lowest loss and the tick with the highest certainty, so the model
 * can answer easy inputs early and think longer on hard ones.
 */
public final class ContinuousThoughtMachine implements Layer {

    private final int neurons;
    private final int memory;
    private final int ticks;
    private final int outputs;
    private final int pairs;
    private final Var synapseW;
    private final Var synapseB;
    private final Var nlmW1;
    private final Var nlmB1;
    private final Var nlmW2;
    private final Var nlmB2;
    private final Var decay;
    private final Var outW;
    private final Var outB;
    private final Var startTrace;
    private final Var startPost;
    private final Tensor left;
    private final Tensor right;

    private ContinuousThoughtMachine(int inputs, int neurons, int memory, int nlmHidden, int ticks, int pairs,
            int outputs, SplittableRandom r) {
        this.neurons = neurons;
        this.memory = memory;
        this.ticks = ticks;
        this.outputs = outputs;
        this.pairs = pairs;
        this.synapseW = gauss(r, (float) Math.sqrt(1.0 / (inputs + neurons)), inputs + neurons, neurons);
        this.synapseB = Var.param(Tensor.zeros(Shape.of(1, neurons)));
        this.nlmW1 = gauss(r, (float) Math.sqrt(1.0 / memory), neurons, memory, nlmHidden);
        this.nlmB1 = Var.param(Tensor.zeros(Shape.of(neurons, 1, nlmHidden)));
        this.nlmW2 = gauss(r, (float) Math.sqrt(1.0 / nlmHidden), neurons, nlmHidden, 1);
        this.nlmB2 = Var.param(Tensor.zeros(Shape.of(neurons, 1, 1)));
        this.decay = Var.param(Tensor.zeros(Shape.of(1, pairs)));
        this.outW = gauss(r, (float) Math.sqrt(1.0 / pairs), pairs, outputs);
        this.outB = Var.param(Tensor.zeros(Shape.of(1, outputs)));
        this.startTrace = gauss(r, 0.1f, 1, neurons, memory);
        this.startPost = gauss(r, 0.1f, 1, neurons);
        float[] li = new float[neurons * pairs];
        float[] ri = new float[neurons * pairs];
        for (int p = 0; p < pairs; p++) {
            int i = r.nextInt(neurons);
            int j = r.nextInt(neurons);
            li[i * pairs + p] = 1;
            ri[j * pairs + p] = 1;
        }
        this.left = Tensor.of(Shape.of(neurons, pairs), li);
        this.right = Tensor.of(Shape.of(neurons, pairs), ri);
    }

    private static Var gauss(SplittableRandom r, float scale, int... dims) {
        Shape s = Shape.of(dims);
        float[] v = new float[Math.toIntExact(s.size())];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (r.nextGaussian() * scale);
        }
        return Var.param(Tensor.of(s, v));
    }

    /**
     * @param inputs input features
     * @param neurons the number of neurons
     * @param memory pre-activations each neuron remembers
     * @param nlmHidden hidden units of each neuron's private MLP
     * @param ticks internal thinking steps
     * @param pairs synchronized neuron pairs that form the representation
     * @param outputs output logits
     * @param seed the initialization seed
     * @return the layer
     */
    public static ContinuousThoughtMachine of(int inputs, int neurons, int memory, int nlmHidden, int ticks,
            int pairs, int outputs, long seed) {
        return new ContinuousThoughtMachine(inputs, neurons, memory, nlmHidden, ticks, pairs, outputs,
                new SplittableRandom(seed));
    }

    /**
     * @param neurons the number of neurons
     * @param memory pre-activations each neuron remembers
     * @param nlmHidden hidden units of each neuron-level model
     * @param ticks internal ticks
     * @param pairs synchronized pairs
     * @param outputs output logits
     * @return a spec for model builders
     */
    public static LayerSpec spec(int neurons, int memory, int nlmHidden, int ticks, int pairs, int outputs) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(outputs);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new ContinuousThoughtMachine(input.dim(input.rank() - 1), neurons, memory, nlmHidden, ticks,
                        pairs, outputs, random);
            }
        };
    }

    /**
     * Thinks for all ticks.
     *
     * @param input {@code [batch, inputs]}
     * @return the logits after each tick
     */
    public List<Var> ticks(Var input) {
        int batch = input.shape().dim(0);
        Var ones = Var.of(Tensor.ones(Shape.of(batch, 1)));
        Var trace = startTrace.mul(Var.of(Tensor.ones(Shape.of(batch, 1, 1))));       // [B, D, M]
        Var post = ones.matmul(startPost);                                              // [B, D]
        Var r = decay.neg().exp().neg().exp();                                          // in (0, 1)
        Var alpha = null;
        Var beta = null;
        List<Var> outs = new ArrayList<>();
        for (int t = 0; t < ticks; t++) {
            Var pre = Var.concat(1, input, post).matmul(synapseW).add(synapseB);
            pre = pre.mul(pre.sigmoid());
            // shift the private histories and append the new pre-activations
            trace = Var.concat(2, trace.slice(2, 1, memory), pre.reshape(batch, neurons, 1));
            // neuron-level models: [D, B, M] x [D, M, H] -> [D, B, H] -> [D, B, 1]
            Var h = trace.permute(1, 0, 2).matmul(nlmW1).add(nlmB1);
            h = h.mul(h.sigmoid());
            post = h.matmul(nlmW2).add(nlmB2).reshape(neurons, batch).transpose();
            // synchronization of sampled pairs, decayed over ticks
            Var prod = post.matmul(Var.of(left)).mul(post.matmul(Var.of(right)));      // [B, P]
            alpha = alpha == null ? prod : alpha.mul(r).add(prod);
            beta = beta == null ? Var.of(Tensor.ones(Shape.of(1, pairs))) : beta.mul(r).add(1f);
            Var sync = alpha.div(beta.sqrt());
            outs.add(sync.matmul(outW).add(outB));
        }
        return outs;
    }

    /**
     * The CTM training loss: the mean of the cross-entropy at the tick with the lowest loss
     * and at the tick with the highest certainty (lowest mean entropy).
     *
     * @param tickLogits the logits of every tick, as {@link #ticks} returns
     * @param labels the class ids {@code [batch]}
     * @return the loss
     */
    public static Var tickLoss(List<Var> tickLogits, Tensor labels) {
        int bestLoss = 0;
        int mostCertain = 0;
        double lowLoss = Double.POSITIVE_INFINITY;
        double lowEntropy = Double.POSITIVE_INFINITY;
        List<Var> losses = new ArrayList<>();
        for (int t = 0; t < tickLogits.size(); t++) {
            Var l = Functional.crossEntropy(tickLogits.get(t), labels);
            losses.add(l);
            double lv = l.value().item();
            if (lv < lowLoss) {
                lowLoss = lv;
                bestLoss = t;
            }
            Var logp = Activation.LOG_SOFTMAX.forward(tickLogits.get(t).detach());
            double entropy = -logp.exp().mul(logp).sum().value().item();
            if (entropy < lowEntropy) {
                lowEntropy = entropy;
                mostCertain = t;
            }
        }
        return losses.get(bestLoss).add(losses.get(mostCertain)).mul(0.5f);
    }

    @Override
    public Var forward(Var input) {
        List<Var> outs = ticks(input);
        return outs.get(outs.size() - 1);
    }

    @Override
    public List<Var> parameters() {
        return List.of(synapseW, synapseB, nlmW1, nlmB1, nlmW2, nlmB2, decay, outW, outB, startTrace, startPost);
    }

    @Override
    public int outputSize(int inputSize) {
        return outputs;
    }

    /** @return the internal ticks */
    public int tickCount() {
        return ticks;
    }
}
