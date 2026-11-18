package io.github.zzuegg.brainiac.experimental.planning;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Conv;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.Padding;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A value iteration network (Tamar et al., NeurIPS 2016,
 * <a href="https://arxiv.org/abs/1602.02867">arXiv:1602.02867</a>): a CNN maps the
 * observation (obstacles and goal channels) to a reward map {@code R}; {@code K} steps of value
 * iteration then run as a recurrent convolution, {@code Q = conv([R, V])} with one output
 * channel per action and {@code V = max_a Q}, sharing the convolution across steps; the
 * policy reads the {@code Q} vector at the agent's cell (soft attention by a one-hot mask)
 * and maps it linearly to action logits. Trained by imitation of optimal actions, it learns a
 * planning computation that transfers to unseen maps, and to larger maps if {@code K} grows.
 *
 * <p>Input {@code [batch, 2, H, W]}; any {@code H, W} works with the same weights.
 */
public final class ValueIterationNetwork {

    private final Model reward;
    private final Model qConv;
    private final Var head;
    private final int actions;
    private int iterations;

    /**
     * Creates the network.
     *
     * @param hidden the reward encoder's hidden channels
     * @param qChannels the number of Q channels
     * @param actions the number of actions
     * @param iterations the value iteration steps {@code K}
     * @param side the map side used to declare input shapes (weights do not depend on it)
     * @param seed the seed
     */
    public ValueIterationNetwork(int hidden, int qChannels, int actions, int iterations, int side, long seed) {
        this.reward = Model.sequential().input(2, side, side).seed(seed)
            .add(Conv.of(hidden, 3, 3).padding(Padding.SAME).activation(Activation.RELU))
            .add(Conv.of(1, 1, 1).padding(Padding.SAME)).build();
        this.qConv = Model.sequential().input(2, side, side).seed(seed + 1)
            .add(Conv.of(qChannels, 3, 3).padding(Padding.SAME).bias(false)).build();
        SplittableRandom r = new SplittableRandom(seed + 2);
        this.head = Var.param(Tensor.normal(Shape.of(qChannels, actions), 0, (float) Math.sqrt(1.0 / qChannels), r));
        this.actions = actions;
        this.iterations = iterations;
    }

    /** @param k the value iteration steps used from now on */
    public void iterations(int k) {
        this.iterations = k;
    }

    /** @return the value iteration steps */
    public int iterations() {
        return iterations;
    }

    /** @return every trainable parameter */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(reward.parameters());
        p.addAll(qConv.parameters());
        p.add(head);
        return p;
    }

    /**
     * The value map after {@code K} iterations.
     *
     * @param maps {@code [batch, 2, H, W]}
     * @return {@code {Q [batch, A, H, W], V [batch, 1, H, W]}}
     */
    public Var[] plan(Var maps) {
        Var r = reward.forward(maps);
        Var v = Var.of(Tensor.zeros(r.shape()));
        Var q = null;
        for (int k = 0; k < iterations; k++) {
            q = qConv.forward(Var.concat(1, r, v));
            v = q.max(1, true);
        }
        if (q == null) {
            q = qConv.forward(Var.concat(1, r, v));
        }
        return new Var[] {q, v};
    }

    /**
     * Action logits at the agents' cells.
     *
     * @param maps {@code [batch, 2, H, W]}: obstacles and a one-hot goal
     * @param cells per batch row the agent's cell {@code y * W + x}
     * @return {@code [batch, actions]}
     */
    public Var forward(Tensor maps, int[] cells) {
        Var q = plan(Var.of(maps))[0];
        int b = maps.shape().dim(0);
        int c = q.shape().dim(1);
        int hw = maps.shape().dim(2) * maps.shape().dim(3);
        float[] mask = new float[b * hw];
        for (int i = 0; i < b; i++) {
            mask[i * hw + cells[i]] = 1;
        }
        Var att = q.reshape(b, c, hw).mul(Tensor.of(Shape.of(b, 1, hw), mask)).sum(2, false);
        return att.matmul(head);
    }

    /**
     * Imitation training on (map, cell, action) samples by cross-entropy.
     *
     * @param maps per sample the map {@code [2 * H * W]}
     * @param cells per sample the agent cell
     * @param labels per sample the expert action
     * @param side the map side
     * @param steps gradient steps
     * @param batch samples per step
     * @param learningRate Adam's rate
     * @param seed the seed
     */
    public void fit(List<float[]> maps, int[] cells, int[] labels, int side, int steps, int batch, float learningRate,
            long seed) {
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom r = new SplittableRandom(seed);
        int plane = 2 * side * side;
        List<Var> params = parameters();
        for (int s = 0; s < steps; s++) {
            float[] x = new float[batch * plane];
            int[] c = new int[batch];
            float[] y = new float[batch];
            for (int i = 0; i < batch; i++) {
                int k = r.nextInt(labels.length);
                System.arraycopy(maps.get(k), 0, x, i * plane, plane);
                c[i] = cells[k];
                y[i] = labels[k];
            }
            Var logits = forward(Tensor.of(Shape.of(batch, 2, side, side), x), c);
            Functional.crossEntropy(logits, Tensor.of(Shape.of(batch), y)).backward();
            opt.step(params);
        }
    }

    /**
     * The greedy action.
     *
     * @param map the map {@code [2 * H * W]}
     * @param side the map side
     * @param cell the agent's cell
     * @return the action with the largest logit
     */
    public int act(float[] map, int side, int cell) {
        float[] l = Var.noGrad(() -> forward(Tensor.of(Shape.of(1, 2, side, side), map), new int[] {cell}))
            .value().toArray();
        int best = 0;
        for (int a = 1; a < actions; a++) {
            if (l[a] > l[best]) {
                best = a;
            }
        }
        return best;
    }
}
