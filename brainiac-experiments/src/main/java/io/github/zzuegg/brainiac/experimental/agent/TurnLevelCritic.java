package io.github.zzuegg.brainiac.experimental.agent;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The turn-level critic of ArCHer (Zhou et al., ICML 2024,
 * <a href="https://arxiv.org/abs/2402.19446">arXiv:2402.19446</a>): a Q function over whole
 * utterances, {@code Q(context, utterance)}, trained off-policy by temporal-difference learning
 * across turns, {@code Q(s, a) <- r + gamma Q'(s', a')} with {@code a'} drawn from the current
 * policy and {@code Q'} a slowly tracking target network. Its advantages
 * {@code Q(s, a) - V(s)} drive the token-level policy within a turn
 * ({@link HierarchicalAgentTrainer}).
 *
 * <p>The network is a two-layer perceptron over one-hot encodings of the context's last
 * {@code window} tokens and the utterance's first {@code utteranceTokens} tokens.
 *
 * <p>{@link #doubleQ} turns on clipped double Q-learning (Fujimoto et al. 2018,
 * <a href="https://arxiv.org/abs/1802.09477">arXiv:1802.09477</a>), which the ArCHer paper uses
 * against over-estimation: two independently initialized Q networks with their own target
 * networks, both regressed onto {@code r + gamma min(Q1'(s', a'), Q2'(s', a'))}, and the
 * minimum of the two as the critic's estimate. {@link #targetRate} changes the Polyak rate of
 * the targets. Both are off (and unchanged) by default.
 */
public final class TurnLevelCritic {

    /**
     * One turn.
     *
     * @param context the conversation before the utterance
     * @param utterance what the agent said
     * @param reward the reward after it
     * @param next the conversation after the user's reply ({@code null} when done)
     * @param done whether the conversation ended
     */
    public record Transition(int[] context, int[] utterance, float reward, int[] next, boolean done) {
    }

    private final int vocab;
    private final int window;
    private final int utteranceTokens;
    private final float gamma;
    private float tau;
    private final Var w1;
    private final Var w2;
    private final float[] target1;
    private final float[] target2;
    private final int hidden;
    private final long seed;
    private Var v1;
    private Var v2;
    private float[] targetV1;
    private float[] targetV2;

    private TurnLevelCritic(int vocab, int window, int utteranceTokens, int hidden, float gamma, float tau,
            long seed) {
        this.vocab = vocab;
        this.window = window;
        this.utteranceTokens = utteranceTokens;
        this.hidden = hidden;
        this.gamma = gamma;
        this.tau = tau;
        this.seed = seed;
        SplittableRandom r = new SplittableRandom(seed);
        int in = inputs();
        this.w1 = Var.param(Tensor.of(Shape.of(in, hidden), gaussian(r, in * hidden, (float) Math.sqrt(2.0 / in))));
        this.w2 = Var.param(Tensor.of(Shape.of(hidden, 1), gaussian(r, hidden, (float) Math.sqrt(1.0 / hidden))));
        this.target1 = w1.value().toArray();
        this.target2 = w2.value().toArray();
    }

    /**
     * @param vocab the vocabulary size
     * @param window context tokens read
     * @param utteranceTokens utterance tokens read
     * @param hidden hidden units
     * @param gamma the discount
     * @param tau the target network's tracking rate per update
     * @param seed the seed
     * @return the critic
     */
    public static TurnLevelCritic of(int vocab, int window, int utteranceTokens, int hidden, float gamma, float tau,
            long seed) {
        return new TurnLevelCritic(vocab, window, utteranceTokens, hidden, gamma, tau, seed);
    }

    /**
     * Turns on clipped double Q-learning: a second Q network (seeded independently) and its
     * target; the TD target and {@link #q} use the minimum of the two.
     *
     * @return this critic
     */
    public TurnLevelCritic doubleQ() {
        if (v1 == null) {
            SplittableRandom r = new SplittableRandom(seed ^ 0x9E3779B97F4A7C15L);
            int in = inputs();
            v1 = Var.param(Tensor.of(Shape.of(in, hidden), gaussian(r, in * hidden, (float) Math.sqrt(2.0 / in))));
            v2 = Var.param(Tensor.of(Shape.of(hidden, 1), gaussian(r, hidden, (float) Math.sqrt(1.0 / hidden))));
            targetV1 = v1.value().toArray();
            targetV2 = v2.value().toArray();
        }
        return this;
    }

    /** @return whether clipped double Q-learning is on */
    public boolean isDoubleQ() {
        return v1 != null;
    }

    /**
     * @param rate the target networks' Polyak tracking rate per update (ArCHer: about 0.01)
     * @return this critic
     */
    public TurnLevelCritic targetRate(float rate) {
        this.tau = rate;
        return this;
    }

    /** @return the discount */
    public float gamma() {
        return gamma;
    }

    private int inputs() {
        return (window + utteranceTokens) * vocab + 1;
    }

    private static float[] gaussian(SplittableRandom r, int n, float scale) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) {
            double u = Math.max(1e-12, r.nextDouble());
            a[i] = (float) (Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r.nextDouble()) * scale);
        }
        return a;
    }

    private Tensor encode(List<int[]> contexts, List<int[]> utterances) {
        int in = inputs();
        float[] x = new float[contexts.size() * in];
        for (int i = 0; i < contexts.size(); i++) {
            int[] c = contexts.get(i);
            int n = Math.min(window, c.length);
            for (int k = 0; k < n; k++) {
                x[i * in + (window - n + k) * vocab + c[c.length - n + k]] = 1;
            }
            int[] u = utterances.get(i);
            for (int k = 0; k < Math.min(utteranceTokens, u.length); k++) {
                x[i * in + (window + k) * vocab + u[k]] = 1;
            }
            x[i * in + in - 1] = 1;
        }
        return Tensor.of(Shape.of(contexts.size(), in), x);
    }

    private Var forward(Tensor x, Var a, Var b) {
        return Var.of(x).matmul(a).relu().matmul(b);
    }

    /**
     * @param contexts the conversations so far
     * @param utterances a candidate utterance for each
     * @return their Q values
     */
    public float[] q(List<int[]> contexts, List<int[]> utterances) {
        if (contexts.isEmpty()) {
            return new float[0];
        }
        Tensor x = encode(contexts, utterances);
        float[] q = Var.noGrad(() -> forward(x, w1, w2)).value().toArray();
        if (v1 != null) {
            float[] q2 = Var.noGrad(() -> forward(x, v1, v2)).value().toArray();
            for (int i = 0; i < q.length; i++) {
                q[i] = Math.min(q[i], q2[i]);
            }
        }
        return q;
    }

    private float[] targetQ(List<int[]> contexts, List<int[]> utterances) {
        if (contexts.isEmpty()) {
            return new float[0];
        }
        Tensor x = encode(contexts, utterances);
        Var a = Var.of(Tensor.of(Shape.of(inputs(), hidden), target1));
        Var b = Var.of(Tensor.of(Shape.of(hidden, 1), target2));
        float[] q = Var.noGrad(() -> forward(x, a, b)).value().toArray();
        if (v1 != null) {
            Var c = Var.of(Tensor.of(Shape.of(inputs(), hidden), targetV1));
            Var d = Var.of(Tensor.of(Shape.of(hidden, 1), targetV2));
            float[] q2 = Var.noGrad(() -> forward(x, c, d)).value().toArray();
            for (int i = 0; i < q.length; i++) {
                q[i] = Math.min(q[i], q2[i]);
            }
        }
        return q;
    }

    /**
     * One TD update on a minibatch.
     *
     * @param batch the transitions
     * @param nextUtterances for each transition, an utterance the current policy says in
     *     {@code next} ({@code null} for finished ones)
     * @param optimizer the optimizer
     * @return the mean squared TD error before the update
     */
    public double update(List<Transition> batch, List<int[]> nextUtterances, Optimizer optimizer) {
        List<int[]> ns = new ArrayList<>();
        List<int[]> na = new ArrayList<>();
        for (int i = 0; i < batch.size(); i++) {
            if (!batch.get(i).done()) {
                ns.add(batch.get(i).next());
                na.add(nextUtterances.get(i));
            }
        }
        float[] boot = targetQ(ns, na);
        float[] y = new float[batch.size()];
        List<int[]> cs = new ArrayList<>();
        List<int[]> us = new ArrayList<>();
        int k = 0;
        for (int i = 0; i < batch.size(); i++) {
            Transition t = batch.get(i);
            y[i] = t.reward() + (t.done() ? 0 : gamma * boot[k++]);
            cs.add(t.context());
            us.add(t.utterance());
        }
        Tensor x = encode(cs, us);
        Tensor target = Tensor.of(Shape.of(y.length, 1), y);
        Var loss = forward(x, w1, w2).sub(target).square().mean();
        double value = loss.value().item();
        if (v1 != null) {
            loss = loss.add(forward(x, v1, v2).sub(target).square().mean());
        }
        loss.backward();
        optimizer.step(parameters());
        track(target1, w1.value().toArray());
        track(target2, w2.value().toArray());
        if (v1 != null) {
            track(targetV1, v1.value().toArray());
            track(targetV2, v2.value().toArray());
        }
        return value;
    }

    private void track(float[] target, float[] online) {
        for (int i = 0; i < target.length; i++) {
            target[i] += tau * (online[i] - target[i]);
        }
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        return v1 == null ? List.of(w1, w2) : List.of(w1, w2, v1, v2);
    }

    @Override
    public String toString() {
        return "TurnLevelCritic(window " + window + ", hidden " + hidden + ", gamma " + gamma + ")";
    }
}
