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
    private final float tau;
    private final Var w1;
    private final Var w2;
    private final float[] target1;
    private final float[] target2;
    private final int hidden;

    private TurnLevelCritic(int vocab, int window, int utteranceTokens, int hidden, float gamma, float tau,
            long seed) {
        this.vocab = vocab;
        this.window = window;
        this.utteranceTokens = utteranceTokens;
        this.hidden = hidden;
        this.gamma = gamma;
        this.tau = tau;
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
        return Var.noGrad(() -> forward(encode(contexts, utterances), w1, w2)).value().toArray();
    }

    private float[] targetQ(List<int[]> contexts, List<int[]> utterances) {
        if (contexts.isEmpty()) {
            return new float[0];
        }
        Var a = Var.of(Tensor.of(Shape.of(inputs(), hidden), target1));
        Var b = Var.of(Tensor.of(Shape.of(hidden, 1), target2));
        return Var.noGrad(() -> forward(encode(contexts, utterances), a, b)).value().toArray();
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
        Var loss = forward(encode(cs, us), w1, w2).sub(Tensor.of(Shape.of(y.length, 1), y)).square().mean();
        double value = loss.value().item();
        loss.backward();
        optimizer.step(parameters());
        track(target1, w1.value().toArray());
        track(target2, w2.value().toArray());
        return value;
    }

    private void track(float[] target, float[] online) {
        for (int i = 0; i < target.length; i++) {
            target[i] += tau * (online[i] - target[i]);
        }
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        return List.of(w1, w2);
    }

    @Override
    public String toString() {
        return "TurnLevelCritic(window " + window + ", hidden " + hidden + ", gamma " + gamma + ")";
    }
}
