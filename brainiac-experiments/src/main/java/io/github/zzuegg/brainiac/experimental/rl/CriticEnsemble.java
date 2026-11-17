package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * An ensemble of {@code Q(s, a)} critics with Polyak-averaged target copies, shared by the
 * high update-to-data actor-critics ({@link HighUtdSac}: REDQ and DroQ) and {@link Mpo}.
 * Targets take the minimum over a random subset of the target critics (REDQ's in-target
 * minimization; the whole ensemble of two is SAC's clipped double Q). Each critic can carry
 * dropout and layer normalization after every hidden linear layer (DroQ), which tames the
 * overestimation that many gradient steps per environment step otherwise cause.
 */
public final class CriticEnsemble {

    private final List<Model> critics = new ArrayList<>();
    private final List<Model> targets = new ArrayList<>();
    private final List<Var> parameters = new ArrayList<>();

    /**
     * @param inputs observation plus action values
     * @param size the number of critics
     * @param dropout the dropout rate after each hidden linear layer (0 for none)
     * @param layerNorm whether to normalize each hidden layer
     * @param seed the initialization seed
     * @param hidden the hidden layer sizes
     */
    public CriticEnsemble(int inputs, int size, float dropout, boolean layerNorm, long seed, int... hidden) {
        for (int i = 0; i < size; i++) {
            Model q = build(inputs, dropout, layerNorm, seed + 31L * i, hidden);
            Model t = build(inputs, dropout, layerNorm, seed + 31L * i, hidden);
            RlTensors.copy(q.parameters(), t.parameters());
            critics.add(q);
            targets.add(t);
            parameters.addAll(q.parameters());
        }
    }

    private static Model build(int inputs, float dropout, boolean layerNorm, long seed, int[] hidden) {
        Model.Builder b = Model.sequential().input(inputs);
        for (int h : hidden) {
            b.dense(h);
            if (dropout > 0) {
                b.dropout(dropout);
            }
            if (layerNorm) {
                b.layer(LayerNorm.of(h));
            }
            b.activation(Activation.RELU);
        }
        return b.dense(1).seed(seed).build();
    }

    /** @return the number of critics */
    public int size() {
        return critics.size();
    }

    /** @return the trainable parameters of all critics */
    public List<Var> parameters() {
        return parameters;
    }

    /**
     * The minimum of a random subset of target critics, without gradients.
     *
     * @param input {@code [batch, obs + action]}
     * @param subset how many critics to take the minimum over (at most {@link #size()})
     * @param random picks the subset
     * @return {@code [batch, 1]}
     */
    public Tensor targetMin(Tensor input, int subset, SplittableRandom random) {
        int n = targets.size();
        int[] idx = new int[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        int m = Math.min(subset, n);
        Tensor min = null;
        for (int i = 0; i < m; i++) {
            int j = i + random.nextInt(n - i);
            int t = idx[i];
            idx[i] = idx[j];
            idx[j] = t;
            Tensor q = targets.get(idx[i]).forward(Var.of(input), false).value();
            min = min == null ? q : min.minimum(q);
        }
        return min;
    }

    /**
     * The mean of all critics, with gradients (for the actor loss).
     *
     * @param input {@code [batch, obs + action]}
     * @param training whether dropout is active
     * @return {@code [batch, 1]}
     */
    public Var mean(Var input, boolean training) {
        Var sum = null;
        for (Model q : critics) {
            Var v = q.forward(input, training);
            sum = sum == null ? v : sum.add(v);
        }
        return sum.mul(1f / critics.size());
    }

    /**
     * The summed squared error of every critic against the same targets, with gradients
     * into the input (for models that train an encoder through the critics).
     *
     * @param input {@code [batch, obs + action]}
     * @param target {@code [batch, 1]}
     * @return the summed mean squared error
     */
    public Var loss(Var input, Tensor target) {
        Var loss = null;
        for (Model q : critics) {
            Var l = q.forward(input, true).sub(target).square().mean();
            loss = loss == null ? l : loss.add(l);
        }
        return loss;
    }

    /**
     * One regression step of every critic towards the same targets.
     *
     * @param input {@code [batch, obs + action]}
     * @param target {@code [batch, 1]}
     * @param optimizer the critics' optimizer
     * @return the summed mean squared error
     */
    public double update(Tensor input, Tensor target, Optimizer optimizer) {
        Var in = Var.of(input);
        Var loss = null;
        for (Model q : critics) {
            Var l = q.forward(in, true).sub(target).square().mean();
            loss = loss == null ? l : loss.add(l);
        }
        loss.backward();
        optimizer.step(parameters);
        return loss.value().item();
    }

    /**
     * Moves the target critics towards the critics.
     *
     * @param tau the Polyak factor
     */
    public void blend(float tau) {
        for (int i = 0; i < critics.size(); i++) {
            RlTensors.blend(critics.get(i).parameters(), targets.get(i).parameters(), tau);
        }
    }
}
