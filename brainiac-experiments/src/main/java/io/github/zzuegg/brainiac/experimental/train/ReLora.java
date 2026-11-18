package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * ReLoRA (Lialin et al. 2023, <a href="https://arxiv.org/abs/2307.05695">arXiv:2307.05695</a>):
 * high-rank training through low-rank updates. Each chosen {@code [in, out]} weight is frozen
 * and trained only through a LoRA adapter {@code W + s A B} (as {@code LoraAdapterDense} does
 * for core {@code Dense}); every {@code restartEvery} steps the adapter is merged into the frozen
 * weight, re-initialized ({@code A} random, {@code B} zero), the optimizer's state for it is
 * reset, and the learning rate restarts with a short warm-up (the "jagged" schedule). The sum of
 * many rank-{@code r} updates is high-rank, while trainable parameters and optimizer memory stay
 * at LoRA's level. With {@code restartEvery = 0} it is plain LoRA.
 *
 * <p>Use {@link #weight(int)} in place of the original weight in the forward pass, train
 * {@link #adapters()} (plus any other parameters), call {@link #afterStep} after every optimizer
 * step and scale the learning rate by {@link #learningRateFactor}.
 */
public final class ReLora {

    private final List<Var> bases;
    private final List<Tensor> initial = new ArrayList<>();
    private final List<Var> a = new ArrayList<>();
    private final List<Var> b = new ArrayList<>();
    private final int rank;
    private final float scale;
    private final int restartEvery;
    private final int warmup;
    private final SplittableRandom random;
    private int lastRestart;
    private int restarts;

    /**
     * @param weights the {@code [in, out]} weights to train through adapters; their values are
     *     the frozen bases and receive every merge
     * @param rank the adapter rank {@code r}
     * @param alpha the LoRA scale numerator, the update is {@code alpha / r * A B}
     * @param restartEvery steps between merges, 0 for plain LoRA
     * @param warmup steps of the learning-rate warm-up after each restart
     * @param seed the seed of the adapters' initialization
     */
    public ReLora(List<Var> weights, int rank, float alpha, int restartEvery, int warmup, long seed) {
        this.bases = List.copyOf(weights);
        this.rank = rank;
        this.scale = alpha / rank;
        this.restartEvery = restartEvery;
        this.warmup = warmup;
        this.random = new SplittableRandom(seed);
        for (Var w : bases) {
            initial.add(w.value());
            a.add(Var.param(Tensor.zeros(Shape.of(w.shape().dim(0), rank))));
            b.add(Var.param(Tensor.zeros(Shape.of(rank, w.shape().dim(1)))));
        }
        for (int i = 0; i < bases.size(); i++) {
            reinitialize(i);
        }
    }

    private void reinitialize(int i) {
        int in = bases.get(i).shape().dim(0);
        a.get(i).assign(Tensor.normal(Shape.of(in, rank), 0, (float) Math.sqrt(1.0 / in), random));
        b.get(i).assign(Tensor.zeros(b.get(i).shape()));
    }

    /**
     * The effective weight {@code W + s A B}, differentiable in the adapter only.
     *
     * @param i the weight's index
     * @return the weight to use in the forward pass
     */
    public Var weight(int i) {
        return Var.of(bases.get(i).value()).add(a.get(i).matmul(b.get(i)).mul(scale));
    }

    /** @return every adapter matrix, for the optimizer */
    public List<Var> adapters() {
        List<Var> out = new ArrayList<>();
        for (int i = 0; i < bases.size(); i++) {
            out.add(a.get(i));
            out.add(b.get(i));
        }
        return out;
    }

    /** @return the number of trainable adapter values */
    public long trainableParameters() {
        long n = 0;
        for (Var v : adapters()) {
            n += v.shape().size();
        }
        return n;
    }

    /** @return merges so far */
    public int restarts() {
        return restarts;
    }

    /**
     * The jagged schedule's factor: a linear warm-up over {@code warmup} steps after the start
     * and after each restart.
     *
     * @param step the step about to be taken
     * @return the factor in {@code (0, 1]}
     */
    public float learningRateFactor(int step) {
        if (warmup <= 0) {
            return 1;
        }
        return Math.min(1f, (step - lastRestart + 1f) / warmup);
    }

    /**
     * Merges and restarts when due; call after every optimizer step.
     *
     * @param step the step just taken, from 0
     * @param optimizer the optimizer, whose adapter state is reset on a restart
     * @return whether a restart happened
     */
    public boolean afterStep(int step, Optimizer optimizer) {
        if (restartEvery <= 0 || (step + 1) % restartEvery != 0) {
            return false;
        }
        for (int i = 0; i < bases.size(); i++) {
            Var w = bases.get(i);
            w.assign(w.value().add(a.get(i).value().matmul(b.get(i).value()).mul(scale)));
            reinitialize(i);
            optimizer.forget(a.get(i));
            optimizer.forget(b.get(i));
        }
        lastRestart = step + 1;
        restarts++;
        return true;
    }

    /**
     * The total update of one weight since construction, merged part and live adapter.
     *
     * @param i the weight's index
     * @return {@code W_now + s A B - W_0}
     */
    public Tensor totalUpdate(int i) {
        return bases.get(i).value().add(a.get(i).value().matmul(b.get(i).value()).mul(scale)).sub(initial.get(i));
    }

    /**
     * The effective rank of a matrix (Roy and Vetterli 2007): {@code exp} of the entropy of its
     * normalized singular values.
     *
     * @param m the matrix
     * @return a number between 1 and {@code min(rows, cols)}
     */
    public static double effectiveRank(Tensor m) {
        float[] s = Linalg.svd(m)[1].toArray();
        double sum = 0;
        for (float v : s) {
            sum += Math.abs(v);
        }
        if (sum == 0) {
            return 0;
        }
        double h = 0;
        for (float v : s) {
            double p = Math.abs(v) / sum;
            if (p > 0) {
                h -= p * Math.log(p);
            }
        }
        return Math.exp(h);
    }

    @Override
    public String toString() {
        return (restartEvery > 0 ? "ReLoRA" : "LoRA") + "(r=" + rank + ")";
    }
}
