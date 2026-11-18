package io.github.zzuegg.brainiac.experimental.compress;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.WeightPruning;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Gradual magnitude pruning (Zhu and Gupta 2017, "To prune, or not to prune",
 * <a href="https://arxiv.org/abs/1710.01878">arXiv:1710.01878</a>): instead of pruning once
 * after training, the sparsity rises during training along the cubic schedule
 *
 * <pre>
 * s_t = s_f + (s_i - s_f) (1 - (t - t0) / (n dt))^3,   t0 &lt;= t &lt;= t0 + n dt
 * </pre>
 *
 * removing the smallest weights every {@code dt} steps while the survivors recover. The fast
 * early pruning takes the redundant weights while the network can still adapt, and the slow
 * tail leaves time to recover. It is the strong baseline later pruning methods are measured
 * against.
 *
 * <p>It is an {@link Optimizer} wrapper, so it drops in wherever the inner optimizer was used:
 * gradients of pruned weights are masked before the inner step, the masks are re-applied after
 * it, and at each pruning step the {@link Criterion} (by default {@link WeightPruning#magnitude},
 * any {@code WeightPruning} criterion fits) chooses which weights go at the scheduled sparsity.
 */
public final class GradualPruningSchedule implements Optimizer {

    /** Prunes one weight matrix to a given sparsity. */
    @FunctionalInterface
    public interface Criterion {
        /**
         * @param weight the weight {@code [in, out]}
         * @param sparsity the share to remove
         * @return the pruned weight; zeros mark removed entries
         */
        Tensor prune(Tensor weight, double sparsity);
    }

    private final Optimizer inner;
    private final List<Var> weights;
    private final Criterion criterion;
    private final double initial;
    private final double target;
    private final int start;
    private final int end;
    private final int frequency;
    private final Map<Var, Tensor> masks = new IdentityHashMap<>();
    private int step;

    /**
     * Creates the schedule.
     *
     * @param inner the optimizer that updates the weights
     * @param weights the weight matrices to prune
     * @param criterion which weights to remove, for example {@code WeightPruning::magnitude}
     * @param initialSparsity {@code s_i}, usually 0
     * @param finalSparsity {@code s_f}
     * @param start {@code t0}, the first pruning step
     * @param end {@code t0 + n dt}, the step at which the final sparsity is reached
     * @param frequency {@code dt}, the steps between pruning events
     */
    public GradualPruningSchedule(Optimizer inner, List<Var> weights, Criterion criterion, double initialSparsity,
            double finalSparsity, int start, int end, int frequency) {
        if (end < start || frequency < 1) {
            throw new IllegalArgumentException("need start <= end and frequency >= 1");
        }
        this.inner = inner;
        this.weights = List.copyOf(weights);
        this.criterion = criterion;
        this.initial = initialSparsity;
        this.target = finalSparsity;
        this.start = start;
        this.end = end;
        this.frequency = frequency;
    }

    /**
     * Magnitude pruning ({@link WeightPruning#magnitude}) from dense to {@code finalSparsity}.
     *
     * @param inner the optimizer
     * @param weights the weight matrices
     * @param finalSparsity the target sparsity
     * @param start the first pruning step
     * @param end the step reaching the target
     * @param frequency steps between pruning events
     * @return the schedule
     */
    public static GradualPruningSchedule magnitude(Optimizer inner, List<Var> weights, double finalSparsity,
            int start, int end, int frequency) {
        return new GradualPruningSchedule(inner, weights, WeightPruning::magnitude, 0, finalSparsity, start, end,
                frequency);
    }

    /**
     * The cubic schedule.
     *
     * @param initialSparsity {@code s_i}
     * @param finalSparsity {@code s_f}
     * @param start {@code t0}
     * @param end {@code t0 + n dt}
     * @param step {@code t}
     * @return {@code s_t}, constant before {@code start} and after {@code end}
     */
    public static double cubic(double initialSparsity, double finalSparsity, int start, int end, int step) {
        if (step <= start) {
            return step < start ? 0 : initialSparsity;
        }
        if (step >= end) {
            return finalSparsity;
        }
        double f = 1 - (step - start) / (double) (end - start);
        return finalSparsity + (initialSparsity - finalSparsity) * f * f * f;
    }

    /** @return the scheduled sparsity at the current step */
    public double scheduledSparsity() {
        return cubic(initial, target, start, end, step);
    }

    /** @return the optimizer steps taken */
    public int steps() {
        return step;
    }

    /**
     * @param w a weight
     * @return its current 0/1 mask, or null before it was first pruned
     */
    public Tensor mask(Var w) {
        return masks.get(w);
    }

    @Override
    public float learningRate() {
        return inner.learningRate();
    }

    @Override
    public void learningRate(float learningRate) {
        inner.learningRate(learningRate);
    }

    @Override
    public void step(List<Var> parameters) {
        for (Var w : weights) {
            Tensor m = masks.get(w);
            if (m != null && w.grad() != null) {
                w.setGrad(w.grad().mul(m));
            }
        }
        inner.step(parameters);
        step++;
        boolean prune = step >= start && step <= end && (step - start) % frequency == 0
                || step == end;
        double s = scheduledSparsity();
        for (Var w : weights) {
            if (prune && s > 0) {
                Tensor pruned = criterion.prune(w.value(), s);
                masks.put(w, pruned.abs().greater(Tensor.zeros(pruned.shape())));
                w.assign(pruned);
            } else if (masks.containsKey(w)) {
                w.assign(w.value().mul(masks.get(w)));
            }
        }
    }
}
