package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.EpochResult;
import io.github.zzuegg.brainiac.train.LearningRateSchedule;
import io.github.zzuegg.brainiac.train.TrainingListener;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

/**
 * WSM, warmup-stable-merge (Tian et al. 2025, "WSM: Decay-Free Learning Rate Schedule via
 * Checkpoint Merging for LLM Pre-training", <a href="https://arxiv.org/abs/2507.17634">arXiv:2507.17634</a>,
 * ICLR 2026 oral): train at a constant learning rate after the warm-up and replace the
 * decay phase by merging the last {@code n} checkpoints. If checkpoint {@code k} comes
 * {@code k} intervals into the window, a weighted average {@code sum c_k theta_k} applies
 * the gradients of interval {@code k} with the effective rate {@code lr * sum_{j >= k} c_j}:
 * any decay shape {@code f} (from 1 to 0 over the window) is emulated by the weights
 * {@code c_k = f((k-1)/n) - f(k/n)}. Uniform weights are linear decay; {@link Shape#COSINE}
 * and {@link Shape#ONE_MINUS_SQRT} give the cosine and "1 - sqrt" cool-downs. The run
 * itself never decays, so it can continue (or branch) at any point, and the merge is free.
 *
 * <p>As a {@link TrainingListener} it takes a checkpoint every {@code every} steps;
 * {@link #swap()} exchanges the merged and current weights. {@link #schedule(int)} is the
 * matching warm-up-then-constant learning-rate schedule.
 */
public final class WarmupStableMerge implements TrainingListener {

    /** The decay shape the merge weights emulate. */
    public enum Shape {
        /** Linear decay: a uniform average (LAWA over the window). */
        LINEAR,
        /** Cosine decay. */
        COSINE,
        /** The "1 - sqrt" cool-down of WSD. */
        ONE_MINUS_SQRT;

        /**
         * @param s the position in the window, 0 to 1
         * @return the emulated learning-rate factor, 1 at 0 and 0 at 1
         */
        public double factor(double s) {
            return switch (this) {
                case LINEAR -> 1 - s;
                case COSINE -> 0.5 * (1 + Math.cos(Math.PI * s));
                case ONE_MINUS_SQRT -> 1 - Math.sqrt(s);
            };
        }
    }

    private final List<Var> parameters;
    private final int window;
    private final int every;
    private final Shape shape;
    private final ArrayDeque<List<Tensor>> checkpoints = new ArrayDeque<>();
    private List<Tensor> swapped;

    /**
     * @param parameters the parameters to merge
     * @param window checkpoints merged, {@code n}
     * @param every steps between checkpoints
     * @param shape the emulated decay shape
     */
    public WarmupStableMerge(List<Var> parameters, int window, int every, Shape shape) {
        if (window < 1 || every < 1) {
            throw new IllegalArgumentException("window and every must be positive");
        }
        this.parameters = List.copyOf(parameters);
        this.window = window;
        this.every = every;
        this.shape = shape;
    }

    /**
     * A linear warm-up over {@code warmup} steps, then the base rate for good. Steps count
     * from 1, as in the core schedules.
     *
     * @param warmup warm-up steps
     * @return the schedule
     */
    public static LearningRateSchedule schedule(int warmup) {
        return (step, base) -> step <= warmup ? base * step / (float) Math.max(1, warmup) : base;
    }

    /**
     * The merge weights for {@code n} checkpoints, oldest first.
     *
     * @param shape the decay shape
     * @param n the checkpoints
     * @return the weights, summing to 1
     */
    public static double[] weights(Shape shape, int n) {
        double[] c = new double[n];
        for (int k = 1; k <= n; k++) {
            c[k - 1] = shape.factor((k - 1) / (double) n) - shape.factor(k / (double) n);
        }
        return c;
    }

    /** Takes a checkpoint, dropping the oldest beyond the window. */
    public void checkpoint() {
        List<Tensor> snap = new ArrayList<>(parameters.size());
        for (Var p : parameters) {
            snap.add(p.value());
        }
        checkpoints.addLast(snap);
        if (checkpoints.size() > window) {
            checkpoints.removeFirst();
        }
    }

    /** @return the checkpoints in the window */
    public int size() {
        return checkpoints.size();
    }

    /**
     * The merged weights of the checkpoints in the window.
     *
     * @return one tensor per parameter
     */
    public List<Tensor> merged() {
        if (checkpoints.isEmpty()) {
            throw new IllegalStateException("no checkpoint taken yet");
        }
        double[] c = weights(shape, checkpoints.size());
        List<Tensor> out = null;
        int k = 0;
        for (List<Tensor> snap : checkpoints) {
            float ck = (float) c[k++];
            if (out == null) {
                out = new ArrayList<>();
                for (Tensor t : snap) {
                    out.add(t.mul(ck));
                }
            } else {
                for (int i = 0; i < out.size(); i++) {
                    out.set(i, out.get(i).add(snap.get(i).mul(ck)));
                }
            }
        }
        Tensor.evaluate(out);
        return out;
    }

    /** Puts the merged weights in the parameters, or the training weights back. */
    public void swap() {
        if (swapped == null) {
            List<Tensor> m = merged();
            swapped = new ArrayList<>();
            for (int i = 0; i < parameters.size(); i++) {
                swapped.add(parameters.get(i).value());
                parameters.get(i).assign(m.get(i));
            }
        } else {
            for (int i = 0; i < parameters.size(); i++) {
                parameters.get(i).assign(swapped.get(i));
            }
            swapped = null;
        }
    }

    @Override
    public void onBatch(int epoch, long step, float loss) {
        if (step % every == 0) {
            checkpoint();
        }
    }

    @Override
    public void onEpoch(EpochResult result) {
    }
}
