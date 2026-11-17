package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Set;
import java.util.function.Function;

/**
 * The ULMFiT fine-tuning recipe as an {@link Optimizer}:
 *
 * <ul>
 *   <li>discriminative learning rates: layer group {@code l} (counted from the top, 0 being the
 *       top) trains at {@code eta / decay^l} ({@code decay = 2.6} in the paper), built on core
 *       {@link Optimizer#groups};</li>
 *   <li>slanted triangular learning rates: a linear warm-up over the first {@code cutFraction}
 *       of the steps to {@code eta}, then a linear decay to {@code eta / ratio};</li>
 *   <li>gradual unfreezing: only the top group trains at first, and one more group below is
 *       unfrozen every {@code unfreezeEvery} steps.</li>
 * </ul>
 *
 * {@link #step} trains only the parameters of unfrozen groups among those it is given (the
 * others keep their gradients for the caller to clear).
 *
 * <p>Paper: Howard and Ruder, "Universal Language Model Fine-tuning for Text Classification"
 * (ACL 2018), <a href="https://arxiv.org/abs/1801.06146">arXiv:1801.06146</a>.
 */
public final class DiscriminativeFineTuning implements Optimizer {

    private final List<Set<Var>> groups = new ArrayList<>();
    private final Optimizer inner;
    private final float topRate;
    private final int totalSteps;
    private final float cutFraction;
    private final float ratio;
    private final int unfreezeEvery;
    private float scale = 1f;
    private int t;

    /**
     * Creates the recipe.
     *
     * @param layers parameter groups from the bottom (embeddings) to the top (head)
     * @param base the optimizer of each group, given its learning rate (for example {@code Optimizer::adam})
     * @param topRate the peak learning rate of the top group
     * @param decay the rate divisor per group below the top (2.6 in the paper; 1 for none)
     * @param totalSteps the planned steps, for the schedule
     * @param cutFraction the warm-up fraction (0.1 in the paper; 0 for a plain linear decay)
     * @param ratio the peak-to-floor ratio of the schedule (32 in the paper; 1 for a constant rate)
     * @param unfreezeEvery steps between unfreezing groups; 0 trains every group from the start
     */
    public DiscriminativeFineTuning(List<List<Var>> layers, Function<Float, Optimizer> base, float topRate, float decay,
            int totalSteps, float cutFraction, float ratio, int unfreezeEvery) {
        if (layers.isEmpty()) {
            throw new IllegalArgumentException("need at least one layer group");
        }
        List<Optimizer.Group> g = new ArrayList<>();
        for (int i = 0; i < layers.size(); i++) {
            int fromTop = layers.size() - 1 - i;
            Set<Var> set = Collections.newSetFromMap(new IdentityHashMap<>());
            set.addAll(layers.get(i));
            groups.add(set);
            g.add(Optimizer.Group.of(layers.get(i), base.apply(topRate / (float) Math.pow(decay, fromTop))));
        }
        this.inner = Optimizer.groups(base.apply(topRate), g.toArray(Optimizer.Group[]::new));
        this.topRate = topRate;
        this.totalSteps = totalSteps;
        this.cutFraction = cutFraction;
        this.ratio = ratio;
        this.unfreezeEvery = unfreezeEvery;
    }

    /**
     * The slanted triangular multiplier of the peak rate at a step.
     *
     * @param step the step, from 0
     * @return the multiplier in {@code [1 / ratio, 1]}
     */
    public float schedule(int step) {
        int cut = (int) Math.floor(totalSteps * cutFraction);
        float p;
        if (step < cut) {
            p = (float) step / cut;
        } else {
            p = 1 - (float) (step - cut) / Math.max(1, totalSteps - cut);
        }
        p = Math.max(0, Math.min(1, p));
        return (1 + p * (ratio - 1)) / ratio;
    }

    /**
     * The number of groups (from the top) that train at a step.
     *
     * @param step the step
     * @return the unfrozen group count
     */
    public int unfrozen(int step) {
        return unfreezeEvery <= 0 ? groups.size() : Math.min(groups.size(), 1 + step / unfreezeEvery);
    }

    @Override
    public void step(List<Var> parameters) {
        int open = unfrozen(t);
        List<Var> train = new ArrayList<>();
        for (Var p : parameters) {
            for (int i = groups.size() - open; i < groups.size(); i++) {
                if (groups.get(i).contains(p)) {
                    train.add(p);
                    break;
                }
            }
        }
        inner.learningRate(topRate * scale * schedule(t));
        inner.step(train);
        t++;
    }

    @Override
    public float learningRate() {
        return topRate * scale;
    }

    @Override
    public void learningRate(float learningRate) {
        scale = learningRate / topRate;
    }

    /** @return the steps taken */
    public int steps() {
        return t;
    }

    @Override
    public String toString() {
        return "DiscriminativeFineTuning(" + groups.size() + " groups, peak " + topRate + ")";
    }
}
