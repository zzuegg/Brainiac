package io.github.zzuegg.brainiac.experimental.ensemble;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.EpochResult;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.TrainingListener;

import java.util.ArrayList;
import java.util.List;

/**
 * Fast geometric ensembling (Garipov et al., NeurIPS 2018,
 * <a href="https://arxiv.org/abs/1802.10026">arXiv:1802.10026</a>): starting from a trained
 * model, a short triangular cyclic learning rate moves the weights along a low-loss path
 * (mode connectivity) and a member is collected at the middle of every cycle, where the rate
 * is lowest. With cycle position {@code t ∈ (0, 1]}:
 * {@code lr = (1 - 2t) lr1 + 2t lr2} for {@code t <= 1/2} and {@code (2 - 2t) lr2 + (2t - 1) lr1}
 * after, so the rate goes {@code lr1 -> lr2 -> lr1}.
 *
 * <p>A drop-in {@link TrainingListener}, used like {@link SnapshotEnsemble}.
 */
public final class FastGeometricEnsemble implements TrainingListener {

    private final Optimizer optimizer;
    private final List<Var> parameters;
    private final int cycle;
    private final float high;
    private final float low;
    private final List<List<Tensor>> members = new ArrayList<>();
    private long steps;

    /**
     * Creates the schedule and sets the first step's rate.
     *
     * @param optimizer the optimizer whose rate is scheduled
     * @param parameters the parameters to collect
     * @param cycleSteps steps per cycle, even
     * @param high the rate {@code lr1} at the cycle ends
     * @param low the rate {@code lr2} at the cycle middle
     */
    public FastGeometricEnsemble(Optimizer optimizer, List<Var> parameters, int cycleSteps, float high, float low) {
        if (cycleSteps < 2 || cycleSteps % 2 != 0) {
            throw new IllegalArgumentException("cycle must be even and at least 2: " + cycleSteps);
        }
        this.optimizer = optimizer;
        this.parameters = List.copyOf(parameters);
        this.cycle = cycleSteps;
        this.high = high;
        this.low = low;
        optimizer.learningRate(rate(1, cycleSteps, high, low));
    }

    /**
     * The triangular rate.
     *
     * @param step the step, from 1
     * @param cycleSteps steps per cycle
     * @param high {@code lr1}
     * @param low {@code lr2}
     * @return the rate of that step
     */
    public static float rate(long step, int cycleSteps, float high, float low) {
        double t = ((step - 1) % cycleSteps + 1) / (double) cycleSteps;
        double r = t <= 0.5 ? (1 - 2 * t) * high + 2 * t * low : (2 - 2 * t) * low + (2 * t - 1) * high;
        return (float) r;
    }

    @Override
    public void onBatch(int epoch, long step, float loss) {
        steps++;
        if (steps % cycle == cycle / 2) {
            members.add(SnapshotEnsemble.copy(parameters));
        }
        optimizer.learningRate(rate(steps + 1, cycle, high, low));
    }

    @Override
    public void onEpoch(EpochResult result) {
    }

    /** @return the collected members, oldest first */
    public List<List<Tensor>> members() {
        return List.copyOf(members);
    }
}
