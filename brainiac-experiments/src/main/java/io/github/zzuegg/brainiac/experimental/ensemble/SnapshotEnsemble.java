package io.github.zzuegg.brainiac.experimental.ensemble;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.EpochResult;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.TrainingListener;

import java.util.ArrayList;
import java.util.List;

/**
 * Snapshot ensembles (Huang et al., ICLR 2017,
 * <a href="https://arxiv.org/abs/1704.00109">arXiv:1704.00109</a>): one training run with a
 * cyclic cosine learning rate, {@code lr(t) = lr_max / 2 · (cos(π (t mod c) / c) + 1)}, that
 * keeps the weights at the end of every cycle, where the rate has annealed to zero, so one run
 * yields one member per cycle, each from a different local minimum.
 *
 * <p>A drop-in {@link TrainingListener}: register it with any training loop (or call
 * {@link #onBatch} after each optimizer step); it sets the optimizer's learning rate for the next
 * step and captures the snapshots. Predict with {@link EnsembleAudit#ofWeights} and average
 * uniformly or with {@link #validationWeights}.
 */
public final class SnapshotEnsemble implements TrainingListener {

    private final Optimizer optimizer;
    private final List<Var> parameters;
    private final int cycle;
    private final float maxRate;
    private final List<List<Tensor>> snapshots = new ArrayList<>();
    private long steps;

    /**
     * Creates the schedule and sets the optimizer to the first step's rate.
     *
     * @param optimizer the optimizer whose rate is scheduled
     * @param parameters the parameters to snapshot
     * @param cycleSteps steps per cycle
     * @param maxRate the rate at the start of every cycle
     */
    public SnapshotEnsemble(Optimizer optimizer, List<Var> parameters, int cycleSteps, float maxRate) {
        if (cycleSteps < 1) {
            throw new IllegalArgumentException("cycle must be positive: " + cycleSteps);
        }
        this.optimizer = optimizer;
        this.parameters = List.copyOf(parameters);
        this.cycle = cycleSteps;
        this.maxRate = maxRate;
        optimizer.learningRate(maxRate);
    }

    /**
     * The cyclic cosine rate.
     *
     * @param step the step, from 0
     * @param cycleSteps steps per cycle
     * @param maxRate the peak rate
     * @return the rate of that step
     */
    public static float cosineRate(long step, int cycleSteps, float maxRate) {
        double t = (step % cycleSteps) / (double) cycleSteps;
        return (float) (maxRate / 2 * (Math.cos(Math.PI * t) + 1));
    }

    @Override
    public void onBatch(int epoch, long step, float loss) {
        steps++;
        if (steps % cycle == 0) {
            snapshots.add(copy(parameters));
        }
        optimizer.learningRate(Math.max(1e-6f, cosineRate(steps, cycle, maxRate)));
    }

    @Override
    public void onEpoch(EpochResult result) {
    }

    /** @return the snapshots taken so far, oldest first */
    public List<List<Tensor>> snapshots() {
        return List.copyOf(snapshots);
    }

    static List<Tensor> copy(List<Var> parameters) {
        List<Tensor> out = new ArrayList<>();
        for (Var p : parameters) {
            out.add(p.value().copy());
        }
        return out;
    }

    /**
     * Simplex weights of the members that minimise the validation log loss of their mixture
     * (see {@link SuperLearner#simplexLogLoss}).
     *
     * @param members validation probabilities {@code [member][sample][class]}
     * @param labels validation labels
     * @return the weights
     */
    public static double[] validationWeights(double[][][] members, int[] labels) {
        return SuperLearner.simplexLogLoss(members, labels, 300);
    }

    /**
     * A weighted mean prediction.
     *
     * @param members {@code [member][sample][class]}
     * @param weights member weights summing to one
     * @return {@code [sample][class]}
     */
    public static double[][] weightedMean(double[][][] members, double[] weights) {
        int n = members[0].length;
        int k = members[0][0].length;
        double[][] out = new double[n][k];
        for (int m = 0; m < members.length; m++) {
            for (int i = 0; i < n; i++) {
                for (int c = 0; c < k; c++) {
                    out[i][c] += weights[m] * members[m][i][c];
                }
            }
        }
        return out;
    }
}
