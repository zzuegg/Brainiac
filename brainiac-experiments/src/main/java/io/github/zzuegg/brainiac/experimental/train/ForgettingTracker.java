package io.github.zzuegg.brainiac.experimental.train;

import java.util.ArrayList;
import java.util.List;

/**
 * Measures catastrophic forgetting in continual learning: the accuracy matrix
 * {@code R[i][j]}, the accuracy on task {@code j} after training on tasks {@code 0..i}
 * (Lopez-Paz and Ranzato 2017, <a href="https://arxiv.org/abs/1706.08840">arXiv:1706.08840</a>;
 * Chaudhry et al. 2018, <a href="https://arxiv.org/abs/1801.10112">arXiv:1801.10112</a>),
 * and the standard summaries: average accuracy, backward transfer and average forgetting.
 */
public final class ForgettingTracker {

    private final List<double[]> rows = new ArrayList<>();

    /** Creates an empty tracker. */
    public ForgettingTracker() {
    }

    /**
     * Records the accuracies on all tasks seen so far after finishing one more.
     *
     * @param accuracies the accuracy on task {@code 0..i}, where {@code i} is the task just
     *     trained
     */
    public void record(double... accuracies) {
        if (accuracies.length != rows.size() + 1) {
            throw new IllegalArgumentException("after task " + rows.size() + " give " + (rows.size() + 1)
                    + " accuracies, got " + accuracies.length);
        }
        rows.add(accuracies.clone());
    }

    /** @return the number of tasks recorded */
    public int tasks() {
        return rows.size();
    }

    /**
     * @param afterTask the task just trained
     * @param task the evaluated task, at most {@code afterTask}
     * @return {@code R[afterTask][task]}
     */
    public double accuracy(int afterTask, int task) {
        return rows.get(afterTask)[task];
    }

    /** @return the mean accuracy over all tasks after the last one */
    public double averageAccuracy() {
        double[] last = rows.getLast();
        double s = 0;
        for (double a : last) {
            s += a;
        }
        return s / last.length;
    }

    /**
     * @return the mean of {@code R[T][j] - R[j][j]} over earlier tasks: negative when
     *     learning later tasks hurt earlier ones
     */
    public double backwardTransfer() {
        int t = rows.size() - 1;
        if (t == 0) {
            return 0;
        }
        double s = 0;
        for (int j = 0; j < t; j++) {
            s += rows.get(t)[j] - rows.get(j)[j];
        }
        return s / t;
    }

    /** @return the mean over earlier tasks of the drop from their best accuracy to the final one */
    public double averageForgetting() {
        int t = rows.size() - 1;
        if (t == 0) {
            return 0;
        }
        double s = 0;
        for (int j = 0; j < t; j++) {
            double best = 0;
            for (int i = j; i < t; i++) {
                best = Math.max(best, rows.get(i)[j]);
            }
            s += best - rows.get(t)[j];
        }
        return s / t;
    }

    @Override
    public String toString() {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < rows.size(); i++) {
            b.append("after task ").append(i).append(':');
            for (double a : rows.get(i)) {
                b.append(String.format(" %.3f", a));
            }
            b.append('\n');
        }
        return b.append(String.format("average %.3f, forgetting %.3f, backward transfer %.3f", averageAccuracy(),
                averageForgetting(), backwardTransfer())).toString();
    }
}
