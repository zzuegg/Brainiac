package io.github.zzuegg.brainiac.experimental.shift;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * SWAD: stochastic weight averaging, densely (every step) and only over the stretch of
 * training where the validation loss is flat (Cha et al., NeurIPS 2021,
 * <a href="https://arxiv.org/abs/2102.08604">arXiv:2102.08604</a>). Plain SWA
 * ({@code experimental.train.StochasticWeightAveraging}) averages from a fixed start; SWAD's
 * overfit-aware window starts at the evaluation where the loss stopped improving for
 * {@code patience} evaluations and ends before the first run of {@code endPatience}
 * evaluations whose loss exceeds {@code tolerance} times the minimum, which finds flatter
 * minima that transfer better to unseen domains.
 *
 * <p>Call {@link #sample()} after every optimiser step and {@link #evaluate} at every
 * validation point; {@link #average()} returns the window's mean weights and {@link #apply()}
 * loads them.
 */
public final class Swad {

    private final List<Var> parameters;
    private final List<double[]> segmentSums = new ArrayList<>();
    private final List<Integer> segmentCounts = new ArrayList<>();
    private final List<Double> losses = new ArrayList<>();
    private double[] current;
    private int currentCount;
    private int start = -1;
    private int end = -1;

    /**
     * @param parameters the parameters to average
     */
    public Swad(List<Var> parameters) {
        this.parameters = List.copyOf(parameters);
    }

    /** Adds the current weights to the running segment. */
    public void sample() {
        if (current == null) {
            current = new double[size()];
        }
        int o = 0;
        for (Var p : parameters) {
            float[] v = p.value().toArray();
            for (float f : v) {
                current[o++] += f;
            }
        }
        currentCount++;
    }

    /**
     * Closes the running segment with its validation loss.
     *
     * @param validationLoss the loss at this evaluation
     */
    public void evaluate(double validationLoss) {
        segmentSums.add(current == null ? new double[size()] : current);
        segmentCounts.add(currentCount);
        losses.add(validationLoss);
        current = null;
        currentCount = 0;
    }

    /**
     * Chooses the averaging window from the loss history.
     *
     * @param patience evaluations without improvement that start the window, 3 in the paper
     * @param tolerance the overfitting ratio to the minimum loss, 1.2 in the paper
     * @param endPatience consecutive overfitted evaluations that end it, 6 in the paper
     * @return {@code {start, end}}, inclusive segment indices
     */
    public int[] window(int patience, double tolerance, int endPatience) {
        int n = losses.size();
        int s = -1;
        double best = Double.POSITIVE_INFINITY;
        int bestAt = 0;
        for (int i = 0; i < n; i++) {
            if (losses.get(i) < best) {
                best = losses.get(i);
                bestAt = i;
            } else if (i - bestAt >= patience) {
                s = bestAt;
                break;
            }
        }
        if (s < 0) {
            s = Math.max(0, n - patience);
        }
        double min = Double.POSITIVE_INFINITY;
        for (double l : losses) {
            min = Math.min(min, l);
        }
        int e = n - 1;
        int run = 0;
        for (int i = s; i < n; i++) {
            run = losses.get(i) > tolerance * min ? run + 1 : 0;
            if (run >= endPatience) {
                e = i - endPatience;
                break;
            }
        }
        start = s;
        end = Math.max(s, e);
        return new int[] {start, end};
    }

    /**
     * @return the mean weights over the window (with the paper's defaults if {@link #window}
     *     was not called)
     */
    public List<Tensor> average() {
        if (start < 0) {
            window(3, 1.2, 6);
        }
        double[] sum = new double[size()];
        long count = 0;
        for (int i = start; i <= end; i++) {
            double[] seg = segmentSums.get(i);
            for (int j = 0; j < sum.length; j++) {
                sum[j] += seg[j];
            }
            count += segmentCounts.get(i);
        }
        List<Tensor> out = new ArrayList<>();
        int o = 0;
        for (Var p : parameters) {
            int len = Math.toIntExact(p.value().size());
            float[] v = new float[len];
            for (int j = 0; j < len; j++) {
                v[j] = (float) (sum[o + j] / Math.max(1, count));
            }
            o += len;
            out.add(Tensor.of(p.value().shape(), v));
        }
        return out;
    }

    /** Loads the averaged weights into the parameters. */
    public void apply() {
        List<Tensor> avg = average();
        for (int i = 0; i < parameters.size(); i++) {
            parameters.get(i).assign(avg.get(i));
        }
    }

    private int size() {
        long s = 0;
        for (Var p : parameters) {
            s += p.value().size();
        }
        return Math.toIntExact(s);
    }
}
