package io.github.zzuegg.brainiac.experimental.data;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * Data mixing for pretraining on several domains (web, code, books, math...): which share
 * of each batch comes from each domain, and how that changes over training.
 * <ul>
 * <li>{@link #temperature}: temperature sampling over domain sizes, {@code w ~ n^(1/T)}
 *     (T5, XLM-R; Conneau et al. 2019, <a href="https://arxiv.org/abs/1911.02116">arXiv:1911.02116</a>):
 *     {@code T = 1} is proportional to size, larger {@code T} up-samples small domains;</li>
 * <li>{@link #schedule}: weights interpolated between stage mixtures over steps, as in the
 *     annealing phase of Llama 3 and OLMo 2, which up-weight high-quality data at the end;</li>
 * <li>{@link #updateWithExcessLoss}: online reweighting by excess loss, the
 *     multiplicative-weights rule of DoReMi (Xie et al. 2023,
 *     <a href="https://arxiv.org/abs/2305.10429">arXiv:2305.10429</a>):
 *     {@code w_i <- w_i exp(eta * max(0, loss_i - refLoss_i))}, renormalized and smoothed
 *     with uniform, so domains the model still finds hard get more data.</li>
 * </ul>
 */
public final class DataMixture {

    private final long[] boundaries;
    private final double[][] stages;
    private double[] online;

    private DataMixture(long[] boundaries, double[][] stages) {
        this.boundaries = boundaries;
        this.stages = stages;
    }

    /**
     * @param sizes the tokens (or documents) of each domain
     * @param temperature the sampling temperature, at least 1 to up-sample small domains
     * @return the weights, summing to 1
     */
    public static double[] temperature(long[] sizes, double temperature) {
        double[] w = new double[sizes.length];
        for (int i = 0; i < w.length; i++) {
            w[i] = Math.pow(sizes[i], 1 / temperature);
        }
        return normalize(w);
    }

    /**
     * A fixed mixture.
     *
     * @param weights the weights (normalized)
     * @return the mixture
     */
    public static DataMixture constant(double... weights) {
        return new DataMixture(new long[] {0}, new double[][] {normalize(weights.clone())});
    }

    /**
     * Mixtures that hold at given steps, linearly interpolated between them.
     *
     * @param steps ascending steps
     * @param weights the mixture at each step
     * @return the schedule
     */
    public static DataMixture schedule(long[] steps, double[][] weights) {
        if (steps.length != weights.length || steps.length == 0) {
            throw new IllegalArgumentException("one mixture per step");
        }
        double[][] w = new double[weights.length][];
        for (int i = 0; i < w.length; i++) {
            w[i] = normalize(weights[i].clone());
        }
        return new DataMixture(steps.clone(), w);
    }

    /**
     * @param step the training step
     * @return the mixture at that step
     */
    public double[] weights(long step) {
        if (online != null) {
            return online.clone();
        }
        if (step <= boundaries[0]) {
            return stages[0].clone();
        }
        for (int i = 1; i < boundaries.length; i++) {
            if (step <= boundaries[i]) {
                double f = (double) (step - boundaries[i - 1]) / (boundaries[i] - boundaries[i - 1]);
                double[] w = new double[stages[i].length];
                for (int k = 0; k < w.length; k++) {
                    w[k] = (1 - f) * stages[i - 1][k] + f * stages[i][k];
                }
                return w;
            }
        }
        return stages[stages.length - 1].clone();
    }

    /**
     * @param step the training step
     * @param random the source of randomness
     * @return a domain drawn from the mixture at that step
     */
    public int sample(long step, RandomGenerator random) {
        double[] w = weights(step);
        double u = random.nextDouble();
        for (int i = 0; i < w.length; i++) {
            u -= w[i];
            if (u < 0) {
                return i;
            }
        }
        return w.length - 1;
    }

    /**
     * @param step the training step
     * @param batch the batch size
     * @return how many samples of each domain a batch gets (largest-remainder rounding)
     */
    public int[] allocate(long step, int batch) {
        double[] w = weights(step);
        int[] n = new int[w.length];
        double[] rest = new double[w.length];
        int used = 0;
        for (int i = 0; i < w.length; i++) {
            n[i] = (int) Math.floor(w[i] * batch);
            rest[i] = w[i] * batch - n[i];
            used += n[i];
        }
        while (used < batch) {
            int best = 0;
            for (int i = 1; i < w.length; i++) {
                best = rest[i] > rest[best] ? i : best;
            }
            n[best]++;
            rest[best] = -1;
            used++;
        }
        return n;
    }

    /**
     * Switches to online reweighting (from the current weights) and applies one DoReMi
     * update.
     *
     * @param step the step whose weights are the starting point (first call only)
     * @param losses the current loss of each domain
     * @param referenceLosses the loss a reference model reaches on each domain
     * @param eta the step size, for example 1
     * @param smoothing the share of uniform mixed in, for example 1e-3
     * @return the new weights
     */
    public double[] updateWithExcessLoss(long step, double[] losses, double[] referenceLosses, double eta,
            double smoothing) {
        double[] w = online == null ? weights(step) : online;
        double[] next = new double[w.length];
        for (int i = 0; i < w.length; i++) {
            next[i] = Math.log(Math.max(w[i], 1e-12)) + eta * Math.max(0, losses[i] - referenceLosses[i]);
        }
        double max = Arrays.stream(next).max().orElse(0);
        for (int i = 0; i < next.length; i++) {
            next[i] = Math.exp(next[i] - max);
        }
        normalize(next);
        for (int i = 0; i < next.length; i++) {
            next[i] = (1 - smoothing) * next[i] + smoothing / next.length;
        }
        online = next;
        return next.clone();
    }

    private static double[] normalize(double[] w) {
        double s = 0;
        for (double x : w) {
            s += x;
        }
        for (int i = 0; i < w.length; i++) {
            w[i] /= s;
        }
        return w;
    }
}
