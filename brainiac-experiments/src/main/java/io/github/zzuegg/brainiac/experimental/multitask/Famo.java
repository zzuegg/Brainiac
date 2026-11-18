package io.github.zzuegg.brainiac.experimental.multitask;

import io.github.zzuegg.brainiac.autograd.Var;

import java.util.List;

/**
 * FAMO, fast adaptive multitask optimisation (Liu et al., NeurIPS 2023): balanced progress on
 * every task, as the gradient-manipulation methods aim for, but with one backward pass and
 * {@code O(1)} extra space and time per step.
 *
 * <p>FAMO keeps task logits {@code xi} with weights {@code z = softmax(xi)}. Each step it
 * descends {@code sum_t z_t log L_t / c} with {@code c = sum_t z_t / L_t} (held fixed), which
 * weighs each task's gradient by {@code z_t / L_t} and so works on log-losses, making it
 * insensitive to loss scales ({@link #weightedLoss}). After the parameter step it measures the
 * decrease of every log-loss on the same batch, {@code delta_t = log L_t(old) - log L_t(new)},
 * and moves the logits by the softmax Jacobian times {@code delta} with Adam and a weight
 * decay {@code gamma} ({@link #update}): tasks that improved less get more weight, steering
 * towards an equal rate of improvement. The extra cost is one forward pass, not one backward
 * pass per task.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2306.03792">arXiv:2306.03792</a>.
 */
public final class Famo {

    private final int tasks;
    private final double rate;
    private final double gamma;
    private final double[] logits;
    private final double[] m;
    private final double[] v;
    private double[] previous;
    private int step;

    /**
     * Creates FAMO with equal weights.
     *
     * @param tasks the number of tasks
     * @param rate the logits' Adam learning rate, 0.025 in the paper
     * @param gamma the logits' weight decay, 1e-3 in the paper
     */
    public Famo(int tasks, double rate, double gamma) {
        this.tasks = tasks;
        this.rate = rate;
        this.gamma = gamma;
        this.logits = new double[tasks];
        this.m = new double[tasks];
        this.v = new double[tasks];
    }

    /** @return the current task weights {@code softmax(xi)} */
    public double[] weights() {
        double max = Double.NEGATIVE_INFINITY;
        for (double x : logits) {
            max = Math.max(max, x);
        }
        double[] z = new double[tasks];
        double sum = 0;
        for (int t = 0; t < tasks; t++) {
            z[t] = Math.exp(logits[t] - max);
            sum += z[t];
        }
        for (int t = 0; t < tasks; t++) {
            z[t] /= sum;
        }
        return z;
    }

    /**
     * The loss to backpropagate this step; remembers the losses for {@link #update}.
     *
     * @param losses the task losses, each a positive scalar
     * @return {@code sum_t z_t log(L_t) / c}
     */
    public Var weightedLoss(List<Var> losses) {
        double[] z = weights();
        previous = new double[tasks];
        double c = 0;
        for (int t = 0; t < tasks; t++) {
            previous[t] = losses.get(t).value().item();
            c += z[t] / (previous[t] + 1e-8);
        }
        Var total = null;
        for (int t = 0; t < tasks; t++) {
            Var term = losses.get(t).add(1e-8f).log().mul((float) (z[t] / c));
            total = total == null ? term : total.add(term);
        }
        return total;
    }

    /**
     * Updates the task logits from the losses on the same batch after the parameter step.
     *
     * @param current the task losses after the step
     */
    public void update(double[] current) {
        if (previous == null) {
            throw new IllegalStateException("call weightedLoss before update");
        }
        double[] z = weights();
        double[] delta = new double[tasks];
        for (int t = 0; t < tasks; t++) {
            delta[t] = Math.log(previous[t] + 1e-8) - Math.log(current[t] + 1e-8);
        }
        // gradient of z . delta with respect to xi: J_softmax^T delta = z * (delta - z . delta)
        double zd = 0;
        for (int t = 0; t < tasks; t++) {
            zd += z[t] * delta[t];
        }
        step++;
        for (int t = 0; t < tasks; t++) {
            double g = z[t] * (delta[t] - zd) + gamma * logits[t];
            m[t] = 0.9 * m[t] + 0.1 * g;
            v[t] = 0.999 * v[t] + 0.001 * g * g;
            double mh = m[t] / (1 - Math.pow(0.9, step));
            double vh = v[t] / (1 - Math.pow(0.999, step));
            logits[t] -= rate * mh / (Math.sqrt(vh) + 1e-8);
        }
    }
}
