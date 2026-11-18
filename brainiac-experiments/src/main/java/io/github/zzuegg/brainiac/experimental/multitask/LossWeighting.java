package io.github.zzuegg.brainiac.experimental.multitask;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.List;

/**
 * Loss weighting for multi-task learning: summing task losses lets the task with the largest
 * loss or gradient dominate, so each scheme turns the per-task losses into one weighted total.
 *
 * <ul>
 *   <li>{@link Kind#EQUAL}: the plain sum, the baseline.</li>
 *   <li>{@link Kind#UNCERTAINTY} (Kendall, Gal and Cipolla, CVPR 2018): homoscedastic
 *       uncertainty weighting with a learned {@code s_t = log sigma_t^2} per task:
 *       {@code sum_t (exp(-s_t) L_t + s_t) / 2}; noisy or large-scale tasks get small weights.
 *       The {@code s_t} are trainable {@link #parameters()}.</li>
 *   <li>{@link Kind#GRADNORM} (Chen et al., ICML 2018): weights {@code w_t} (summing to the
 *       number of tasks) are tuned so that the gradient norms {@code G_t = |grad_W w_t L_t|} on
 *       the last shared layer {@code W} track {@code mean(G) r_t^alpha}, where {@code r_t} is
 *       task {@code t}'s relative inverse training rate {@code (L_t / L_t(0)) / mean}. The
 *       weights take a gradient step on {@code sum_t |G_t - target_t|} (target held fixed),
 *       costing one extra backward pass per task.</li>
 *   <li>{@link Kind#DWA} (Liu, Johns and Davison, CVPR 2019): dynamic weight averaging, with
 *       no extra gradients: {@code w_t = T softmax(r_t / temperature)} with {@code r_t} the
 *       ratio of the task's mean loss over the last two periods.</li>
 * </ul>
 *
 * <p>Papers: <a href="https://arxiv.org/abs/1705.07115">arXiv:1705.07115</a>,
 * <a href="https://arxiv.org/abs/1711.02257">arXiv:1711.02257</a>,
 * <a href="https://arxiv.org/abs/1803.10704">arXiv:1803.10704</a>.
 */
public final class LossWeighting {

    /** The weighting scheme. */
    public enum Kind {
        /** The plain sum. */
        EQUAL,
        /** Learned homoscedastic uncertainty. */
        UNCERTAINTY,
        /** Gradient normalisation. */
        GRADNORM,
        /** Dynamic weight averaging. */
        DWA
    }

    private final Kind kind;
    private final int tasks;
    private final float alpha;
    private final float temperature;
    private final int period;
    private final float rate;
    private final Var logVariance;
    private final double[] weights;
    private final double[] initial;
    private final double[] adamM;
    private final double[] adamV;
    private final double[] periodSum;
    private final double[] lastPeriod;
    private final double[] previousPeriod;
    private int periodSteps;
    private int completedPeriods;
    private int step;

    /**
     * Creates a weighting.
     *
     * @param kind the scheme
     * @param tasks the number of tasks
     * @param alpha GradNorm's asymmetry, 1.5 in the paper (ignored otherwise)
     * @param temperature DWA's softmax temperature, 2 in the paper (ignored otherwise)
     * @param period DWA's averaging period in steps (an epoch in the paper)
     * @param rate GradNorm's Adam learning rate on the weights, 0.025 in the paper
     */
    public LossWeighting(Kind kind, int tasks, float alpha, float temperature, int period, float rate) {
        this.kind = kind;
        this.tasks = tasks;
        this.alpha = alpha;
        this.temperature = temperature;
        this.period = period;
        this.rate = rate;
        this.logVariance = Var.param(Tensor.zeros(Shape.of(tasks)));
        this.weights = new double[tasks];
        Arrays.fill(weights, 1);
        this.initial = new double[tasks];
        this.adamM = new double[tasks];
        this.adamV = new double[tasks];
        this.periodSum = new double[tasks];
        this.lastPeriod = new double[tasks];
        this.previousPeriod = new double[tasks];
    }

    /**
     * A weighting with the papers' settings ({@code alpha = 1.5}, temperature 2, period 20
     * steps, rate 0.025).
     *
     * @param kind the scheme
     * @param tasks the number of tasks
     * @return the weighting
     */
    public static LossWeighting of(Kind kind, int tasks) {
        return new LossWeighting(kind, tasks, 1.5f, 2, 20, 0.025f);
    }

    /** @return the trainable parameters: the log-variances for {@link Kind#UNCERTAINTY}, else none */
    public List<Var> parameters() {
        return kind == Kind.UNCERTAINTY ? List.of(logVariance) : List.of();
    }

    /** @return the current task weights (for UNCERTAINTY {@code exp(-s_t)}) */
    public double[] weights() {
        if (kind == Kind.UNCERTAINTY) {
            float[] s = logVariance.value().toArray();
            double[] w = new double[tasks];
            for (int t = 0; t < tasks; t++) {
                w[t] = Math.exp(-s[t]);
            }
            return w;
        }
        return weights.clone();
    }

    /**
     * Combines one step's task losses and advances the scheme's state.
     *
     * @param losses one scalar per task
     * @param lastShared the last shared layer's weights (used by GradNorm only)
     * @return the weighted total to backpropagate
     */
    public Var combine(List<Var> losses, List<Var> lastShared) {
        double[] l = new double[tasks];
        for (int t = 0; t < tasks; t++) {
            l[t] = losses.get(t).value().item();
        }
        if (step == 0) {
            System.arraycopy(l, 0, initial, 0, tasks);
        }
        step++;
        switch (kind) {
            case UNCERTAINTY -> {
                Var total = null;
                for (int t = 0; t < tasks; t++) {
                    Var s = logVariance.slice(0, t, t + 1).sum();
                    Var term = losses.get(t).mul(s.neg().exp()).add(s).mul(0.5f);
                    total = total == null ? term : total.add(term);
                }
                return total;
            }
            case GRADNORM -> gradNorm(losses, l, lastShared);
            case DWA -> dwa(l);
            default -> {
            }
        }
        Var total = null;
        for (int t = 0; t < tasks; t++) {
            Var term = losses.get(t).mul((float) weights[t]);
            total = total == null ? term : total.add(term);
        }
        return total;
    }

    private void gradNorm(List<Var> losses, double[] l, List<Var> lastShared) {
        double[] norms = new double[tasks];
        double[] ratio = new double[tasks];
        double meanRatio = 0;
        for (int t = 0; t < tasks; t++) {
            norms[t] = TaskGradients.norm(TaskGradients.flatten(Var.gradients(losses.get(t), lastShared, false)));
            ratio[t] = l[t] / Math.max(1e-12, initial[t]);
            meanRatio += ratio[t] / tasks;
        }
        double meanG = 0;
        for (int t = 0; t < tasks; t++) {
            meanG += weights[t] * norms[t] / tasks;
        }
        for (int t = 0; t < tasks; t++) {
            double target = meanG * Math.pow(ratio[t] / meanRatio, alpha);
            double g = Math.signum(weights[t] * norms[t] - target) * norms[t];
            adamM[t] = 0.9 * adamM[t] + 0.1 * g;
            adamV[t] = 0.999 * adamV[t] + 0.001 * g * g;
            double mh = adamM[t] / (1 - Math.pow(0.9, step));
            double vh = adamV[t] / (1 - Math.pow(0.999, step));
            weights[t] = Math.max(1e-3, weights[t] - rate * mh / (Math.sqrt(vh) + 1e-8));
        }
        double sum = 0;
        for (double w : weights) {
            sum += w;
        }
        for (int t = 0; t < tasks; t++) {
            weights[t] *= tasks / sum;
        }
    }

    private void dwa(double[] l) {
        for (int t = 0; t < tasks; t++) {
            periodSum[t] += l[t];
        }
        periodSteps++;
        if (periodSteps == period) {
            System.arraycopy(lastPeriod, 0, previousPeriod, 0, tasks);
            for (int t = 0; t < tasks; t++) {
                lastPeriod[t] = periodSum[t] / period;
                periodSum[t] = 0;
            }
            periodSteps = 0;
            completedPeriods++;
            if (completedPeriods >= 2) {
                double[] e = new double[tasks];
                double sum = 0;
                for (int t = 0; t < tasks; t++) {
                    e[t] = Math.exp(lastPeriod[t] / Math.max(1e-12, previousPeriod[t]) / temperature);
                    sum += e[t];
                }
                for (int t = 0; t < tasks; t++) {
                    weights[t] = tasks * e[t] / sum;
                }
            }
        }
    }
}
