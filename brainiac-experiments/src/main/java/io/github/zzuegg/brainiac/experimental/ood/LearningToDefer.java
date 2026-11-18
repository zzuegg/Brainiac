package io.github.zzuegg.brainiac.experimental.ood;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;

/**
 * Learning to defer to an expert: a classifier with an extra "defer" output that hands an
 * input to a human or larger model exactly where the expert is more likely right.
 *
 * <ul>
 *   <li>{@link #softmaxSurrogate}: the consistent cross-entropy surrogate of Mozannar and Sontag
 *       (ICML 2020, <a href="https://arxiv.org/abs/2006.01862">arXiv:2006.01862</a>):
 *       {@code -log p_y - [m = y] log p_defer} over a softmax on {@code classes + 1}
 *       outputs; its expert-correctness estimate is {@code p_defer / (1 - p_defer)}, which is
 *       not bounded by 1 and tends to be miscalibrated.</li>
 *   <li>{@link #oneVsAllSurrogate}: one-vs-all logistic losses (Verma and Nalisnick, ICML 2022,
 *       <a href="https://arxiv.org/abs/2202.03673">arXiv:2202.03673</a>):
 *       {@code phi(g_y) + sum_{k != y} phi(-g_k) + phi(-g_defer) + [m = y] (phi(g_defer) -
 *       phi(-g_defer))} with {@code phi(z) = log(1 + exp(-z))}; {@code sigmoid(g_defer)} is a
 *       calibrated estimate of the expert's correctness.</li>
 *   <li>Both defer when the defer output beats every class output; {@link #deferByBudget}
 *       instead defers the inputs with the largest margin up to a deferral budget, and
 *       {@link #twoStage} is the baseline that compares classifier confidence with a separately
 *       learned expert-correctness model.</li>
 * </ul>
 */
public final class LearningToDefer {

    private LearningToDefer() {
    }

    /**
     * The softmax (cross-entropy) surrogate.
     *
     * @param logits {@code [n, classes + 1]}, the last column the defer output
     * @param labels true classes {@code [n]}
     * @param expertCorrect 1 where the expert's answer equals the label, else 0, {@code [n]}
     * @return the mean surrogate loss
     */
    public static Var softmaxSurrogate(Var logits, Tensor labels, Tensor expertCorrect) {
        int n = logits.shape().dim(0);
        int c = logits.shape().dim(1);
        Var logP = logits.sub(logits.logsumexp(1, true));
        float[] y = labels.toArray();
        float[] m = expertCorrect.toArray();
        float[] w = new float[n * c];
        for (int i = 0; i < n; i++) {
            w[i * c + (int) y[i]] = 1;
            w[i * c + c - 1] = m[i];
        }
        return logP.mul(Tensor.of(Shape.of(n, c), w)).sum(1, false).mean().neg();
    }

    /**
     * The one-vs-all surrogate.
     *
     * @param logits {@code [n, classes + 1]}, the last column the defer output
     * @param labels true classes {@code [n]}
     * @param expertCorrect 1 where the expert is right, else 0, {@code [n]}
     * @return the mean surrogate loss
     */
    public static Var oneVsAllSurrogate(Var logits, Tensor labels, Tensor expertCorrect) {
        int n = logits.shape().dim(0);
        int c = logits.shape().dim(1);
        float[] y = labels.toArray();
        float[] m = expertCorrect.toArray();
        // sign s = +1 where the logistic target is "on": the true class, and the defer output when the expert is right
        float[] sign = new float[n * c];
        Arrays.fill(sign, -1);
        for (int i = 0; i < n; i++) {
            sign[i * c + (int) y[i]] = 1;
            if (m[i] > 0.5f) {
                sign[i * c + c - 1] = 1;
            }
        }
        // phi(s z) = softplus(-s z)
        return logits.mul(Tensor.of(Shape.of(n, c), sign)).neg().softplus().sum(1, false).mean();
    }

    /**
     * @param logits {@code [n, classes + 1]}
     * @return whether the defer output is the largest
     */
    public static boolean[] defer(Tensor logits) {
        int c = logits.shape().dim(1);
        int[] arg = logits.argmax(1).toIntArray();
        boolean[] out = new boolean[arg.length];
        for (int i = 0; i < arg.length; i++) {
            out[i] = arg[i] == c - 1;
        }
        return out;
    }

    /**
     * @param logits {@code [n, classes + 1]}
     * @return the defer output minus the largest class output (the deferral margin)
     */
    public static double[] deferMargin(Tensor logits) {
        int c = logits.shape().dim(1);
        double[] v = logits.toDoubleArray();
        double[] out = new double[v.length / c];
        for (int i = 0; i < out.length; i++) {
            double best = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < c - 1; j++) {
                best = Math.max(best, v[i * c + j]);
            }
            out[i] = v[i * c + c - 1] - best;
        }
        return out;
    }

    /**
     * @param logits {@code [n, classes + 1]}
     * @return the predicted class among the real classes
     */
    public static int[] predict(Tensor logits) {
        int c = logits.shape().dim(1);
        return logits.slice(1, 0, c - 1).argmax(1).toIntArray();
    }

    /**
     * The softmax model's estimate of the expert's correctness.
     *
     * @param logits {@code [n, classes + 1]}
     * @return {@code min(1, p_defer / (1 - p_defer))}
     */
    public static double[] expertCorrectnessSoftmax(Tensor logits) {
        int c = logits.shape().dim(1);
        double[] p = logits.sub(logits.logsumexp(1, true)).exp().toDoubleArray();
        double[] out = new double[p.length / c];
        for (int i = 0; i < out.length; i++) {
            double d = p[i * c + c - 1];
            out[i] = Math.min(1, d / Math.max(1e-12, 1 - d));
        }
        return out;
    }

    /**
     * The one-vs-all model's estimate of the expert's correctness.
     *
     * @param logits {@code [n, classes + 1]}
     * @return {@code sigmoid(g_defer)}
     */
    public static double[] expertCorrectnessOneVsAll(Tensor logits) {
        int c = logits.shape().dim(1);
        double[] v = logits.toDoubleArray();
        double[] out = new double[v.length / c];
        for (int i = 0; i < out.length; i++) {
            out[i] = 1 / (1 + Math.exp(-v[i * c + c - 1]));
        }
        return out;
    }

    /**
     * Defers the inputs with the largest scores, at most a budget share.
     *
     * @param score larger means "defer first"
     * @param budget the share of inputs that may be deferred
     * @return the deferral decisions
     */
    public static boolean[] deferByBudget(double[] score, double budget) {
        Integer[] order = new Integer[score.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(score[b], score[a]));
        boolean[] out = new boolean[score.length];
        int m = (int) Math.round(budget * score.length);
        for (int i = 0; i < m; i++) {
            out[order[i]] = true;
        }
        return out;
    }

    /**
     * The two-stage rule: defer where the expert-correctness model's estimate beats the
     * classifier's confidence.
     *
     * @param confidence the classifier's maximum softmax probability
     * @param expertCorrect the estimated probability that the expert is right
     * @return the margin {@code expert - confidence} (defer when positive)
     */
    public static double[] twoStage(double[] confidence, double[] expertCorrect) {
        double[] out = new double[confidence.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = expertCorrect[i] - confidence[i];
        }
        return out;
    }

    /**
     * The accuracy of the classifier-plus-expert system.
     *
     * @param predicted the classifier's answers
     * @param expert the expert's answers
     * @param labels the truth
     * @param defer where the expert answers
     * @return the share answered right
     */
    public static double systemAccuracy(int[] predicted, int[] expert, int[] labels, boolean[] defer) {
        int ok = 0;
        for (int i = 0; i < labels.length; i++) {
            ok += (defer[i] ? expert[i] : predicted[i]) == labels[i] ? 1 : 0;
        }
        return ok / (double) labels.length;
    }
}
