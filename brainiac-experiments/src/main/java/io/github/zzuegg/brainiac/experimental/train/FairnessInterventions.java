package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Group-fairness interventions for a binary classifier and a binary protected attribute
 * {@code a}, one at each stage of the pipeline, plus the gaps they target.
 *
 * <ul>
 *   <li>Pre-processing, {@link #reweighing} (Kamiran and Calders 2012): example weights
 *       {@code P(a) P(y) / P(a, y)} that make label and attribute independent in the
 *       weighted training set.</li>
 *   <li>In-processing, {@link #adversarialLoss} (Zhang, Lemoine, Mitchell 2018,
 *       <a href="https://arxiv.org/abs/1801.07593">arXiv:1801.07593</a>): an adversary
 *       predicts {@code a} from the classifier's logit (and, for equalized odds, the true
 *       label); the classifier minimizes its loss minus {@code alpha} times the adversary's,
 *       and the adversary minimizes its own.</li>
 *   <li>Post-processing, {@link #equalizedOddsThresholds} (Hardt, Price, Srebro 2016,
 *       <a href="https://arxiv.org/abs/1610.02413">arXiv:1610.02413</a>): a decision
 *       threshold per group, chosen on held-out scores to maximize accuracy with the
 *       true- and false-positive rates of the groups within {@code tolerance} (the
 *       deterministic-threshold form of the paper's derived predictor).</li>
 * </ul>
 */
public final class FairnessInterventions {

    private FairnessInterventions() {
    }

    /**
     * Reweighing weights.
     *
     * @param y labels, 0 or 1
     * @param a attributes, 0 or 1
     * @return one weight per example, averaging 1
     */
    public static double[] reweighing(int[] y, int[] a) {
        int n = y.length;
        double[][] joint = new double[2][2];
        double[] py = new double[2];
        double[] pa = new double[2];
        for (int i = 0; i < n; i++) {
            joint[a[i]][y[i]] += 1.0 / n;
            py[y[i]] += 1.0 / n;
            pa[a[i]] += 1.0 / n;
        }
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            w[i] = pa[a[i]] * py[y[i]] / joint[a[i]][y[i]];
        }
        return w;
    }

    /**
     * The adversary's input for adversarial debiasing: the logit, plus for equalized odds
     * the logit times the label and the label (so the adversary sees {@code a} only through
     * what the prediction adds beyond the label).
     *
     * @param logit the classifier's logits {@code [batch, 1]}
     * @param y labels {@code [batch, 1]} as 0/1 floats
     * @param equalizedOdds whether to condition on the label
     * @return {@code [batch, 1]} or {@code [batch, 3]}
     */
    public static Var adversaryInput(Var logit, Tensor y, boolean equalizedOdds) {
        if (!equalizedOdds) {
            return logit;
        }
        return Var.concat(1, logit, logit.mul(y), Var.of(y));
    }

    /**
     * Binary cross-entropy from logits, {@code softplus(z) - y z}, averaged.
     *
     * @param logit {@code [batch, 1]}
     * @param y targets {@code [batch, 1]}, 0/1 floats
     * @return the mean loss
     */
    public static Var binaryCrossEntropy(Var logit, Tensor y) {
        return logit.softplus().sub(logit.mul(y)).mean();
    }

    /**
     * The classifier's adversarial-debiasing objective, {@code task - alpha adversary}: the
     * classifier is pushed to make the adversary's loss large. The adversary is trained
     * separately on its own loss with the classifier's logit detached.
     *
     * @param taskLoss the classifier's loss
     * @param adversaryLoss the adversary's loss on the classifier's (attached) output
     * @param alpha the trade-off
     * @return the objective
     */
    public static Var adversarialLoss(Var taskLoss, Var adversaryLoss, float alpha) {
        return taskLoss.sub(adversaryLoss.mul(alpha));
    }

    /**
     * Per-group thresholds for equalized odds.
     *
     * @param scores held-out scores, higher meaning positive
     * @param y held-out labels
     * @param a held-out attributes
     * @param tolerance the largest allowed gap in true- and in false-positive rate
     * @return {@code {threshold for a = 0, threshold for a = 1}}; the accuracy-maximizing
     *         single threshold for both if no pair meets the tolerance
     */
    public static double[] equalizedOddsThresholds(double[] scores, int[] y, int[] a, double tolerance) {
        double[] grid = quantiles(scores, 60);
        double best = -1;
        double[] out = null;
        double bestGap = Double.POSITIVE_INFINITY;
        double[] fallback = null;
        for (double t0 : grid) {
            for (double t1 : grid) {
                double[] th = {t0, t1};
                double[] r = rates(scores, y, a, th);
                double gap = Math.max(Math.abs(r[0] - r[2]), Math.abs(r[1] - r[3]));
                double acc = r[4];
                if (gap <= tolerance && acc > best) {
                    best = acc;
                    out = th;
                }
                if (gap < bestGap) {
                    bestGap = gap;
                    fallback = th;
                }
            }
        }
        return out != null ? out : fallback;
    }

    private static double[] quantiles(double[] v, int k) {
        double[] s = v.clone();
        java.util.Arrays.sort(s);
        double[] q = new double[k];
        for (int i = 0; i < k; i++) {
            q[i] = s[(int) ((long) i * (s.length - 1) / (k - 1))];
        }
        return q;
    }

    /** {@code {tpr0, fpr0, tpr1, fpr1, accuracy}} of per-group thresholds. */
    private static double[] rates(double[] scores, int[] y, int[] a, double[] th) {
        double[][] c = new double[2][4];   // per group: tp, fn, fp, tn
        int correct = 0;
        for (int i = 0; i < scores.length; i++) {
            boolean pred = scores[i] >= th[a[i]];
            int k = y[i] == 1 ? (pred ? 0 : 1) : (pred ? 2 : 3);
            c[a[i]][k]++;
            correct += pred == (y[i] == 1) ? 1 : 0;
        }
        return new double[] {c[0][0] / Math.max(1, c[0][0] + c[0][1]), c[0][2] / Math.max(1, c[0][2] + c[0][3]),
            c[1][0] / Math.max(1, c[1][0] + c[1][1]), c[1][2] / Math.max(1, c[1][2] + c[1][3]),
            correct / (double) scores.length};
    }

    /**
     * The demographic-parity gap {@code |P(pred = 1 | a = 0) - P(pred = 1 | a = 1)|}.
     *
     * @param pred predictions, 0 or 1
     * @param a attributes
     * @return the gap
     */
    public static double demographicParityGap(int[] pred, int[] a) {
        double[] pos = new double[2];
        double[] cnt = new double[2];
        for (int i = 0; i < pred.length; i++) {
            pos[a[i]] += pred[i];
            cnt[a[i]]++;
        }
        return Math.abs(pos[0] / cnt[0] - pos[1] / cnt[1]);
    }

    /**
     * The equalized-odds gap: the larger of the groups' differences in true-positive and in
     * false-positive rate.
     *
     * @param pred predictions
     * @param y labels
     * @param a attributes
     * @return the gap
     */
    public static double equalizedOddsGap(int[] pred, int[] y, int[] a) {
        double[] scores = new double[pred.length];
        for (int i = 0; i < pred.length; i++) {
            scores[i] = pred[i];
        }
        double[] r = rates(scores, y, a, new double[] {0.5, 0.5});
        return Math.max(Math.abs(r[0] - r[2]), Math.abs(r[1] - r[3]));
    }
}
