package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;

/**
 * Pretraining-data detection: was this text in the model's training set? Used to flag
 * benchmark contamination and as a membership-inference audit. Three scores of a sequence
 * from the model's next-token distributions, all higher for likely members:
 *
 * <ul>
 *   <li>{@link #logLikelihood}: the mean token log-probability, the loss-threshold attack
 *       (Yeom et al. 2018) that perplexity filters amount to;</li>
 *   <li>{@link #minK}: Min-K% Prob (Shi et al. 2024, <a href="https://arxiv.org/abs/2310.16789">arXiv:2310.16789</a>),
 *       the mean of the {@code k} lowest token log-probabilities: unseen text has a few
 *       surprising tokens, seen text has none;</li>
 *   <li>{@link #minKPlusPlus}: Min-K%++ (Zhang et al. 2024, <a href="https://arxiv.org/abs/2404.02936">arXiv:2404.02936</a>,
 *       ICLR 2025), which first normalises each token's log-probability by the mean and
 *       standard deviation of the log-probability under the model's own distribution at that
 *       position, {@code (log p(x_t) - E_z[log p(z)]) / std_z[log p(z)]}. Training makes the
 *       sequence's tokens local maxima of the likelihood along the vocabulary, which the
 *       normalisation detects regardless of how hard the position is.</li>
 * </ul>
 * Scores take the logits at every position ({@code [T, V]}, row {@code t} predicting
 * {@code targets[t]}).
 */
public final class ContaminationDetector {

    private ContaminationDetector() {
    }

    private static double[][] logSoftmax(Tensor logits) {
        int rows = logits.shape().dim(0);
        int v = logits.shape().dim(1);
        float[] z = logits.toArray();
        double[][] out = new double[rows][v];
        for (int t = 0; t < rows; t++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < v; k++) {
                max = Math.max(max, z[t * v + k]);
            }
            double sum = 0;
            for (int k = 0; k < v; k++) {
                sum += Math.exp(z[t * v + k] - max);
            }
            double lse = max + Math.log(sum);
            for (int k = 0; k < v; k++) {
                out[t][k] = z[t * v + k] - lse;
            }
        }
        return out;
    }

    private static double lowestMean(double[] x, double k) {
        double[] s = x.clone();
        Arrays.sort(s);
        int n = Math.max(1, (int) Math.ceil(k * s.length));
        double m = 0;
        for (int i = 0; i < n; i++) {
            m += s[i] / n;
        }
        return m;
    }

    /**
     * @param logits {@code [T, V]}
     * @param targets the {@code T} tokens
     * @return the mean token log-probability
     */
    public static double logLikelihood(Tensor logits, int[] targets) {
        double[][] lp = logSoftmax(logits);
        double m = 0;
        for (int t = 0; t < targets.length; t++) {
            m += lp[t][targets[t]] / targets.length;
        }
        return m;
    }

    /**
     * Min-K% Prob.
     *
     * @param logits {@code [T, V]}
     * @param targets the {@code T} tokens
     * @param k the share of lowest tokens averaged, for example 0.2
     * @return the score
     */
    public static double minK(Tensor logits, int[] targets, double k) {
        double[][] lp = logSoftmax(logits);
        double[] x = new double[targets.length];
        for (int t = 0; t < targets.length; t++) {
            x[t] = lp[t][targets[t]];
        }
        return lowestMean(x, k);
    }

    /**
     * Min-K%++.
     *
     * @param logits {@code [T, V]}
     * @param targets the {@code T} tokens
     * @param k the share of lowest normalised scores averaged, for example 0.2
     * @return the score
     */
    public static double minKPlusPlus(Tensor logits, int[] targets, double k) {
        double[][] lp = logSoftmax(logits);
        double[] x = new double[targets.length];
        for (int t = 0; t < targets.length; t++) {
            double mu = 0;
            double sq = 0;
            for (double l : lp[t]) {
                double p = Math.exp(l);
                mu += p * l;
                sq += p * l * l;
            }
            double sigma = Math.sqrt(Math.max(sq - mu * mu, 1e-12));
            x[t] = (lp[t][targets[t]] - mu) / sigma;
        }
        return lowestMean(x, k);
    }

    /**
     * The area under the ROC curve of telling members from non-members by a higher score.
     *
     * @param members the scores of member sequences
     * @param nonMembers the scores of non-members
     * @return the probability that a random member outscores a random non-member, ties half
     */
    public static double auroc(double[] members, double[] nonMembers) {
        double s = 0;
        for (double m : members) {
            for (double n : nonMembers) {
                s += m > n ? 1 : m == n ? 0.5 : 0;
            }
        }
        return s / ((double) members.length * nonMembers.length);
    }
}
