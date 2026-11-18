package io.github.zzuegg.brainiac.experimental.ood;

import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;

import java.util.Arrays;

/**
 * The standard OOD detection metrics, for scores where higher means "more in-distribution":
 * AUROC (the probability that a random in-distribution input outscores a random OOD input)
 * and FPR at 95% TPR (the share of OOD inputs accepted when the threshold keeps 95% of the
 * in-distribution inputs), as in Hendrycks and Gimpel (ICLR 2017).
 */
public final class OodMetrics {

    private OodMetrics() {
    }

    /**
     * The AUROC of separating in-distribution from OOD inputs.
     *
     * @param in the scores of in-distribution inputs
     * @param out the scores of OOD inputs
     * @return the AUROC in {@code [0, 1]}
     */
    public static double auroc(double[] in, double[] out) {
        double[] s = new double[in.length + out.length];
        boolean[] pos = new boolean[s.length];
        System.arraycopy(in, 0, s, 0, in.length);
        System.arraycopy(out, 0, s, in.length, out.length);
        Arrays.fill(pos, 0, in.length, true);
        return MlMetrics.auroc(s, pos);
    }

    /**
     * The false-positive rate at 95% true-positive rate.
     *
     * @param in the scores of in-distribution inputs
     * @param out the scores of OOD inputs
     * @return the share of OOD inputs at or above the 5th percentile of the in-distribution scores
     */
    public static double fpr95(double[] in, double[] out) {
        double[] sorted = in.clone();
        Arrays.sort(sorted);
        double threshold = sorted[(int) Math.floor(0.05 * (sorted.length - 1))];
        int accepted = 0;
        for (double o : out) {
            accepted += o >= threshold ? 1 : 0;
        }
        return accepted / (double) out.length;
    }
}
