package io.github.zzuegg.brainiac.experimental.labels;

import io.github.zzuegg.brainiac.experimental.ml.GaussianMixture;
import io.github.zzuegg.brainiac.experimental.semisup.SoftTargets;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;

/**
 * Picking the probably-clean examples of a noisy-labelled batch. Networks fit clean examples
 * before memorising noisy ones, so small-loss examples are probably clean.
 *
 * <ul>
 *   <li>Co-teaching (Han et al., NeurIPS 2018, <a href="https://arxiv.org/abs/1804.06872">arXiv:1804.06872</a>):
 *       two networks; each updates only on the {@link #keepRate} share of the batch with the
 *       smallest loss <em>under its peer</em> ({@link #smallLoss}), the share falling linearly from
 *       1 to {@code 1 - noise rate} over the first {@code Tk} steps. The cross-update keeps one
 *       network's selection errors from reinforcing themselves.</li>
 *   <li>Co-teaching+ (Yu et al., ICML 2019): the same, restricted to the examples the two networks
 *       disagree on ({@link #disagreement}).</li>
 *   <li>DivideMix's split (Li, Socher, Hoi, ICLR 2020, <a href="https://arxiv.org/abs/2002.07394">arXiv:2002.07394</a>):
 *       a two-component Gaussian mixture ({@link GaussianMixture}) on the per-example losses; the
 *       posterior of the low-mean component is the probability of being clean
 *       ({@link #cleanProbability}). Clean examples keep refined labels
 *       ({@link #refineLabels}), the rest become unlabelled, and both are trained with MixMatch
 *       ({@code experimental.semisup.MixMatch}) by two networks that divide the data for each
 *       other.</li>
 * </ul>
 */
public final class SampleSelection {

    private SampleSelection() {
    }

    /**
     * Co-teaching's schedule of the kept share.
     *
     * @param step the training step
     * @param noiseRate the (estimated) noise rate
     * @param rampSteps {@code Tk}, the steps over which the share falls
     * @return {@code 1 - min(step / Tk, 1) * noiseRate}
     */
    public static double keepRate(int step, double noiseRate, int rampSteps) {
        return 1 - Math.min(step / (double) rampSteps, 1) * noiseRate;
    }

    /**
     * The positions of the smallest losses.
     *
     * @param losses per-example losses
     * @param keep the share to keep
     * @return the indices of the {@code ceil(keep * n)} smallest losses
     */
    public static int[] smallLoss(double[] losses, double keep) {
        Integer[] order = new Integer[losses.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(losses[a], losses[b]));
        int k = Math.max(1, (int) Math.ceil(keep * losses.length));
        int[] out = new int[k];
        for (int i = 0; i < k; i++) {
            out[i] = order[i];
        }
        return out;
    }

    /**
     * Per-example cross-entropy of logits against labels, without gradient.
     *
     * @param logits {@code [n, classes]}
     * @param labels the (noisy) labels
     * @return {@code -log p_y} per row
     */
    public static double[] losses(Tensor logits, int[] labels) {
        float[][] p = SoftTargets.probabilities(logits, 1f);
        double[] out = new double[labels.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = -Math.log(Math.max(p[i][labels[i]], 1e-12));
        }
        return out;
    }

    /**
     * Co-teaching+'s disagreement set.
     *
     * @param a the first network's logits
     * @param b the second network's logits
     * @return the rows whose predicted classes differ (all rows if none do)
     */
    public static int[] disagreement(Tensor a, Tensor b) {
        int[] pa = SoftTargets.argmax(SoftTargets.probabilities(a, 1f));
        int[] pb = SoftTargets.argmax(SoftTargets.probabilities(b, 1f));
        int n = 0;
        for (int i = 0; i < pa.length; i++) {
            n += pa[i] != pb[i] ? 1 : 0;
        }
        if (n == 0) {
            int[] all = new int[pa.length];
            for (int i = 0; i < all.length; i++) {
                all[i] = i;
            }
            return all;
        }
        int[] out = new int[n];
        int k = 0;
        for (int i = 0; i < pa.length; i++) {
            if (pa[i] != pb[i]) {
                out[k++] = i;
            }
        }
        return out;
    }

    /**
     * DivideMix's clean probability: fits a two-component mixture to the min-max normalised
     * losses and returns each example's posterior under the component with the smaller mean.
     *
     * @param losses per-example losses over the whole training set
     * @param seed the mixture's seed
     * @return the probability of each example being clean
     */
    public static double[] cleanProbability(double[] losses, long seed) {
        double lo = Double.POSITIVE_INFINITY;
        double hi = Double.NEGATIVE_INFINITY;
        for (double l : losses) {
            lo = Math.min(lo, l);
            hi = Math.max(hi, l);
        }
        double[][] x = new double[losses.length][1];
        for (int i = 0; i < x.length; i++) {
            x[i][0] = (losses[i] - lo) / Math.max(hi - lo, 1e-12);
        }
        GaussianMixture gmm = GaussianMixture.fit(x, 2, 20, 5e-4, seed);
        int clean = gmm.mean(0)[0] <= gmm.mean(1)[0] ? 0 : 1;
        double[] w = gmm.weights();
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double[] d = new double[2];
            for (int j = 0; j < 2; j++) {
                double m = gmm.mean(j)[0];
                double v = gmm.covariance(j)[0][0];
                d[j] = w[j] * Math.exp(-(x[i][0] - m) * (x[i][0] - m) / (2 * v)) / Math.sqrt(v);
            }
            out[i] = d[clean] / Math.max(d[0] + d[1], 1e-300);
        }
        return out;
    }

    /**
     * DivideMix's label refinement of a clean-set example: {@code w y + (1 - w) p}, sharpened.
     *
     * @param labels the noisy labels
     * @param cleanProbability the clean probability {@code w} of each row
     * @param predictions the network's averaged predictions on augmented views
     * @param temperature the sharpening temperature, 0.5
     * @return refined label rows
     */
    public static float[][] refineLabels(int[] labels, double[] cleanProbability, float[][] predictions,
            double temperature) {
        int classes = predictions[0].length;
        float[][] out = new float[labels.length][classes];
        for (int i = 0; i < labels.length; i++) {
            for (int c = 0; c < classes; c++) {
                out[i][c] = (float) ((1 - cleanProbability[i]) * predictions[i][c]);
            }
            out[i][labels[i]] += (float) cleanProbability[i];
        }
        return SoftTargets.sharpen(out, temperature);
    }
}
