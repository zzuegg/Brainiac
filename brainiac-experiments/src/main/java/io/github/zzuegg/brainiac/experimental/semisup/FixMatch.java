package io.github.zzuegg.brainiac.experimental.semisup;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Augmentation;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * FixMatch (Sohn et al., NeurIPS 2020, <a href="https://arxiv.org/abs/2001.07685">arXiv:2001.07685</a>):
 * weak-to-strong consistency with a confidence threshold. The prediction on a weakly augmented
 * view (flip and shift) becomes a hard pseudo-label for a strongly augmented view (RandAugment
 * plus Cutout) wherever its confidence reaches the threshold {@code tau = 0.95}; the unlabelled
 * loss is the masked cross-entropy averaged over the whole unlabelled batch, and the unlabelled
 * batch is {@code mu} times the labelled one.
 *
 * <p>Which rows count, and how much, is a pluggable {@link Mask}: {@link #fixed(float)} is
 * FixMatch's own hard threshold; {@link AdaptiveThreshold} gives FlexMatch, FreeMatch and
 * SoftMatch.
 */
public final class FixMatch {

    /** Decides each unlabelled row's weight from the weak view's probabilities. */
    public interface Mask {

        /**
         * Weighs a batch and updates any internal state (class learning status, EMA thresholds).
         *
         * @param probs the weak view's probability rows
         * @param indices the rows' indices in the unlabelled pool (FlexMatch tracks them)
         * @return a weight in {@code [0, 1]} per row
         */
        float[] weights(float[][] probs, int[] indices);
    }

    /**
     * FixMatch's fixed threshold.
     *
     * @param tau the confidence threshold
     * @return a mask that keeps rows whose top probability is at least {@code tau}
     */
    public static Mask fixed(float tau) {
        return (probs, indices) -> {
            float[] w = new float[probs.length];
            for (int i = 0; i < probs.length; i++) {
                float max = 0;
                for (float p : probs[i]) {
                    max = Math.max(max, p);
                }
                w[i] = max >= tau ? 1 : 0;
            }
            return w;
        };
    }

    private final Mask mask;
    private final float lambdaU;
    private int kept;
    private int correct;
    private int seen;

    /**
     * Creates the method.
     *
     * @param mask the confidence mask
     * @param lambdaU the weight of the unlabelled loss, 1 in the paper
     */
    public FixMatch(Mask mask, float lambdaU) {
        this.mask = mask;
        this.lambdaU = lambdaU;
    }

    /**
     * The unlabelled loss.
     *
     * @param strongLogits the student's logits on the strong views, with gradient
     * @param weakLogits the logits on the weak views, used without gradient
     * @param indices the rows' indices in the unlabelled pool
     * @param truth the hidden true classes, only for the pseudo-label statistics; may be null
     * @return {@code lambdaU * sum_i w_i CE(strong_i, argmax weak_i) / batch}
     */
    public Var loss(Var strongLogits, Tensor weakLogits, int[] indices, int[] truth) {
        float[][] p = SoftTargets.probabilities(weakLogits, 1f);
        int[] hard = SoftTargets.argmax(p);
        float[] w = mask.weights(p, indices);
        for (int i = 0; i < w.length; i++) {
            seen++;
            if (w[i] >= 0.5f) {
                kept++;
                if (truth != null && hard[i] == truth[i]) {
                    correct++;
                }
            }
        }
        return SoftTargets.softCrossEntropy(strongLogits, SoftTargets.tensor(SoftTargets.oneHot(hard, p[0].length)),
                w, w.length).mul(lambdaU);
    }

    /** @return the share of unlabelled rows so far whose weight was at least 0.5 (quantity) */
    public double keptShare() {
        return seen == 0 ? 0 : kept / (double) seen;
    }

    /** @return the share of kept rows whose pseudo-label was right (quality) */
    public double pseudoLabelPrecision() {
        return kept == 0 ? 0 : correct / (double) kept;
    }

    /** Clears the quantity and quality counters. */
    public void resetStatistics() {
        kept = 0;
        correct = 0;
        seen = 0;
    }

    /**
     * FixMatch's weak augmentation for {@code [channels, height, width]} images: a random shift
     * of up to {@code pad} pixels and a horizontal flip.
     *
     * @param channels channels
     * @param height height
     * @param width width
     * @param pad the largest shift
     * @return the augmentation
     */
    public static Augmentation weak(int channels, int height, int width, int pad) {
        return Augmentation.cropAndFlip(channels, height, width, pad);
    }

    /**
     * FixMatch's strong augmentation: the weak one, then RandAugment with two operations, then
     * Cutout (random erasing of one patch).
     *
     * @param channels channels
     * @param height height
     * @param width width
     * @param magnitude the RandAugment magnitude, 0 to 10
     * @return the augmentation
     */
    public static Augmentation strong(int channels, int height, int width, double magnitude) {
        Augmentation ra = Augmentation.randAugment(channels, height, width, 2, magnitude);
        Augmentation cut = Augmentation.randomErasing(channels, height, width, 1.0, 0.05, 0.2);
        return weak(channels, height, width, 1).then(ra).then(cut);
    }

    /**
     * Applies a per-sample augmentation to a batch, leaving the input unchanged.
     *
     * @param x {@code [batch, ...]} inputs
     * @param augmentation the augmentation
     * @param random the random source
     * @return the augmented copy
     */
    public static Tensor augment(Tensor x, Augmentation augmentation, SplittableRandom random) {
        float[] v = x.toArray();
        int n = x.shape().dim(0);
        int per = v.length / n;
        for (int i = 0; i < n; i++) {
            augmentation.apply(v, i * per, random);
        }
        return Tensor.of(x.shape(), v);
    }
}
