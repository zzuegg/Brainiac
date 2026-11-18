package io.github.zzuegg.brainiac.experimental.labels;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.semisup.SoftTargets;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Long-tail training by class margins and by decoupling representation from classifier.
 *
 * <ul>
 *   <li>{@link #ldam} (Cao et al., NeurIPS 2019, <a href="https://arxiv.org/abs/1906.07413">arXiv:1906.07413</a>):
 *       label-distribution-aware margins {@code m_y = C / n_y^(1/4)} subtracted from the true
 *       class's logit inside the cross-entropy, scaled so the largest margin is
 *       {@code maxMargin}; rare classes get the widest margin.</li>
 *   <li>{@link #drwWeights}: deferred re-weighting, class-balanced weights from the effective
 *       number of samples {@code (1 - beta) / (1 - beta^n_y)} (Cui et al., 2019), switched on only
 *       in the last training stage ({@link #weightedCrossEntropy}).</li>
 *   <li>Decoupling (Kang et al., ICLR 2020, <a href="https://arxiv.org/abs/1910.09217">arXiv:1910.09217</a>):
 *       train the features with plain instance sampling, then re-train only the classifier with
 *       class-balanced sampling ({@link #classBalanced}, cRT), or rescale its weight columns by
 *       {@code ||w_y||^-tau} with no training at all ({@link #tauNormalize}).</li>
 * </ul>
 */
public final class MarginAndDecoupling {

    private MarginAndDecoupling() {
    }

    /**
     * The LDAM loss.
     *
     * @param counts training examples per class
     * @param maxMargin the margin of the rarest class
     * @param scale multiplies the margined logits before the softmax (30 with normalised
     *     features in the paper; 1 for raw logits)
     * @param classWeights per-class loss weights (DRW), or null
     * @return the loss over logits and class ids
     */
    public static Loss ldam(int[] counts, double maxMargin, float scale, double[] classWeights) {
        float[] m = new float[counts.length];
        double max = 0;
        for (int k = 0; k < m.length; k++) {
            max = Math.max(max, Math.pow(counts[k], -0.25));
        }
        for (int k = 0; k < m.length; k++) {
            m[k] = (float) (maxMargin * Math.pow(counts[k], -0.25) / max);
        }
        Tensor margins = Tensor.of(Shape.of(1, m.length), m);
        return (logits, ids) -> {
            Tensor oh = RobustLosses.oneHot(logits, ids);
            Var shifted = logits.sub(oh.mul(margins)).mul(scale);
            return weighted(shifted, ids, oh, classWeights);
        };
    }

    /**
     * Class-balanced weights from the effective number of samples, normalised to sum to the
     * number of classes.
     *
     * @param counts training examples per class
     * @param beta the effective-number parameter, such as 0.999 (0 gives all ones)
     * @return the weights
     */
    public static double[] drwWeights(int[] counts, double beta) {
        double[] w = new double[counts.length];
        double sum = 0;
        for (int k = 0; k < w.length; k++) {
            w[k] = (1 - beta) / (1 - Math.pow(beta, counts[k]));
            sum += w[k];
        }
        for (int k = 0; k < w.length; k++) {
            w[k] *= w.length / sum;
        }
        return w;
    }

    /**
     * Class-weighted cross-entropy (DRW's second stage, or plain inverse-frequency reweighting).
     *
     * @param classWeights per-class weights
     * @return the loss
     */
    public static Loss weightedCrossEntropy(double[] classWeights) {
        return (logits, ids) -> weighted(logits, ids, RobustLosses.oneHot(logits, ids), classWeights);
    }

    private static Var weighted(Var logits, Tensor ids, Tensor oh, double[] classWeights) {
        Var perRow = SoftTargets.logSoftmax(logits).mul(oh).sum(-1, false).neg();
        if (classWeights == null) {
            return perRow.mean();
        }
        float[] w = new float[classWeights.length];
        for (int k = 0; k < w.length; k++) {
            w[k] = (float) classWeights[k];
        }
        Tensor rowWeights = oh.mul(Tensor.of(Shape.of(1, w.length), w)).sum(-1, false);
        return perRow.mul(rowWeights).sum().div(rowWeights.sum());
    }

    /**
     * Draws a class-balanced minibatch: a uniformly random class, then a random example of it.
     *
     * @param labels the training labels
     * @param classes the number of classes
     * @param batch the batch size
     * @param random the random source
     * @return example indices
     */
    public static int[] classBalanced(int[] labels, int classes, int batch, SplittableRandom random) {
        List<List<Integer>> byClass = new ArrayList<>();
        for (int k = 0; k < classes; k++) {
            byClass.add(new ArrayList<>());
        }
        for (int i = 0; i < labels.length; i++) {
            byClass.get(labels[i]).add(i);
        }
        int[] out = new int[batch];
        for (int i = 0; i < batch; i++) {
            List<Integer> c;
            do {
                c = byClass.get(random.nextInt(classes));
            } while (c.isEmpty());
            out[i] = c.get(random.nextInt(c.size()));
        }
        return out;
    }

    /**
     * Tau-normalisation of a classifier's weight columns.
     *
     * @param weight {@code [d, classes]}, as a core {@code Dense} layer stores it
     * @param tau the power, 0 for none and 1 for full normalisation
     * @return {@code w_k / ||w_k||^tau}
     */
    public static Tensor tauNormalize(Tensor weight, double tau) {
        return weight.div(weight.square().sum(0, true).sqrt().add(1e-12f).pow((float) tau));
    }
}
