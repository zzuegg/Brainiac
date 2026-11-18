package io.github.zzuegg.brainiac.experimental.semisup;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.LabelCorruption;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * MixMatch (Berthelot et al., NeurIPS 2019, <a href="https://arxiv.org/abs/1905.02249">arXiv:1905.02249</a>).
 * It guesses a label for each unlabelled input by averaging the model's predictions over
 * {@code K} augmentations and sharpening the average with temperature {@code T} (entropy
 * minimisation), then shuffles the labelled and unlabelled batches together and applies MixUp
 * with {@code lambda' = max(lambda, 1 - lambda)}, so each mixed row stays closer to its own
 * input. The loss is cross-entropy on the mixed labelled rows plus {@code lambdaU} times the
 * squared error on the mixed unlabelled rows.
 *
 * <p>Unlike {@code BatchMixing.mixup}, which mixes one batch with a shuffled copy of itself and a
 * free {@code lambda}, MixMatch mixes across the two sets with the larger weight on the row's own
 * input, so the first rows stay labelled-dominated.
 */
public final class MixMatch {

    private final double temperature;
    private final float alpha;
    private final float lambdaU;
    private final int rampSteps;
    private final boolean sharpen;
    private final boolean mixUp;

    /**
     * Creates the method.
     *
     * @param temperature the sharpening temperature, 0.5 in the paper
     * @param alpha the Beta(alpha, alpha) parameter of MixUp, 0.75 in the paper
     * @param lambdaU the unlabelled weight after ramp-up
     * @param rampSteps the linear ramp-up of {@code lambdaU}, in steps
     * @param sharpen whether to sharpen guesses (off for the ablation)
     * @param mixUp whether to mix (off for the ablation)
     */
    public MixMatch(double temperature, float alpha, float lambdaU, int rampSteps, boolean sharpen, boolean mixUp) {
        this.temperature = temperature;
        this.alpha = alpha;
        this.lambdaU = lambdaU;
        this.rampSteps = rampSteps;
        this.sharpen = sharpen;
        this.mixUp = mixUp;
    }

    /**
     * Guesses labels: the mean softmax over the augmentations, sharpened.
     *
     * @param logitsPerAugmentation the model's logits on each of the {@code K} augmented copies
     * @return the guessed label rows
     */
    public float[][] guess(List<Tensor> logitsPerAugmentation) {
        float[][] mean = null;
        for (Tensor l : logitsPerAugmentation) {
            float[][] p = SoftTargets.probabilities(l, 1f);
            if (mean == null) {
                mean = new float[p.length][p[0].length];
            }
            for (int i = 0; i < p.length; i++) {
                for (int c = 0; c < p[i].length; c++) {
                    mean[i][c] += p[i][c] / logitsPerAugmentation.size();
                }
            }
        }
        return sharpen ? SoftTargets.sharpen(mean, temperature) : mean;
    }

    /**
     * A mixed batch.
     *
     * @param x the mixed inputs, the labelled rows first
     * @param y the mixed target rows
     * @param labelled how many leading rows are labelled
     */
    public record Mixed(Tensor x, Tensor y, int labelled) {
    }

    /**
     * Mixes a labelled and an unlabelled batch.
     *
     * @param xl labelled inputs (augmented)
     * @param yl their one-hot (or soft) targets
     * @param xu unlabelled inputs (one augmentation)
     * @param qu their guessed labels
     * @param random the random source
     * @return the mixed batch
     */
    public Mixed mix(Tensor xl, float[][] yl, Tensor xu, float[][] qu, SplittableRandom random) {
        Tensor x = Tensor.concat(0, xl, xu);
        float[][] y = new float[yl.length + qu.length][];
        System.arraycopy(yl, 0, y, 0, yl.length);
        System.arraycopy(qu, 0, y, yl.length, qu.length);
        if (!mixUp) {
            return new Mixed(x, SoftTargets.tensor(y), yl.length);
        }
        int n = y.length;
        int[] perm = LabelCorruption.permutation(n, random);
        float lam = (float) beta(alpha, random);
        lam = Math.max(lam, 1 - lam);
        Tensor xs = SoftTargets.rows(x, perm);
        float[][] ys = new float[n][];
        for (int i = 0; i < n; i++) {
            ys[i] = new float[y[i].length];
            for (int c = 0; c < y[i].length; c++) {
                ys[i][c] = lam * y[i][c] + (1 - lam) * y[perm[i]][c];
            }
        }
        return new Mixed(x.mul(lam).add(xs.mul(1 - lam)), SoftTargets.tensor(ys), yl.length);
    }

    /**
     * The MixMatch loss on a mixed batch.
     *
     * @param logits the model's logits on {@link Mixed#x()}
     * @param mixed the mixed batch
     * @param step the training step, for the ramp-up
     * @return {@code CE(labelled rows) + lambdaU * ramp * mean squared error (unlabelled rows)}
     */
    public Var loss(Var logits, Mixed mixed, int step) {
        int nl = mixed.labelled();
        int n = logits.shape().dim(0);
        int classes = logits.shape().dim(1);
        Var lx = SoftTargets.softCrossEntropy(logits.slice(0, 0, nl), mixed.y().slice(0, 0, nl), null, nl);
        if (n == nl) {
            return lx;
        }
        Var pu = SoftTargets.softmax(logits.slice(0, nl, n));
        Var lu = pu.sub(mixed.y().slice(0, nl, n)).square().sum().mul(1f / ((n - nl) * classes));
        float ramp = rampSteps <= 0 ? 1 : (float) Math.min(1, step / (double) rampSteps);
        return lx.add(lu.mul(lambdaU * ramp));
    }

    /**
     * Draws from Beta(a, a) by two Gamma draws (Marsaglia-Tsang, with the {@code a < 1} boost).
     *
     * @param a the parameter
     * @param r the random source
     * @return a value in {@code [0, 1]}
     */
    static double beta(double a, SplittableRandom r) {
        double x = gamma(a, r);
        double y = gamma(a, r);
        return x / (x + y);
    }

    private static double gamma(double a, SplittableRandom r) {
        if (a < 1) {
            return gamma(a + 1, r) * Math.pow(r.nextDouble(), 1 / a);
        }
        double d = a - 1.0 / 3;
        double c = 1 / Math.sqrt(9 * d);
        while (true) {
            double z = r.nextGaussian();
            double v = 1 + c * z;
            if (v <= 0) {
                continue;
            }
            v = v * v * v;
            double u = r.nextDouble();
            if (Math.log(u) < 0.5 * z * z + d - d * v + d * Math.log(v)) {
                return d * v;
            }
        }
    }
}
