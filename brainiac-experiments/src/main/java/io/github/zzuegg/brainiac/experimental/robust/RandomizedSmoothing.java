package io.github.zzuegg.brainiac.experimental.robust;

import io.github.zzuegg.brainiac.experimental.ml.NormalTails;
import io.github.zzuegg.brainiac.experimental.ml.ProbabilisticProgram;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Randomized smoothing, certified {@code ℓ2} robustness for any classifier (Cohen, Rosenfeld
 * and Kolter, ICML 2019, <a href="https://arxiv.org/abs/1902.02918">arXiv:1902.02918</a>).
 * The smoothed classifier predicts the class the base classifier returns most often under
 * Gaussian noise, {@code g(x) = argmax_c P(f(x + η) = c)}, {@code η ~ N(0, σ² I)}. If the top
 * class has probability at least {@code p_A > 1/2}, {@code g} is constant within the
 * {@code ℓ2} radius {@code σ Φ⁻¹(p_A)}. {@link #certify} follows the paper's CERTIFY: a first
 * batch of {@code n0} noisy samples picks the class, a second of {@code n} samples gives a
 * one-sided Clopper-Pearson lower bound on its probability at confidence {@code 1 - α}, and the
 * input is abstained on when that bound is not above one half. {@link #predict} is the paper's
 * PREDICT, a binomial test between the two most frequent classes.
 */
public final class RandomizedSmoothing {

    /**
     * A certificate.
     *
     * @param label the predicted class, or -1 for abstention
     * @param radius the certified {@code ℓ2} radius (0 when abstaining)
     * @param lowerBound the lower confidence bound on the top class's probability
     */
    public record Certificate(int label, double radius, double lowerBound) {
    }

    private final Function<Tensor, int[]> classifier;
    private final float sigma;
    private final int batch;

    private RandomizedSmoothing(Function<Tensor, int[]> classifier, float sigma, int batch) {
        this.classifier = Objects.requireNonNull(classifier);
        this.sigma = sigma;
        this.batch = batch;
    }

    /**
     * Smooths a base classifier.
     *
     * @param classifier maps a batch of inputs {@code [b, ...]} to predicted classes
     * @param sigma the noise level
     * @param batch noisy copies classified per call
     * @return the smoothed classifier
     */
    public static RandomizedSmoothing of(Function<Tensor, int[]> classifier, float sigma, int batch) {
        return new RandomizedSmoothing(classifier, sigma, batch);
    }

    /** @return the noise level */
    public float sigma() {
        return sigma;
    }

    /**
     * Class counts of the base classifier under noise.
     *
     * @param x one input, any shape
     * @param samples noisy copies
     * @param classes the number of classes
     * @param random the noise source
     * @return the counts
     */
    public int[] counts(Tensor x, int samples, int classes, SplittableRandom random) {
        int[] counts = new int[classes];
        float[] base = x.toArray();
        int d = base.length;
        int[] dims = x.shape().toArray();
        int left = samples;
        while (left > 0) {
            int b = Math.min(batch, left);
            float[] v = new float[b * d];
            for (int i = 0; i < b; i++) {
                for (int j = 0; j < d; j++) {
                    v[i * d + j] = base[j] + (float) (random.nextGaussian() * sigma);
                }
            }
            int[] shape = new int[dims.length + 1];
            shape[0] = b;
            System.arraycopy(dims, 0, shape, 1, dims.length);
            for (int c : classifier.apply(Tensor.of(Shape.of(shape), v))) {
                counts[c]++;
            }
            left -= b;
        }
        return counts;
    }

    /**
     * CERTIFY.
     *
     * @param x one input
     * @param classes the number of classes
     * @param n0 samples for picking the class
     * @param n samples for the bound
     * @param alpha the failure probability
     * @param random the noise source
     * @return the certificate
     */
    public Certificate certify(Tensor x, int classes, int n0, int n, double alpha, SplittableRandom random) {
        int[] c0 = counts(x, n0, classes, random);
        int top = argmax(c0);
        int[] c = counts(x, n, classes, random);
        double pA = clopperPearsonLower(c[top], n, alpha);
        if (pA <= 0.5) {
            return new Certificate(-1, 0, pA);
        }
        return new Certificate(top, sigma * NormalTails.quantile(pA), pA);
    }

    /**
     * PREDICT: the top class if a two-sided binomial test against the runner-up rejects at
     * level {@code α}, else abstain.
     *
     * @param x one input
     * @param classes the number of classes
     * @param n samples
     * @param alpha the failure probability
     * @param random the noise source
     * @return the class, or -1
     */
    public int predict(Tensor x, int classes, int n, double alpha, SplittableRandom random) {
        int[] c = counts(x, n, classes, random);
        int a = argmax(c);
        int b = -1;
        for (int k = 0; k < classes; k++) {
            if (k != a && (b < 0 || c[k] > c[b])) {
                b = k;
            }
        }
        int na = c[a];
        int total = na + (b < 0 ? 0 : c[b]);
        double p = 2 * binomialUpperTail(na, total, 0.5);
        return p <= alpha ? a : -1;
    }

    /**
     * The one-sided Clopper-Pearson lower confidence bound on a binomial proportion: the
     * {@code p} with {@code P(Bin(n, p) >= k) = α}, found by bisection.
     *
     * @param k successes
     * @param n trials
     * @param alpha the failure probability
     * @return the lower bound
     */
    public static double clopperPearsonLower(int k, int n, double alpha) {
        if (k == 0) {
            return 0;
        }
        double lo = 0;
        double hi = (double) k / n;
        for (int i = 0; i < 60; i++) {
            double mid = (lo + hi) / 2;
            if (binomialUpperTail(k, n, mid) > alpha) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        return lo;
    }

    /**
     * {@code P(Bin(n, p) >= k)}, summed in log space.
     *
     * @param k the threshold
     * @param n trials
     * @param p the success probability
     * @return the tail probability
     */
    public static double binomialUpperTail(int k, int n, double p) {
        if (k <= 0) {
            return 1;
        }
        if (p <= 0) {
            return 0;
        }
        if (p >= 1) {
            return 1;
        }
        double lgN = ProbabilisticProgram.logGamma(n + 1.0);
        double lp = Math.log(p);
        double lq = Math.log1p(-p);
        double max = Double.NEGATIVE_INFINITY;
        double[] terms = new double[n - k + 1];
        for (int i = k; i <= n; i++) {
            double t = lgN - ProbabilisticProgram.logGamma(i + 1.0) - ProbabilisticProgram.logGamma(n - i + 1.0)
                    + i * lp + (n - i) * lq;
            terms[i - k] = t;
            max = Math.max(max, t);
        }
        double s = 0;
        for (double t : terms) {
            s += Math.exp(t - max);
        }
        return Math.min(1, Math.exp(max + Math.log(s)));
    }

    private static int argmax(int[] c) {
        int b = 0;
        for (int i = 1; i < c.length; i++) {
            b = c[i] > c[b] ? i : b;
        }
        return b;
    }
}
