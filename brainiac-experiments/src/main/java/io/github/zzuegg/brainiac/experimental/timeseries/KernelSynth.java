package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.experimental.ml.KernelFunction;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * KernelSynth (Ansari et al., Chronos, 2024, <a href="https://arxiv.org/abs/2403.07815">arXiv:2403.07815</a>):
 * synthetic training series drawn from Gaussian-process priors whose kernels are random
 * compositions. Up to {@code maxTerms} kernels are drawn from a bank (RBF at several
 * length-scales, periodic at several periods, linear trends, rational quadratic, constant and
 * white noise, all as {@link KernelFunction}s over time) and combined by random sums and
 * products; a series is one draw from the zero-mean GP through a Cholesky factor of the Gram
 * matrix ({@link MlMatrices#cholesky}). The variety of compositions covers trends,
 * seasonality, smooth and rough noise, so a model trained only on it can forecast unseen real
 * series zero-shot.
 */
public final class KernelSynth {

    private final List<KernelFunction> bank = new ArrayList<>();
    private final int maxTerms;

    /**
     * Creates the generator with a bank scaled to the series length.
     *
     * @param length the series length
     * @param maxTerms the largest number of kernels combined, for example 5
     */
    public KernelSynth(int length, int maxTerms) {
        this.maxTerms = maxTerms;
        for (double l : new double[] {0.02, 0.05, 0.1, 0.3}) {
            double gamma = 1 / (2 * l * l * length * length);
            bank.add(KernelFunction.rbf(gamma));
        }
        for (double p : new double[] {4, 7, 12, 24, 30, 52}) {
            bank.add(periodic(p, 1.0));
        }
        bank.add((a, b) -> 0.5 + a[0] * b[0] / (length * (double) length));         // linear trend
        bank.add((a, b) -> {
            double d2 = (a[0] - b[0]) * (a[0] - b[0]) / (0.1 * length * 0.1 * length);
            return Math.pow(1 + d2 / 2, -2);                                           // rational quadratic
        });
        bank.add((a, b) -> 1.0);                                                       // constant
        bank.add((a, b) -> a[0] == b[0] ? 0.1 : 0);                                    // white noise
    }

    /**
     * The exp-sine-squared periodic kernel.
     *
     * @param period the period in steps
     * @param lengthscale the smoothness within a period
     * @return the kernel over time points {@code [t]}
     */
    public static KernelFunction periodic(double period, double lengthscale) {
        return (a, b) -> {
            double s = Math.sin(Math.PI * Math.abs(a[0] - b[0]) / period);
            return Math.exp(-2 * s * s / (lengthscale * lengthscale));
        };
    }

    /**
     * A random composite kernel.
     *
     * @param random the source
     * @return the kernel
     */
    public KernelFunction randomKernel(SplittableRandom random) {
        int terms = 1 + random.nextInt(maxTerms);
        KernelFunction k = bank.get(random.nextInt(bank.size()));
        for (int i = 1; i < terms; i++) {
            KernelFunction a = k;
            KernelFunction b = bank.get(random.nextInt(bank.size()));
            k = random.nextBoolean() ? (x, y) -> a.value(x, y) + b.value(x, y) : (x, y) -> a.value(x, y) * b.value(x, y);
        }
        return k;
    }

    /**
     * Draws one series.
     *
     * @param length steps
     * @param random the source
     * @return a draw from a GP with a random composite kernel
     */
    public float[] sample(int length, SplittableRandom random) {
        KernelFunction k = randomKernel(random);
        double[][] t = new double[length][1];
        for (int i = 0; i < length; i++) {
            t[i][0] = i;
        }
        double[][] gram = k.gram(t);
        for (int i = 0; i < length; i++) {
            gram[i][i] += 1e-4 * (1 + Math.abs(gram[i][i]));
        }
        double[][] l = MlMatrices.cholesky(gram);
        double[] z = new double[length];
        for (int i = 0; i < length; i++) {
            z[i] = random.nextGaussian();
        }
        float[] y = new float[length];
        for (int i = 0; i < length; i++) {
            double s = 0;
            for (int j = 0; j <= i; j++) {
                s += l[i][j] * z[j];
            }
            y[i] = (float) s;
        }
        return y;
    }
}
