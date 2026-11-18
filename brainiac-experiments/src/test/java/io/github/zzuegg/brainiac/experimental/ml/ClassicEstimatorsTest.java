package io.github.zzuegg.brainiac.experimental.ml;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.LinearChainCrf;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class ClassicEstimatorsTest {

    private static double[][] randomPoints(int n, int d, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][d];
        for (double[] row : x) {
            for (int j = 0; j < d; j++) {
                row[j] = r.nextGaussian();
            }
        }
        return x;
    }

    @Test
    void kernelGradientsMatchFiniteDifferences() {
        double[] a = {0.3, -0.7};
        double[] b = {-0.2, 0.4};
        double worst = 0;
        for (StationaryKernel.Family f : StationaryKernel.Family.values()) {
            StationaryKernel k = new StationaryKernel(f, 0.8, 1.3);
            double eps = 1e-6;
            double fd = (k.with(0.8 * Math.exp(eps), 1.3).value(a, b) - k.with(0.8 * Math.exp(-eps), 1.3).value(a, b))
                    / (2 * eps);
            worst = Math.max(worst, Math.abs(fd - k.dLogLengthscale(a, b)));
            double[] g = k.gradient(a, b);
            for (int i = 0; i < 2; i++) {
                double[] ap = a.clone();
                double[] am = a.clone();
                ap[i] += eps;
                am[i] -= eps;
                worst = Math.max(worst, Math.abs((k.value(ap, b) - k.value(am, b)) / (2 * eps) - g[i]));
            }
        }
        System.out.printf("stationary kernel gradients: max error vs finite differences %.2e%n", worst);
        assertTrue(worst < 1e-6, "max error " + worst);
    }

    @Test
    void marginalLikelihoodGradientMatchesFiniteDifferences() {
        double[][] x = randomPoints(25, 2, 1);
        double[] y = new double[25];
        for (int i = 0; i < 25; i++) {
            y[i] = Math.sin(2 * x[i][0]) + 0.1 * x[i][1];
        }
        StationaryKernel k = StationaryKernel.matern52(0.9, 1.1);
        double noise = 0.05;
        double[] g = GaussianProcess.fit(x, y, k, noise).logMarginalGradient();
        double eps = 1e-5;
        double[] fd = {
            (GaussianProcess.fit(x, y, k.with(0.9 * Math.exp(eps), 1.1), noise).logMarginalLikelihood()
                    - GaussianProcess.fit(x, y, k.with(0.9 * Math.exp(-eps), 1.1), noise).logMarginalLikelihood()) / (2 * eps),
            (GaussianProcess.fit(x, y, k.with(0.9, 1.1 * Math.exp(eps)), noise).logMarginalLikelihood()
                    - GaussianProcess.fit(x, y, k.with(0.9, 1.1 * Math.exp(-eps)), noise).logMarginalLikelihood()) / (2 * eps),
            (GaussianProcess.fit(x, y, k, noise * Math.exp(eps)).logMarginalLikelihood()
                    - GaussianProcess.fit(x, y, k, noise * Math.exp(-eps)).logMarginalLikelihood()) / (2 * eps),
        };
        System.out.printf("GP log-evidence gradient: analytic %.4f %.4f %.4f, finite differences %.4f %.4f %.4f%n",
                g[0], g[1], g[2], fd[0], fd[1], fd[2]);
        assertArrayEquals(fd, g, 1e-4);
    }

    @Test
    void randomFourierFeaturesApproximateTheRbfKernelAndDropIntoAModel() {
        double[][] x = randomPoints(40, 3, 2);
        KernelFunction rbf = KernelFunction.rbf(0.5);
        double err100 = 0;
        double err2000 = 0;
        RandomFourierFeatures small = new RandomFourierFeatures(3, 100, 0.5, new SplittableRandom(3));
        RandomFourierFeatures big = new RandomFourierFeatures(3, 2000, 0.5, new SplittableRandom(3));
        for (int i = 0; i < 40; i++) {
            for (int j = 0; j < 40; j++) {
                double exact = rbf.value(x[i], x[j]);
                err100 += Math.abs(small.approximateKernel(x[i], x[j]) - exact) / 1600;
                err2000 += Math.abs(big.approximateKernel(x[i], x[j]) - exact) / 1600;
            }
        }
        System.out.printf("RFF kernel error: D=100 %.4f, D=2000 %.4f%n", err100, err2000);
        assertTrue(err2000 < err100 / 2 && err2000 < 0.03, err100 + " " + err2000);
        Model m = Model.sequential().input(3).add(RandomFourierFeatures.spec(64, 0.5)).dense(2).seed(4).build();
        Tensor out = m.forward(Var.of(Tensor.of(MlMatrices.toFloat(x))), false).value();
        assertEquals(Shape.of(40, 2), out.shape());
    }

    @Test
    void metricsBehave() {
        int[] a = {0, 0, 1, 1, 2, 2};
        int[] relabelled = {5, 5, 3, 3, 9, 9};
        assertEquals(1, MlMetrics.adjustedRandIndex(a, relabelled), 1e-12);
        double[][] x = randomPoints(30, 4, 5);
        assertEquals(1, MlMetrics.trustworthiness(x, x, 5), 1e-12);
        assertEquals(1, MlMetrics.auroc(new double[] {0.1, 0.2, 0.8, 0.9}, new boolean[] {false, false, true, true}));
        assertEquals(0.5, MlMetrics.normalCdf(0), 1e-7);
    }

    @Test
    void kMeansPlusPlusSeedingBeatsRandomSeedingOnAverage() {
        double[][] centres = new double[12][];
        for (int c = 0; c < 12; c++) {
            centres[c] = new double[] {(c % 4) * 3.0, (c / 4) * 3.0};
        }
        SplittableRandom r = new SplittableRandom(6);
        double[][] x = new double[600][];
        int[] y = new int[600];
        for (int i = 0; i < 600; i++) {
            y[i] = i % 12;
            x[i] = new double[] {centres[y[i]][0] + 0.3 * r.nextGaussian(), centres[y[i]][1] + 0.3 * r.nextGaussian()};
        }
        double random = 0;
        double plus = 0;
        for (long seed = 0; seed < 10; seed++) {
            random += MlMetrics.adjustedRandIndex(y, KMeans.fit(x, 12, KMeans.Init.RANDOM, 1, 100, seed).labels()) / 10;
            plus += MlMetrics.adjustedRandIndex(y, KMeans.fit(x, 12, KMeans.Init.PLUS_PLUS, 1, 100, seed).labels()) / 10;
        }
        System.out.printf("12 blobs, mean ARI over 10 seeds: k-means++ %.3f vs random init %.3f%n", plus, random);
        assertTrue(plus > random, plus + " vs " + random);
    }

    @Test
    void gaussianMixtureBicPicksTheTrueComponentCount() {
        SplittableRandom r = new SplittableRandom(7);
        double[][] x = new double[450][];
        for (int i = 0; i < 450; i++) {
            int c = i % 3;
            x[i] = new double[] {4 * c + r.nextGaussian(), (c == 1 ? 3 : 0) + 0.5 * r.nextGaussian()};
        }
        GaussianMixture best = GaussianMixture.selectByBic(x, 6, 100, 8);
        System.out.printf("GMM BIC selection: %d components (true 3)%n", best.components());
        assertEquals(3, best.components());
    }

    @Test
    void hmmForwardMatchesBruteForceEnumeration() {
        HiddenMarkovModel m = HiddenMarkovModel.discrete(new double[] {0.6, 0.4}, new double[][] {{0.7, 0.3}, {0.2, 0.8}},
                new double[][] {{0.5, 0.4, 0.1}, {0.1, 0.3, 0.6}});
        double[] obs = {0, 2, 1, 2};
        double total = 0;
        for (int path = 0; path < 16; path++) {
            double p = 1;
            int prev = -1;
            for (int t = 0; t < 4; t++) {
                int s = (path >> t) & 1;
                p *= (t == 0 ? new double[] {0.6, 0.4}[s] : m.transition()[prev][s]) * m.emission(s, obs[t]);
                prev = s;
            }
            total += p;
        }
        assertEquals(Math.log(total), m.logLikelihood(obs), 1e-12);
        double[][] post = m.posteriors(obs);
        assertEquals(1, post[2][0] + post[2][1], 1e-12);
    }

    @Test
    void crfLogPartitionMatchesBruteForce() {
        LinearChainCrf crf = new LinearChainCrf(3, new SplittableRandom(9));
        SplittableRandom r = new SplittableRandom(10);
        int steps = 4;
        float[] e = new float[steps * 3];
        for (int i = 0; i < e.length; i++) {
            e[i] = (float) r.nextGaussian();
        }
        double[][] tr = crf.transitionScores();
        double z = 0;
        double gold = 0;
        int[] goldPath = {0, 2, 2, 1};
        for (int code = 0; code < 81; code++) {
            int[] p = {code % 3, code / 3 % 3, code / 9 % 3, code / 27};
            double s = 0;
            for (int t = 0; t < steps; t++) {
                s += e[t * 3 + p[t]] + (t > 0 ? tr[p[t - 1]][p[t]] : 0);
            }
            z += Math.exp(s);
            if (java.util.Arrays.equals(p, goldPath)) {
                gold = s;
            }
        }
        double nll = crf.negativeLogLikelihood(Var.of(Tensor.of(Shape.of(1, steps, 3), e)),
                Tensor.of(Shape.of(1, steps), 0, 2, 2, 1)).value().toArray()[0];
        assertEquals(Math.log(z) - gold, nll, 1e-4);
        int[] best = crf.decode(Tensor.of(Shape.of(1, steps, 3), e))[0];
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int code = 0; code < 81; code++) {
            int[] p = {code % 3, code / 3 % 3, code / 9 % 3, code / 27};
            double s = 0;
            for (int t = 0; t < steps; t++) {
                s += e[t * 3 + p[t]] + (t > 0 ? tr[p[t - 1]][p[t]] : 0);
            }
            bestScore = Math.max(bestScore, s);
        }
        double decoded = 0;
        for (int t = 0; t < steps; t++) {
            decoded += e[t * 3 + best[t]] + (t > 0 ? tr[best[t - 1]][best[t]] : 0);
        }
        assertEquals(bestScore, decoded, 1e-6);
        assertEquals(3, crf.parameters().size());
    }

    @Test
    void variableEliminationMatchesEnumeration() {
        FactorGraph g = FactorGraph.ising(new double[][] {{0, 0.5, -0.3}, {0.5, 0, 0.8}, {-0.3, 0.8, 0}},
                new double[] {0.1, -0.2, 0.3}, 1);
        double[] up = new double[3];
        double z = 0;
        for (int code = 0; code < 8; code++) {
            int[] m = {2 * (code & 1) - 1, 2 * (code >> 1 & 1) - 1, 2 * (code >> 2 & 1) - 1};
            double e = 0.5 * m[0] * m[1] - 0.3 * m[0] * m[2] + 0.8 * m[1] * m[2] + 0.1 * m[0] - 0.2 * m[1] + 0.3 * m[2];
            double w = Math.exp(e);
            z += w;
            for (int i = 0; i < 3; i++) {
                up[i] += m[i] > 0 ? w : 0;
            }
        }
        double[][] exact = g.exactMarginals();
        for (int i = 0; i < 3; i++) {
            assertEquals(up[i] / z, exact[i][1], 1e-12);
        }
        assertEquals(3, g.variables());
    }
}
