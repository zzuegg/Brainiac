package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class GradientBoostedTreesTest {

    @Test
    void regressionBeatsTheMeanOnANonlinearFunction() {
        SplittableRandom r = new SplittableRandom(3);
        int n = 600;
        double[][] x = new double[n][3];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i][0] = r.nextDouble(-3, 3);
            x[i][1] = r.nextDouble(-3, 3);
            x[i][2] = r.nextDouble(-3, 3);   // noise feature
            y[i] = Math.sin(x[i][0]) + (x[i][1] > 0 ? 1 : -1);
        }
        GradientBoostedTrees m = GradientBoostedTrees.fit(x, y, GradientBoostedTrees.Objective.REGRESSION,
                GradientBoostedTrees.Settings.defaults());
        double mean = java.util.Arrays.stream(y).average().orElse(0);
        double mse = 0;
        double base = 0;
        double[] p = m.predict(x);
        for (int i = 0; i < n; i++) {
            mse += (p[i] - y[i]) * (p[i] - y[i]);
            base += (mean - y[i]) * (mean - y[i]);
        }
        assertTrue(mse < 0.05 * base, mse + " vs " + base);
        assertEquals(100, m.size());
        double[] imp = m.featureImportance();
        assertTrue(imp[2] < 0.05, java.util.Arrays.toString(imp));
        assertEquals(1, imp[0] + imp[1] + imp[2], 1e-9);
    }

    @Test
    void binaryClassifiesAnXorPattern() {
        SplittableRandom r = new SplittableRandom(5);
        int n = 800;
        double[][] x = new double[n][2];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i][0] = r.nextDouble(-1, 1);
            x[i][1] = r.nextDouble(-1, 1);
            y[i] = x[i][0] * x[i][1] > 0 ? 1 : 0;
        }
        GradientBoostedTrees m = GradientBoostedTrees.fit(x, y, GradientBoostedTrees.Objective.BINARY,
                new GradientBoostedTrees.Settings(60, 3, 0.3, 1, 1e-3, 0.8, 32, 7));
        int right = 0;
        for (int i = 0; i < n; i++) {
            double p = m.predict(x[i]);
            assertTrue(p >= 0 && p <= 1);
            right += (p > 0.5) == (y[i] == 1) ? 1 : 0;
        }
        assertTrue(right > 0.95 * n, right + " of " + n);
    }

    @Test
    void constantFeaturesGiveTheBase() {
        double[][] x = {{1}, {1}, {1}, {1}};
        double[] y = {1, 2, 3, 4};
        GradientBoostedTrees m = GradientBoostedTrees.fit(x, y, GradientBoostedTrees.Objective.REGRESSION,
                GradientBoostedTrees.Settings.defaults().withTrees(5));
        assertEquals(2.5, m.predict(new double[] {1}), 1e-9);
    }

    @Test
    void customPoissonLossLearnsRates() {
        GradientBoostedTrees.Loss poisson = new GradientBoostedTrees.Loss() {
            @Override
            public void derivatives(double[] raw, double[] y, double[] g, double[] h) {
                for (int i = 0; i < raw.length; i++) {
                    double rate = Math.exp(raw[i]);
                    g[i] = rate - y[i];
                    h[i] = rate;
                }
            }

            @Override
            public double base(double[] y) {
                return Math.log(java.util.Arrays.stream(y).average().orElse(1));
            }

            @Override
            public double link(double raw) {
                return Math.exp(raw);
            }
        };
        SplittableRandom r = new SplittableRandom(5);
        int n = 500;
        double[][] x = new double[n][2];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i][0] = r.nextDouble(-2, 2);
            x[i][1] = r.nextDouble(-2, 2);
            y[i] = Math.exp(0.8 * x[i][0]);
        }
        GradientBoostedTrees m = GradientBoostedTrees.fit(x, y, poisson,
                GradientBoostedTrees.Settings.defaults().withTrees(200));
        double worst = 0;
        for (int i = 0; i < n; i++) {
            worst = Math.max(worst, Math.abs(m.predict(x[i]) / y[i] - 1));
            assertEquals(Math.log(m.predict(x[i])), m.raw(x[i]), 1e-9);
        }
        assertTrue(worst < 0.15, "largest relative error " + worst);
    }

    @Test
    void weightTwoActsAsTwoCopies() {
        SplittableRandom r = new SplittableRandom(9);
        int n = 120;
        double[][] x = new double[n][2];
        double[] y = new double[n];
        double[] w = new double[n];
        java.util.List<double[]> dupX = new java.util.ArrayList<>();
        java.util.List<Double> dupY = new java.util.ArrayList<>();
        for (int i = 0; i < n; i++) {
            x[i][0] = r.nextDouble(-2, 2);
            x[i][1] = r.nextDouble(-2, 2);
            y[i] = x[i][0] * x[i][1];
            w[i] = i % 3 == 0 ? 2 : 1;
            for (int c = 0; c < w[i]; c++) {
                dupX.add(x[i]);
                dupY.add(y[i]);
            }
        }
        // a regression loss starting at 0, so the unweighted base does not differ
        GradientBoostedTrees.Loss squared = (raw, t, g, h) -> {
            for (int i = 0; i < raw.length; i++) {
                g[i] = raw[i] - t[i];
                h[i] = 1;
            }
        };
        GradientBoostedTrees.Settings s = GradientBoostedTrees.Settings.defaults().withTrees(20);
        GradientBoostedTrees weighted = GradientBoostedTrees.fit(x, y, w, squared, s);
        GradientBoostedTrees copies = GradientBoostedTrees.fit(dupX.toArray(double[][]::new),
                dupY.stream().mapToDouble(Double::doubleValue).toArray(), squared, s);
        for (int i = 0; i < n; i++) {
            assertEquals(copies.predict(x[i]), weighted.predict(x[i]), 1e-9);
        }
    }

    @Test
    void aMarginOfTheBaseGivesTheSameTreesAndObjectivesAreLosses() {
        SplittableRandom r = new SplittableRandom(11);
        int n = 200;
        double[][] x = new double[n][2];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i][0] = r.nextDouble(-1, 1);
            x[i][1] = r.nextDouble(-1, 1);
            y[i] = r.nextDouble() < 1 / (1 + Math.exp(-3 * x[i][0] * x[i][1])) ? 1 : 0;
        }
        GradientBoostedTrees.Loss binary = GradientBoostedTrees.Objective.BINARY.loss();
        GradientBoostedTrees.Settings s = GradientBoostedTrees.Settings.defaults().withTrees(20);
        GradientBoostedTrees plain = GradientBoostedTrees.fit(x, y, binary, s);
        double base = binary.base(y);
        double[] margin = new double[n];
        java.util.Arrays.fill(margin, base);
        GradientBoostedTrees shifted = GradientBoostedTrees.fit(x, y, null, margin, binary, s);
        for (int i = 0; i < n; i++) {
            assertEquals(plain.predict(x[i]), shifted.predict(x[i], base), 1e-12);
            assertEquals(plain.raw(x[i]), base + shifted.raw(x[i]), 1e-12);
        }
    }

    @Test
    void monotoneConstraintsHold() {
        // y rises with x0 but noise makes an unconstrained fit wiggle; x1 is noise
        SplittableRandom r = new SplittableRandom(13);
        int n = 300;
        double[][] x = new double[n][2];
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            x[i][0] = r.nextDouble(0, 1);
            x[i][1] = r.nextDouble(0, 1);
            y[i] = x[i][0] + 0.3 * Math.sin(12 * x[i][0]) + r.nextGaussian() * 0.2;
        }
        GradientBoostedTrees.Settings s = GradientBoostedTrees.Settings.defaults().withTrees(60);
        GradientBoostedTrees up = GradientBoostedTrees.fit(x, y, null, null, GradientBoostedTrees.Objective.REGRESSION.loss(),
                s, new int[] {1, 0});
        GradientBoostedTrees down = GradientBoostedTrees.fit(x, y, null, null,
                GradientBoostedTrees.Objective.REGRESSION.loss(), s, new int[] {-1, 0});
        GradientBoostedTrees free = GradientBoostedTrees.fit(x, y, GradientBoostedTrees.Objective.REGRESSION, s);
        boolean freeWiggles = false;
        for (double other : new double[] {0.1, 0.5, 0.9}) {
            double lastUp = Double.NEGATIVE_INFINITY;
            double lastDown = Double.POSITIVE_INFINITY;
            double lastFree = Double.NEGATIVE_INFINITY;
            for (int k = 0; k <= 100; k++) {
                double[] row = {k / 100.0, other};
                double u = up.predict(row);
                double d = down.predict(row);
                assertTrue(u >= lastUp - 1e-12, "not increasing at " + k);
                assertTrue(d <= lastDown + 1e-12, "not decreasing at " + k);
                freeWiggles |= free.predict(row) < lastFree - 1e-9;
                lastUp = u;
                lastDown = d;
                lastFree = free.predict(row);
            }
        }
        assertTrue(freeWiggles, "the data should make an unconstrained fit wiggle");
    }
}
