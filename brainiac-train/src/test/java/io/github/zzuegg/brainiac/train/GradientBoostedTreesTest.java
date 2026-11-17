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
}
