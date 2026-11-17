package io.github.zzuegg.brainiac.experimental.train;

import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RidgeRegressionTest {
    @Test
    void recoversALinearMapAndShrinksWithLambda() {
        SplittableRandom r = new SplittableRandom(1);
        double[][] truth = {{1, -2}, {0.5, 3}, {-1, 0}};
        double[][] x = new double[200][3];
        double[][] y = new double[200][];
        for (int i = 0; i < 200; i++) {
            for (int j = 0; j < 3; j++) {
                x[i][j] = r.nextGaussian();
            }
            y[i] = RidgeRegression.predict(truth, x[i]);
        }
        double[][] w = RidgeRegression.fit(x, y, 0);
        for (int i = 0; i < 3; i++) {
            for (int k = 0; k < 2; k++) {
                assertEquals(truth[i][k], w[i][k], 1e-9);
            }
        }
        double[][] shrunk = RidgeRegression.fit(x, y, 1000);
        assertTrue(Math.abs(shrunk[1][1]) < 0.5 * Math.abs(truth[1][1]));
    }

    @Test
    void singularDesignNeedsAPenalty() {
        double[][] x = {{1, 1}, {2, 2}};
        double[][] y = {{1}, {2}};
        assertThrows(ArithmeticException.class, () -> RidgeRegression.fit(x, y, 0));
        double[][] w = RidgeRegression.fit(x, y, 1e-6);
        assertEquals(0.5, w[0][0], 1e-4);   // minimum-norm split between the two equal columns
        assertEquals(0.5, w[1][0], 1e-4);
    }
}
