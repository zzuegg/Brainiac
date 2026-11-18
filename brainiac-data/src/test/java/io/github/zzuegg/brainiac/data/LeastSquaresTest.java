package io.github.zzuegg.brainiac.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.zzuegg.brainiac.tensor.Distributions;
import org.junit.jupiter.api.Test;

/** OLS estimates, standard errors and tests, against NumPy. */
class LeastSquaresTest {

    private static final double[][] X = {{1, 2}, {2, 1}, {3, 4}, {4, 3}, {5, 6}, {6, 5}, {7, 8}};
    private static final double[] Y = {3.1, 3.9, 8.2, 8.8, 13.1, 14.2, 18.9};

    @Test
    void matchesNumpy() {
        LeastSquares.Fit fit = LeastSquares.ols(X, Y, true);
        assertArrayEquals(new double[] {-0.51845238095238, 1.6744047619047582, 0.9291666666666709}, fit.coefficients(), 1e-10);
        assertArrayEquals(new double[] {0.2047575788579587, 0.1030140310325207, 0.09232716585900133}, fit.standardErrors(), 1e-10);
        assertArrayEquals(new double[] {-2.5320302371422008, 16.254142713589783, 10.063849117665544}, fit.tValues(), 1e-8);
        assertEquals(0.9988068159726272, fit.rSquared(), 1e-12);
        assertEquals(4, fit.residualDegreesOfFreedom());
        assertEquals(Distributions.studentTTwoSided(-2.5320302371422008, 4), fit.pValues()[0], 1e-12);
        assertEquals(-0.51845238095238 + 1.6744047619047582 * 2 + 0.9291666666666709 * 3, fit.predict(new double[] {2, 3}, true), 1e-10);
    }

    @Test
    void collinearPredictorsAreRefused() {
        double[][] x = {{1, 2}, {2, 4}, {3, 6}, {4, 8}};
        assertThrows(IllegalArgumentException.class, () -> LeastSquares.ols(x, new double[] {1, 2, 3, 4}, true));
    }

    @Test
    void weightedAndRidgeMatchNumpy() {
        double[] w = {1, 2, 0.5, 1, 3, 1, 2};
        LeastSquares.Fit fit = LeastSquares.wls(X, Y, w, true);
        assertArrayEquals(new double[] {-0.5572025052192053, 1.7321503131523974, 0.8778705636743244}, fit.coefficients(), 1e-10);
        assertArrayEquals(new double[] {0.27046956322863136, 0.1502171100615605, 0.12018476549666553}, fit.standardErrors(), 1e-10);
        assertEquals(0.9984071112452304, fit.rSquared(), 1e-12);
        assertArrayEquals(new double[] {-0.1752705805182, 1.4801574286651347, 1.0331912102328642},
                LeastSquares.ridge(X, Y, w, 2, true), 1e-10);
        assertArrayEquals(new double[] {1.4354942233632837, 1.0406931964056507}, LeastSquares.ridge(X, Y, null, 2, false), 1e-10);
        assertArrayEquals(LeastSquares.ols(X, Y, true).coefficients(), LeastSquares.ridge(X, Y, null, 0, true), 1e-9);
    }
}
