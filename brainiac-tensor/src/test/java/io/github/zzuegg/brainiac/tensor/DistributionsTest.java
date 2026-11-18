package io.github.zzuegg.brainiac.tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

/** Incomplete gamma and beta, and the t, chi-squared, F and Weibull distributions, against tables. */
class DistributionsTest {

    @Test
    void incompleteGammaAndBeta() {
        // torch.special.gammainc and gammaincc
        assertEquals(0.3614300768962048, Special.gammaP(2.5, 1.7), 1e-13);
        assertEquals(3.572865928700233e-15, Special.gammaQ(3, 40), 1e-26);
        assertEquals(0.5, Special.betaI(3.5, 3.5, 0.5), 1e-14);
        // I_x(1, b) = 1 - (1 - x)^b
        assertEquals(1 - Math.pow(0.7, 4), Special.betaI(1, 4, 0.3), 1e-14);
    }

    @Test
    void studentT() {
        assertEquals(0.95, Distributions.studentTCdf(2.015048373, 5), 1e-9);
        assertEquals(2.228138852, Distributions.studentTQuantile(0.975, 10), 1e-8);
        assertEquals(-2.228138852, Distributions.studentTQuantile(0.025, 10), 1e-8);
        assertEquals(0.05, Distributions.studentTTwoSided(2.228138852, 10), 1e-9);
        // one degree of freedom is the Cauchy distribution
        assertEquals(0.5 + Math.atan(1.3) / Math.PI, Distributions.studentTCdf(1.3, 1), 1e-13);
    }

    @Test
    void chiSquaredAndF() {
        assertEquals(7.814727903, Distributions.chiSquaredQuantile(0.95, 3), 1e-8);
        assertEquals(0.05, Distributions.chiSquaredSurvival(7.814727903, 3), 1e-10);
        assertEquals(3.325834530, Distributions.fQuantile(0.95, 5, 10), 1e-8);
        assertEquals(0.05, Distributions.fSurvival(3.325834530, 5, 10), 1e-10);
        // F(1, d) is the square of t(d)
        assertEquals(Distributions.studentTTwoSided(1.7, 8), Distributions.fSurvival(1.7 * 1.7, 1, 8), 1e-13);
    }

    @Test
    void weibull() {
        // shape 1 is the exponential
        assertEquals(1 - Math.exp(-2.0 / 3), Distributions.weibullCdf(2, 1, 3), 1e-15);
        assertEquals(1.7, Distributions.weibullQuantile(Distributions.weibullCdf(1.7, 2.5, 1.2), 2.5, 1.2), 1e-12);
        assertEquals(Math.exp(-1), Distributions.weibullPdf(1, 1, 1), 1e-15);
    }
}
