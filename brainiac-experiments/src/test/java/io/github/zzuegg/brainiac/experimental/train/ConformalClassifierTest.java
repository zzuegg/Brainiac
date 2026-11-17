package io.github.zzuegg.brainiac.experimental.train;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.SplittableRandom;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class ConformalClassifierTest {

    /** A synthetic, deliberately overconfident classifier on 5 classes: noisy logits. */
    static double[][] probs(int n, int[] labels, long seed, double difficulty, double margin) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] p = new double[n][5];
        for (int i = 0; i < n; i++) {
            labels[i] = r.nextInt(5);
            double[] z = new double[5];
            for (int c = 0; c < 5; c++) {
                z[c] = 3 * r.nextGaussian() * difficulty + (c == labels[i] ? margin : 0);
            }
            p[i] = Calibration.softmax(z, 1);
        }
        return p;
    }

    @ParameterizedTest
    @EnumSource(ConformalClassifier.Score.class)
    void setsCoverTheTrueClassAtTheRequestedRate(ConformalClassifier.Score score) {
        int[] cl = new int[1000];
        double[][] cal = probs(1000, cl, 1, 1, 3);
        int[] tl = new int[3000];
        double[][] test = probs(3000, tl, 2, 1, 3);
        for (double alpha : new double[] {0.05, 0.1, 0.2}) {
            ConformalClassifier cp = ConformalClassifier.calibrate(cal, cl, alpha, score);
            double cov = cp.coverage(test, tl);
            assertTrue(cov >= 1 - alpha - 0.02 && cov <= 1 - alpha + 0.04, score + " alpha " + alpha + " cov " + cov);
        }
        // the plain top-1 answer covers less than 90% here, so sets must be bigger than one
        ConformalClassifier cp = ConformalClassifier.calibrate(cal, cl, 0.1, score);
        assertTrue(cp.meanSetSize(test) > 1);
        // confidently separated inputs get smaller sets
        int[] el = new int[1000];
        assertTrue(cp.meanSetSize(probs(1000, el, 3, 0.3, 9)) < cp.meanSetSize(test));
        assertTrue(Calibration.accuracy(test, tl) < 0.9);
    }
}
