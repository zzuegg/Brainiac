package io.github.zzuegg.brainiac.experimental.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class CalibrationTest {

    @Test
    void calibratedPredictionsHaveLowEceOverconfidentOnesHigh() {
        SplittableRandom r = new SplittableRandom(1);
        int n = 20_000;
        double[][] calibrated = new double[n][];
        double[][] overconfident = new double[n][];
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) {
            double p = r.nextDouble(0.5, 1);
            labels[i] = r.nextDouble() < p ? 0 : 1;   // class 0 is right with probability p
            calibrated[i] = new double[] {p, 1 - p};
            double q = Math.min(1, p + 0.3);
            overconfident[i] = new double[] {q, 1 - q};
        }
        assertTrue(Calibration.expectedCalibrationError(calibrated, labels, 15) < 0.02);
        assertTrue(Calibration.expectedCalibrationError(overconfident, labels, 15) > 0.15);
        assertTrue(Calibration.negativeLogLikelihood(calibrated, labels)
                < Calibration.negativeLogLikelihood(overconfident, labels));
    }

    @Test
    void temperatureScalingRecoversTheTrueTemperature() {
        SplittableRandom r = new SplittableRandom(2);
        int n = 5_000;
        double[][] logits = new double[n][3];
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) {
            double[] trueLogits = {r.nextGaussian() * 2, r.nextGaussian() * 2, r.nextGaussian() * 2};
            double[] p = Calibration.softmax(trueLogits, 1);
            double u = r.nextDouble();
            labels[i] = u < p[0] ? 0 : u < p[0] + p[1] ? 1 : 2;
            for (int c = 0; c < 3; c++) {
                logits[i][c] = trueLogits[c] * 3;   // a model three times too confident
            }
        }
        assertEquals(3, Calibration.fitTemperature(logits, labels), 0.3);
    }

    @Test
    void mutualInformationIsZeroWhenMembersAgree() {
        assertEquals(0, Calibration.mutualInformation(new double[][] {{0.7, 0.3}, {0.7, 0.3}}), 1e-12);
        assertEquals(Math.log(2), Calibration.mutualInformation(new double[][] {{1, 0}, {0, 1}}), 1e-9);
    }

    @Test
    void selectiveCurveRanksErrorsLast() {
        PopulationEnsemble e = PopulationEnsemble.ofFunctions(List.of(
                x -> new double[][] {{0.9, 0.1}, {0.6, 0.4}, {0.95, 0.05}},
                x -> new double[][] {{0.8, 0.2}, {0.3, 0.7}, {0.9, 0.1}}));
        float[][] x = new float[3][1];
        PopulationEnsemble.SelectiveCurve c = e.selective(x, new int[] {0, 0, 0},
                PopulationEnsemble.Uncertainty.ENTROPY);
        assertArrayEquals(new double[] {0, 0, 1 / 3.0}, c.risk(), 1e-12);
        assertArrayEquals(new int[] {0, -1, 0},
                e.predictOrAbstain(x, PopulationEnsemble.Uncertainty.DISAGREEMENT, 0.4));
    }
}
