package io.github.zzuegg.brainiac.experiments.evaluation;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.eval.CurveExtrapolator;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class CurveExtrapolationTest {

    @Test
    void extrapolationBeatsLastValueOnPriorCurves() {
        SplittableRandom r = new SplittableRandom(3);
        CurveExtrapolator.Pfn pfn = new CurveExtrapolator.Pfn(10, 40, 24, 1, 0.003f, new SplittableRandom(4));
        double loss = pfn.pretrain(250, 32, 5);
        double[] err = new double[4];
        int curves = 30;
        double[] x = new double[20];
        for (int i = 0; i < x.length; i++) {
            x[i] = 0.1 * (i + 1) / x.length;
        }
        for (int k = 0; k < curves; k++) {
            double[] full = CurveExtrapolator.samplePriorCurve(x, r);
            double[] y = java.util.Arrays.copyOf(full, x.length);
            double truth = full[x.length];
            err[0] += Math.abs(y[y.length - 1] - truth) / curves;
            err[1] += Math.abs(CurveExtrapolator.powerLawLeastSquares(x, y).predict(1) - truth) / curves;
            err[2] += Math.abs(CurveExtrapolator.ensemble(x, y, 1, 300, k).mean() - truth) / curves;
            err[3] += Math.abs(pfn.predict(x, y, 1).mean() - truth) / curves;
        }
        System.out.printf("curve extrapolation from 10%% of the budget, 30 prior curves: MAE last value %.4f, power law "
                + "%.4f, ensemble %.4f, LC-PFN %.4f (PFN loss %.2f)%n", err[0], err[1], err[2], err[3], loss);
        assertTrue(err[2] < err[0], "the ensemble beats the last value");
    }
}
