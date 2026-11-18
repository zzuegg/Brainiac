package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.IrregularSeries;

import org.junit.jupiter.api.Test;

import java.util.Map;

class StateSpaceEmTest {

    static String line(Map<String, double[]> r) {
        StringBuilder b = new StringBuilder();
        r.forEach((k, v) -> {
            b.append(k).append(" {");
            for (int i = 0; i < v.length; i++) {
                b.append(String.format(i == 0 ? "%.3f" : ", %.3f", v[i]));
            }
            b.append("} | ");
        });
        return b.toString();
    }

    @Test
    void emFittedLdsImputesLinearGaussianSeries() {
        Map<String, double[]> mar = StateSpaceEm.imputation(IrregularSeries.Kind.LINEAR_GAUSSIAN,
                IrregularSeries.Missing.MAR, 0.5, 8, 100, 2, 30, 80, 1);
        System.out.printf("11.3.14 linear-Gaussian 50%% MAR {imputation MAE, one-step MAE, ms}: %s%n", line(mar));
        Map<String, double[]> block = StateSpaceEm.imputation(IrregularSeries.Kind.LINEAR_GAUSSIAN,
                IrregularSeries.Missing.BLOCK, 0.3, 8, 100, 2, 30, 0, 2);
        System.out.printf("11.3.14 linear-Gaussian 30%% block {imputation MAE, one-step MAE, ms}: %s%n", line(block));
        Map<String, double[]> mnar = StateSpaceEm.imputation(IrregularSeries.Kind.LINEAR_GAUSSIAN,
                IrregularSeries.Missing.MNAR, 0.5, 8, 100, 2, 30, 0, 3);
        System.out.printf("11.3.14 linear-Gaussian 50%% MNAR {imputation MAE, one-step MAE, ms}: %s%n", line(mnar));
        assertTrue(mar.get("LDS EM")[0] < mar.get("linear interpolation")[0], line(mar));
        assertTrue(mar.get("LDS EM")[1] < mar.get("local linear trend smoother")[1], line(mar));
    }

    @Test
    void emRecoversPlantedDynamics() {
        double[] full = StateSpaceEm.recovery(200, 4, 0, 40, 1);
        double[] half = StateSpaceEm.recovery(200, 4, 0.5, 40, 1);
        System.out.printf("11.3.14 recovery {eigenvalue error, R rel. error, LL decreases, LL/obs first, last}: "
                + "complete {%.3f, %.3f, %.0f, %.3f, %.3f} | 50%% missing {%.3f, %.3f, %.0f, %.3f, %.3f}%n", full[0],
                full[1], full[2], full[3], full[4], half[0], half[1], half[2], half[3], half[4]);
        assertEquals(0, full[2], "the EM log-likelihood never decreases");
        assertTrue(full[4] > full[3], "EM improves the likelihood");
        assertTrue(full[0] < 0.15, "eigenvalues recovered: " + full[0]);
    }
}
