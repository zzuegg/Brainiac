package io.github.zzuegg.brainiac.experiments.events;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.IrregularSeries;
import org.junit.jupiter.api.Test;

import java.util.Map;

class IrregularSeriesReconstructionTest {

    static final IrregularSeriesShootout.Budget BUDGET = new IrregularSeriesShootout.Budget(200, 60, 32, 2, 60, 16, 24);

    @Test
    void modelsRunAndInterpolationBeatsLocf() {
        Map<String, double[]> r = IrregularSeriesShootout.reconstruction(IrregularSeries.Kind.SEASONAL, 0.6,
                IrregularSeries.Missing.MAR, BUDGET, 1);
        String line = PointProcessShootoutTest.line("irregular seasonal 60% MAR interp MSE, extrap MSE, ms", r);
        System.out.println(line);
        assertTrue(r.get("linear interpolation")[0] < r.get("LOCF")[0], line);
        for (double[] v : r.values()) {
            assertTrue(Double.isFinite(v[1]), line);
        }
    }
}
