package io.github.zzuegg.brainiac.experiments.events;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.IrregularSeries;
import org.junit.jupiter.api.Test;

import java.util.Map;

class IrregularSeriesImputationTest {

    @Test
    void imputersRunAndBeatTheMean() {
        Map<String, double[]> r = IrregularSeriesShootout.imputation(IrregularSeries.Kind.SEASONAL, 0.5,
                IrregularSeries.Missing.BLOCK, IrregularSeriesReconstructionTest.BUDGET, 20, 5, 1);
        String line = PointProcessShootoutTest.line("imputation seasonal 50% block MAE, CRPS, ms", r);
        System.out.println(line);
        double mean = r.get("mean")[0];
        assertTrue(r.get("linear interpolation")[0] < mean, line);
        assertTrue(r.get("SAITS")[0] < mean && r.get("BRITS")[0] < mean, line);
        assertTrue(Double.isFinite(r.get("CSDI")[1]), line);
    }
}
