package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class HierarchicalForecastingTest {

    @Test
    void reconciliationImprovesBaseForecasts() {
        Map<String, double[]> clean = HierarchicalForecasting.compare(false, 6, 1);
        System.out.printf("11.3.15 clean {MASE total, regions, leaves, all}: %s%n", StateSpaceEmTest.line(clean));
        Map<String, double[]> dirty = HierarchicalForecasting.compare(true, 6, 1);
        System.out.printf("11.3.15 outliers {MASE total, regions, leaves, all}: %s%n", StateSpaceEmTest.line(dirty));
        assertTrue(clean.get("MINT_SHRINK")[3] < clean.get("base forecasts (unreconciled)")[3],
                StateSpaceEmTest.line(clean));
        assertTrue(clean.get("MINT_SHRINK")[3] < clean.get("OLS")[3], StateSpaceEmTest.line(clean));
        assertTrue(clean.get("MINT_SHRINK")[3] < clean.get("BOTTOM_UP")[3], StateSpaceEmTest.line(clean));
        assertTrue(dirty.get("ROBUST_HUBER")[3] < dirty.get("MINT_SHRINK")[3], StateSpaceEmTest.line(dirty));
    }
}
