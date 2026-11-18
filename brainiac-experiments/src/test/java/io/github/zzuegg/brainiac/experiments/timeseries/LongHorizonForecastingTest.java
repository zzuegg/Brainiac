package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class LongHorizonForecastingTest {

    @Test
    void transformersAgainstALinearMap() {
        Map<Integer, Map<String, LongHorizonForecasting.Score>> r = LongHorizonForecasting.horizons(48,
                new int[] {24}, 40, 1);
        r.forEach((h, row) -> row.forEach((k, v) -> System.out.printf("long horizon H=%d %s: MSE %s%n", h, k, v)));
        for (Map<String, LongHorizonForecasting.Score> row : r.values()) {
            double naive = row.get("seasonal naive").mse();
            assertTrue(row.get("DLinear").mse() < naive, row.toString());
            assertTrue(row.get("N-HiTS").mse() < naive, row.toString());
            for (String m : new String[] {"Autoformer", "FEDformer", "TimesNet", "vanilla transformer"}) {
                assertTrue(row.get(m).mse() < 8 * naive, m + " " + row);
            }
        }
    }
}
