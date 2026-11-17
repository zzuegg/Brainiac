package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class TimeseriesExperimentsTest {

    @Test
    void learnedForecastersBeatNaive() {
        Map<String, ForecastingBenchmark.Score> r = ForecastingBenchmark.run(48, 12, 12, 0.004, 0.1, 300, 1);
        System.out.println("forecasting: " + r);
        double naive = r.get("naive").mse();
        for (String m : new String[] {"MLP + RevIN", "DLinear", "N-BEATS", "PatchTST", "TCN + RevIN"}) {
            assertTrue(r.get(m).mse() < naive / 2, m + " " + r);
        }
        assertTrue(r.get("DLinear").mse() < r.get("seasonal naive").mse(), r.toString());
        assertTrue(r.get("MLP + RevIN").mse() < r.get("MLP").mse(), r.toString());
    }

    @Test
    void probabilisticForecastsGiveCalibratedIntervals() {
        Map<String, ProbabilisticForecasting.Intervals> r = ProbabilisticForecasting.run(48, 12, 12, 150, 1);
        System.out.println("intervals: " + r);
        for (ProbabilisticForecasting.Intervals i : r.values()) {
            assertTrue(i.coverage() > 0.5 && i.coverage() < 0.99, r.toString());
            assertTrue(i.width() > 0, r.toString());
        }
    }
}
