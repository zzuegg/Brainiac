package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

/** The wave 8 rows of {@link ForecastingBenchmark}: ETS, ARIMA, N-HiTS, the N-BEATS ensemble and TS2Vec. */
class ForecastingBenchmarkExtendedTest {

    @Test
    void statisticalAndDeepRows() {
        Map<String, ForecastingBenchmark.Score> r = ForecastingBenchmark.extended(36, 12, 12, 80, 3, 0, 1);
        r.forEach((k, v) -> System.out.printf("forecasting extended %s: mse %.4f mae %.4f%n", k, v.mse(), v.mae()));
        double naive = r.get("seasonal naive").mse();
        assertTrue(r.get("ETS").mse() < naive, r.toString());
        assertTrue(r.get("ARIMA").mse() < naive * 1.05, r.toString());
        assertTrue(r.get("N-HiTS").mse() < naive * 1.2, r.toString());
        assertTrue(r.get("TS2Vec + ridge").mse() < naive * 1.5, r.toString());
    }
}
