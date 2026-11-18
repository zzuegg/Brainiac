package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

/** ETS and ARIMA intervals ({@link ProbabilisticForecasting#stateSpace}). */
class StateSpaceForecastingTest {

    @Test
    void stateSpaceIntervals() {
        Map<String, ProbabilisticForecasting.Probabilistic> r = ProbabilisticForecasting.stateSpace(12, 100, 4, 1);
        r.forEach((k, v) -> System.out.printf("state space %s: mse %.4f crps %.4f coverage %.2f width %.2f%n", k,
                v.mse(), v.crps(), v.coverage(), v.width()));
        assertTrue(r.get("ETS").mse() < r.get("seasonal naive").mse(), r.toString());
        assertTrue(Math.abs(r.get("ETS").coverage() - 0.8) < 0.15, r.toString());
    }
}
