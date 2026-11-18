package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

/** Conformal intervals under drift ({@link ProbabilisticForecasting#conformal}). */
class ConformalForecastingTest {

    @Test
    void aciRecoversCoverageAfterTheShift() {
        Map<String, ProbabilisticForecasting.Drift> r = ProbabilisticForecasting.conformal(200, 0.05, 1);
        r.forEach((k, v) -> System.out.printf("conformal %s: before %.2f after %.2f recovered %.2f width %.2f%n", k,
                v.before(), v.after(), v.recovered(), v.width()));
        assertTrue(Math.abs(r.get("CQR").before() - 0.8) < 0.12, r.toString());
        assertTrue(r.get("ACI").recovered() > r.get("CQR").recovered(), r.toString());
        assertTrue(r.get("ACI").recovered() > 0.65, r.toString());
    }
}
