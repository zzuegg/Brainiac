package io.github.zzuegg.brainiac.experiments.science;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class EpidemicsTest {

    @Test
    void rtEstimatorsTrackTheTruthAndNowcastingRemovesTheReportingBias() {
        Map<String, double[]> r = Epidemics.reproductionNumber(120, 1);
        System.out.printf("13.3.7 R_t (renewal outbreaks, 120 days): %s%n", Lines.of(r));
        assertTrue(r.get("step EPIESTIM {RMSE, 90% coverage}")[0] < 0.4, Lines.of(r));
        assertTrue(r.get("smooth RW_KALMAN {RMSE, 90% coverage}")[0] < 0.4, Lines.of(r));
        double[] nc = r.get("step last-7-day relative error {reported so far, nowcast}");
        assertTrue(nc[1] < nc[0], Lines.of(r));
    }

    @Test
    void seirFitForecastsTheOutbreak() {
        Map<String, double[]> r = Epidemics.forecasts(20_000, 2);
        System.out.printf("13.3.7 forecasts (Gillespie SEIR, N=20000): %s%n", Lines.of(r));
        double[] fit = r.get("peak fit R0 {estimate, profile lo, profile hi, truth}");
        assertTrue(Math.abs(fit[0] - 2.0) < 0.5, Lines.of(r));
    }
}
