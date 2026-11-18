package io.github.zzuegg.brainiac.experiments.timeseries;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

/** The multivariate, covariate and two-period rows of {@link LongHorizonForecasting}. */
class MultivariateForecastingTest {

    @Test
    void invertedTransformerUsesCoupledChannels() {
        Map<String, LongHorizonForecasting.Score> coupled = LongHorizonForecasting.multivariate(true, 50, 1);
        Map<String, LongHorizonForecasting.Score> independent = LongHorizonForecasting.multivariate(false, 50, 1);
        coupled.forEach((k, v) -> System.out.printf("multivariate coupled %s: MSE %s%n", k, v));
        independent.forEach((k, v) -> System.out.printf("multivariate independent %s: MSE %s%n", k, v));
        assertTrue(coupled.get("iTransformer").mse() < coupled.get("PatchTST").mse(), coupled.toString());
    }

    @Test
    void tideUsesKnownCovariates() {
        Map<String, LongHorizonForecasting.CovariateScore> r = LongHorizonForecasting.covariates(120, 1);
        r.forEach((k, v) -> System.out.printf("covariates %s: MSE %s%n", k, v));
        assertTrue(r.get("TiDE").holidayMse() < r.get("DLinear (no covariates)").holidayMse(), r.toString());
        assertTrue(r.get("TiDE").mse() < r.get("seasonal naive").mse(), r.toString());
    }

    @Test
    void timesNetOnTwoPeriods() {
        Map<String, LongHorizonForecasting.Score> r = LongHorizonForecasting.twoPeriods(25, 1);
        r.forEach((k, v) -> System.out.printf("two periods %s: MSE %s%n", k, v));
        assertTrue(r.get("TimesNet").mse() < 1.0, r.toString());
    }
}
