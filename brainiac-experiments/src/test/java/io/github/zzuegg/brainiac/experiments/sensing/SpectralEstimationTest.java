package io.github.zzuegg.brainiac.experiments.sensing;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class SpectralEstimationTest {

    @Test
    void spectralEstimation() {
        Map<String, double[]> r = SpectralEstimation.table(30, 3);
        String s = AdaptiveFiltersTest.line(r, "%.3f");
        System.out.printf("13.4.16 spectral estimation {lsd dB n=256/64; freq RMSE (1/n); P(resolve 0.8/n, n=64);"
                + " irregular AUC/RMSE}: %s%n", s);
        assertTrue(r.get("lsd / multitaper (NW=4)")[0] < r.get("lsd / periodogram")[0], s);
        assertTrue(r.get("resolve / Burg (AIC, p<=20)")[0] > r.get("resolve / Welch (segments n/4)")[0], s);
        assertTrue(r.get("irregular / Lomb-Scargle")[0] > 0.5, s);
    }
}
