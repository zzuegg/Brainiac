package io.github.zzuegg.brainiac.experiments.classic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class BoostingShootoutProbabilisticTest {

    @Test
    void ngboostAndQuantileForestOnHeteroscedasticNoise() {
        Map<String, double[]> r = BoostingShootout.probabilistic(500, false, 150, 1);
        String line = BoostingShootoutLightGbmTest.line(
                "Probabilistic heteroscedastic NLL/CRPS/pinball/cover90/width/RMSE", r, " %.3f");
        System.out.println(line);
        double[] ng = r.get("NGBoost Normal, log score");
        assertTrue(ng[0] < r.get("forest Gaussian (tree spread)")[0], line);
        assertTrue(ng[3] > 0.75, line);
        assertTrue(r.get("quantile forest + CQR")[3] >= r.get("quantile regression forest")[3] - 1e-9, line);
    }

    @Test
    void quantileForestOnBimodalNoise() {
        Map<String, double[]> r = BoostingShootout.probabilistic(500, true, 150, 2);
        String line = BoostingShootoutLightGbmTest.line("Probabilistic bimodal NLL/CRPS/pinball/cover90/width/RMSE",
                r, " %.3f");
        System.out.println(line);
        assertTrue(r.get("quantile regression forest")[1] < r.get("forest Gaussian (tree spread)")[1], line);
        assertTrue(r.get("quantile regression forest")[2] < 0.5, line);
    }
}
