package io.github.zzuegg.brainiac.experiments.markets;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.MarketPaths;

import org.junit.jupiter.api.Test;

import java.util.EnumMap;
import java.util.Map;

class PortfolioSelectionTest {

    @Test
    void onlinePortfolios() {
        Map<MarketPaths.Regime, Map<String, double[]>> all = new EnumMap<>(MarketPaths.Regime.class);
        StringBuilder sb = new StringBuilder();
        for (MarketPaths.Regime regime : MarketPaths.Regime.values()) {
            Map<String, double[]> r = PortfolioSelection.run(regime, 1500, 10, new double[] {0, 0.002}, 31);
            all.put(regime, r);
            sb.append(regime).append(": ").append(Lines.of(r)).append(' ');
        }
        System.out.printf("12.4.18 {log wealth at cost 0/0.2%%, regret to BCRP} 10 assets 1500 periods: %s%n", sb);
        Map<String, double[]> rev = all.get(MarketPaths.Regime.MEAN_REVERTING);
        assertTrue(rev.get("OLMAR")[0] > rev.get("UCRP")[0], sb.toString());
        for (Map<String, double[]> r : all.values()) {
            assertTrue(r.get("EG")[2] < 0.05 * 1500, sb.toString());
            assertTrue(r.get("UCRP")[2] >= -1e-6, sb.toString());
        }
    }
}
