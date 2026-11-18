package io.github.zzuegg.brainiac.experiments.markets;

import io.github.zzuegg.brainiac.experimental.data.MarketPaths;
import io.github.zzuegg.brainiac.experimental.markets.OnlinePortfolios;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Online portfolio selection (wave 12, item 12.4.18): universal portfolio, EG, ONS and OLMAR
 * against buy-and-hold, the uniform CRP and the hindsight BCRP, per market regime, with and
 * without proportional costs.
 */
public final class PortfolioSelection {

    private PortfolioSelection() {
    }

    /**
     * Runs every strategy on one market.
     *
     * @param regime the market dynamics
     * @param periods periods
     * @param assets assets
     * @param costs proportional costs to test
     * @param seed the seed
     * @return strategy to {@code {log wealth at each cost..., regret to BCRP at zero cost}}; the
     *     BCRP row holds its log wealth
     */
    public static Map<String, double[]> run(MarketPaths.Regime regime, int periods, int assets, double[] costs,
            long seed) {
        double[][] x = MarketPaths.priceRelatives(periods, assets, regime, new SplittableRandom(seed));
        double bcrp = OnlinePortfolios.bestConstantRebalanced(x, 500)[0];
        Map<String, double[]> out = new LinkedHashMap<>();
        for (OnlinePortfolios.Method m : OnlinePortfolios.Method.values()) {
            double[] row = new double[costs.length + 1];
            for (int c = 0; c < costs.length; c++) {
                row[c] = OnlinePortfolios.logWealth(m, x, costs[c], seed);
            }
            row[costs.length] = bcrp - OnlinePortfolios.logWealth(m, x, 0, seed);
            out.put(m.name(), row);
        }
        double[] b = new double[costs.length + 1];
        b[0] = bcrp;
        out.put("BCRP (hindsight, no costs)", b);
        return out;
    }
}
