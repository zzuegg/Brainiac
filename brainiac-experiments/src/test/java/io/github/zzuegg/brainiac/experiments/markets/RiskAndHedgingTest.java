package io.github.zzuegg.brainiac.experiments.markets;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class RiskAndHedgingTest {

    @Test
    void volatilityModels() {
        int[] h = {1, 5, 20};
        StringBuilder sb = new StringBuilder();
        Map<String, double[]> garch = null;
        Map<String, double[]> gjr = null;
        for (RiskAndHedging.Process p : RiskAndHedging.Process.values()) {
            Map<String, double[]> r = RiskAndHedging.volatility(p, 1500, 1000, h, 300, 11);
            sb.append(p).append(": ").append(Lines.of(r)).append(' ');
            if (p == RiskAndHedging.Process.GARCH_T) {
                garch = r;
            } else if (p == RiskAndHedging.Process.GJR_T) {
                gjr = r;
            }
        }
        System.out.printf("12.4.15 {QLIKE h 1/5/20, VaR99 exceedance} train 1500 test 1000: %s%n", sb);
        assertTrue(garch.get("GARCH-t")[0] < garch.get("rolling 60d")[0], sb.toString());
        assertTrue(gjr.get("GJR-t")[0] < gjr.get("GARCH-t")[0] + 0.005, sb.toString());
        assertTrue(garch.get("GARCH-t")[3] < 0.03, sb.toString());
    }

    @Test
    void deepHedging() {
        Map<String, double[]> free = RiskAndHedging.hedging(false, 0, 2000, 4000, 120, 12);
        Map<String, double[]> costly = RiskAndHedging.hedging(false, 0.01, 2000, 4000, 120, 12);
        Map<String, double[]> heston = RiskAndHedging.hedging(true, 0, 2000, 4000, 120, 13);
        System.out.printf("12.4.16 {mean PnL, std, CVaR95 loss, costs} GBM no cost: %s GBM 1%% cost: %s Heston: %s%n",
                Lines.of(free), Lines.of(costly), Lines.of(heston));
        assertTrue(free.get("deep hedge (CVaR95)")[2] < free.get("no hedge")[2] / 2, Lines.of(free));
        assertTrue(costly.get("deep hedge (CVaR95)")[2] < costly.get("no hedge")[2],
                Lines.of(costly));
    }
}
