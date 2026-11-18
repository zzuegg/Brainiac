package io.github.zzuegg.brainiac.experiments.explainability;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.explainability.EffectPlotComparison.Fitted;
import io.github.zzuegg.brainiac.experiments.explainability.EffectPlotComparison.Row;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class EffectPlotComparisonTest {

    @Test
    void aleStaysOnTheDataAndHFindsThePlantedPair() {
        Map<Fitted, Row> correlated = new EnumMap<>(Fitted.class);
        for (double rho : new double[] {0, 0.9}) {
            for (Fitted m : Fitted.values()) {
                Row r = EffectPlotComparison.run(m, rho, 1);
                if (rho > 0) {
                    correlated.put(m, r);
                }
                System.out.printf("Effects rho %.1f %-6s PDP error %.3f ALE error %.3f H-statistic pair AUROC %.2f%n",
                        rho, m, r.pdpError(), r.aleError(), r.hAuroc());
            }
            EffectPlotComparison.Baselines b = EffectPlotComparison.baselines(rho, 1);
            System.out.printf("Effects rho %.1f baselines: EBM shape error %.3f GBT KernelSHAP-dependence error %.3f"
                    + " EBM FAST pair AUROC %.2f%n", rho, b.ebmError(), b.shapError(), b.ebmPairAuroc());
        }
        assertTrue(correlated.get(Fitted.TRUTH).aleError() < 0.15, "ALE recovers additive shapes of the truth");
        assertTrue(correlated.get(Fitted.MLP).aleError() < correlated.get(Fitted.MLP).pdpError(),
                "ALE beats PDP on a fitted model under correlation");
        assertTrue(correlated.get(Fitted.MLP).hAuroc() >= 0.8, "H ranks the planted pair near the top");
    }
}
