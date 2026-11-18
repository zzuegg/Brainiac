package io.github.zzuegg.brainiac.experiments.explainability;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.explainability.ShapleyMethods.Detector;
import io.github.zzuegg.brainiac.experiments.explainability.ShapleyMethods.DetectorScores;
import io.github.zzuegg.brainiac.experiments.explainability.ShapleyMethods.EstimatorRow;

import java.util.Arrays;
import java.util.Map;

import org.junit.jupiter.api.Test;

class ShapleyMethodsTest {

    @Test
    void treeShapIsExactAndFast() {
        ShapleyMethods.TreeTable t = ShapleyMethods.trees(1);
        System.out.printf("TreeSHAP p=8: path err %.1e interventional err %.1e, %.0f us/row vs KernelSHAP(2000) %.0f us/row"
                + " (%.0fx) err %.3f; irrelevant-share path %.3f interventional %.3f; noise rank gain %d TreeSHAP %d%n",
                t.pathError(), t.interventionalError(), t.treeShapMicros(), t.kernelShapMicros(), t.speedup(),
                t.kernelShapError(), t.pathIrrelevantShare(), t.interventionalIrrelevantShare(), t.gainNoiseRank(),
                t.shapNoiseRank());
        assertTrue(t.interventionalError() < 1e-9 && t.pathError() < 1e-9);
    }

    @Test
    void interactionIndicesFindThePlantedPairs() {
        for (boolean trees : new boolean[] {true, false}) {
            Map<Detector, DetectorScores> m = ShapleyMethods.interactions(trees, 128, 1);
            m.forEach((d, s) -> System.out.printf("Interactions %s %-20s precision@5 %.2f error-to-SII %.4f calls/row %.0f%n",
                    trees ? "GBT" : "MLP", d, s.precision(), s.errorToExact(), s.evaluations()));
            assertTrue(m.get(Detector.K_SII_EXACT).precision() >= (trees ? 0.6 : 0.8), "k-SII finds planted pairs");
        }
    }

    @Test
    void leverageShapAgainstOtherEstimators() {
        int[] multiples = {2, 8, 32};
        for (int p : new int[] {8, 16, 32}) {
            Map<EstimatorRow, double[]> m = ShapleyMethods.estimators(p, multiples, p == 32 ? 4 : 6, 1);
            m.forEach((e, v) -> System.out.printf("Estimators p=%d %-24s MSE at 2p/8p/32p %s%n", p, e,
                    Arrays.toString(Arrays.stream(v).mapToObj(x -> String.format("%.2e", x)).toArray())));
            assertTrue(m.get(EstimatorRow.LEVERAGE_SHAP)[2] < m.get(EstimatorRow.PERMUTATION)[2],
                    "Leverage SHAP beats permutation sampling at 32p");
        }
    }
}
