package io.github.zzuegg.brainiac.experiments.data;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.data.DataValuation.Detection;
import io.github.zzuegg.brainiac.experiments.data.DataValuation.Selection;
import io.github.zzuegg.brainiac.experiments.data.DataValuation.Valuation;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class DataValuationTest {

    @Test
    void cheapValuationsFindFlippedLabels() {
        Map<Valuation, Detection> rows = new EnumMap<>(Valuation.class);
        for (Valuation v : Valuation.values()) {
            Detection d = DataValuation.detection(v, 0.2, 1);
            rows.put(v, d);
            System.out.printf("Valuation %-21s flipped share in lowest 20%% %.2f  acc after removal %.3f  %d ms%n",
                    v, d.flippedShare(), d.accuracy(), d.millis());
        }
        long t0 = System.nanoTime();
        ShapleyPruning.Result shapley = ShapleyPruning.run(ShapleyPruning.Method.IN_RUN_SHAPLEY, 0.2, 0.2, 1);
        long shapleyMs = (System.nanoTime() - t0) / 1_000_000;
        t0 = System.nanoTime();
        double tracIn = TrainingAttributionExperiment.detection(TrainingAttributionExperiment.Method.TRACIN, 0.2, 1);
        long tracInMs = (System.nanoTime() - t0) / 1_000_000;
        ShapleyPruning.Result el2n = ShapleyPruning.run(ShapleyPruning.Method.EL2N, 0.2, 0.2, 1);
        t0 = System.nanoTime();
        double influence = TrainingAttributionExperiment.detection(TrainingAttributionExperiment.Method.INFLUENCE,
                0.2, 1);
        long influenceMs = (System.nanoTime() - t0) / 1_000_000;
        System.out.printf("Valuation baselines: In-Run Shapley %.2f (acc %.3f, %d ms incl. retrain), EL2N %.2f "
                + "(acc %.3f), TracIn %.2f (%d ms), influence %.2f (%d ms)%n", shapley.flippedRemoved(),
                shapley.accuracy(), shapleyMs, el2n.flippedRemoved(), el2n.accuracy(), tracIn, tracInMs, influence,
                influenceMs);
        assertTrue(rows.get(Valuation.KNN_SHAPLEY).flippedShare() > 0.5, "KNN-Shapley finds flipped labels");
        assertTrue(rows.get(Valuation.DATA_OOB).flippedShare() > 0.4, "Data-OOB finds flipped labels");
        assertTrue(rows.get(Valuation.LEVERAGE).flippedShare() < 0.35, "leverage is label-blind");
    }

    @Test
    void leverageSelectsDiverseSubsets() {
        for (int size : new int[] {30, 60}) {
            double[] acc = new double[Selection.values().length];
            for (long seed = 1; seed <= 3; seed++) {
                for (Selection s : Selection.values()) {
                    acc[s.ordinal()] += DataValuation.selection(s, size, seed) / 3;
                }
            }
            System.out.printf("Valuation subset of %d labels (3 seeds): random %.3f  leverage(pixels) %.3f  "
                    + "leverage(SSL) %.3f  top KNN-Shapley %.3f%n", size, acc[0], acc[1], acc[2], acc[3]);
            assertTrue(acc[0] > 0.2, "trains");
        }
    }
}
