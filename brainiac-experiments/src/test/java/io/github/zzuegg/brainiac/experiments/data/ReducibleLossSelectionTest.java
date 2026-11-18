package io.github.zzuegg.brainiac.experiments.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.ReducibleLossSelector;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class ReducibleLossSelectionTest {

    @Test
    void reducibleLossSkipsNoisyAndLearntPoints() {
        ReducibleLossSelector s = new ReducibleLossSelector(new double[] {3.0, 0.1, 0.1, 0.1});
        int[] pick = s.select(new int[] {0, 1, 2, 3}, new double[] {3.2, 2.0, 0.15, 1.0}, 2);
        assertArrayEquals(new int[] {1, 3}, pick);
        assertArrayEquals(new int[] {0, 1}, ReducibleLossSelector.highestLoss(new int[] {0, 1, 2, 3},
                new double[] {3.2, 2.0, 0.15, 1.0}, 2));
    }

    @Test
    void rhoLossAgainstHighestLossAndUniform() {
        DataCuration.NoisyDigits data = DataCuration.noisyDigits(1);
        Map<DataCuration.Selection, DataCuration.SelectionRun> runs = new EnumMap<>(DataCuration.Selection.class);
        for (DataCuration.Selection m : DataCuration.Selection.values()) {
            runs.put(m, DataCuration.selection(m, 300, 0.8, data, 2));
            System.out.println("online batch selection, 20% label noise, 300 steps: " + runs.get(m));
        }
        DataCuration.SelectionRun rho = runs.get(DataCuration.Selection.RHO_LOSS);
        DataCuration.SelectionRun high = runs.get(DataCuration.Selection.HIGHEST_LOSS);
        DataCuration.SelectionRun uniform = runs.get(DataCuration.Selection.UNIFORM);
        System.out.printf("RHO-LOSS: steps to 0.8 %d vs uniform %d, highest loss %d; noisy share %.2f vs %.2f (highest "
                + "loss) and %.2f (uniform)%n", rho.stepsToTarget(), uniform.stepsToTarget(), high.stepsToTarget(),
                rho.noisyShare(), high.noisyShare(), uniform.noisyShare());
        assertTrue(high.noisyShare() > uniform.noisyShare(), "highest-loss selection picks noise");
        assertTrue(rho.noisyShare() < high.noisyShare(), "RHO-LOSS filters it");
    }
}
