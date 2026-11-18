package io.github.zzuegg.brainiac.experiments.data;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.ActiveLearner.Acquisition;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class ActiveLearningTest {

    @Test
    void acquisitionFunctionsAgainstRandomSampling() {
        Map<Acquisition, Double> dupArea = new EnumMap<>(Acquisition.class);
        Map<Acquisition, Double> cleanArea = new EnumMap<>(Acquisition.class);
        for (boolean dup : new boolean[] {false, true}) {
            for (int batch : dup ? new int[] {10} : new int[] {10, 50}) {
                int budget = batch == 10 ? 120 : 270;
                for (Acquisition a : Acquisition.values()) {
                    ActiveLearning.Curve c = ActiveLearning.run(a, dup, batch, budget, 1);
                    System.out.printf("Active learning %-10s pool %-9s batch %2d budget %3d: curve area %.3f  "
                            + "final %.3f  %d ms%n", a, dup ? "30%-dupes" : "clean", batch, budget, c.area(),
                            c.last(), c.millis());
                    if (batch == 10) {
                        (dup ? dupArea : cleanArea).put(a, c.area());
                    }
                }
            }
        }
        assertTrue(cleanArea.get(Acquisition.RANDOM) > 0.3, "the learner learns");
        double bestDiverse = Math.max(dupArea.get(Acquisition.BADGE), Math.max(dupArea.get(Acquisition.CORE_SET),
                dupArea.get(Acquisition.LEVERAGE)));
        assertTrue(bestDiverse > dupArea.get(Acquisition.ENTROPY) - 0.02,
                "a diversity-aware method is not beaten by top-k entropy on the duplicated pool");
    }
}
