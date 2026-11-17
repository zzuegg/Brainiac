package io.github.zzuegg.brainiac.experiments.data;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.data.TrainingAttributionExperiment.Method;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class TrainingAttributionExperimentTest {

    @Test
    void influenceAndTracInFindFlippedLabelsAndPredictRemovals() {
        Map<Method, double[]> all = new EnumMap<>(Method.class);
        for (Method m : Method.values()) {
            long t = System.nanoTime();
            double detect = TrainingAttributionExperiment.detection(m, 0.1, 1);
            long t2 = System.nanoTime();
            double corr = TrainingAttributionExperiment.leaveOut(m, 16, 60, 1);
            all.put(m, new double[] {detect, corr});
            System.out.printf("Data attribution %-15s flipped precision %.3f (random 0.10) leave-60-out r %.3f"
                    + " (%d + %d ms)%n", m, detect, corr, (t2 - t) / 1_000_000, (System.nanoTime() - t2) / 1_000_000);
        }
        assertTrue(all.get(Method.TRACIN)[0] > 0.3, "TracIn self-influence finds flipped labels");
        assertTrue(all.get(Method.INFLUENCE)[1] > 0.3, "influence predicts removals");
    }
}
