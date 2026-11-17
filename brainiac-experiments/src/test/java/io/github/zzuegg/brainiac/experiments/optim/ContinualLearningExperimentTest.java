package io.github.zzuegg.brainiac.experiments.optim;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.train.ContinualBrain;
import io.github.zzuegg.brainiac.experimental.train.ForgettingTracker;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class ContinualLearningExperimentTest {

    @Test
    void growingBrainWithFrozenRegionsDoesNotForget() {
        List<ContinualLearningExperiment.Task> tasks = ContinualLearningExperiment.permutedDigits(3, 30, 1);
        Map<ContinualBrain.Protection, ForgettingTracker> results = new EnumMap<>(ContinualBrain.Protection.class);
        for (ContinualBrain.Protection p : ContinualBrain.Protection.values()) {
            ForgettingTracker t = ContinualLearningExperiment.run(tasks, p, 32, 15, 1);
            results.put(p, t);
            System.out.println(p + "\n" + t);
        }
        ForgettingTracker frozen = results.get(ContinualBrain.Protection.GROW_FROZEN);
        ForgettingTracker shared = results.get(ContinualBrain.Protection.SHARED);
        assertEquals(0, frozen.averageForgetting(), 1e-9, "frozen regions cannot forget");
        assertTrue(shared.averageForgetting() > 0.05, "plain fine-tuning forgets: " + shared.averageForgetting());
        assertTrue(frozen.averageAccuracy() > shared.averageAccuracy());
        assertTrue(results.get(ContinualBrain.Protection.SHARED_EWC).averageForgetting() < shared.averageForgetting());
    }
}
