package io.github.zzuegg.brainiac.experiments.explainability;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.explainability.InterpretableNetworks.Guided;
import io.github.zzuegg.brainiac.experiments.explainability.InterpretableNetworks.GuidedScores;

import java.util.Map;

import org.junit.jupiter.api.Test;

class GuidedTrainingBaselinesTest {

    @Test
    void groupRobustBaselinesNeedGroupLabels() {
        Map<Guided, GuidedScores> m = InterpretableNetworks.guided(new Guided[] {Guided.ERM, Guided.GROUP_DRO, Guided.JTT,
            Guided.LAST_LAYER, Guided.AUGMENT}, 400, 1);
        m.forEach((k, s) -> System.out.printf("Guided %-15s shortcut-broken acc %.3f aligned acc %.3f patch saliency %.3f%n",
                k, s.brokenAccuracy(), s.alignedAccuracy(), s.patchSaliency()));
        assertTrue(m.get(Guided.AUGMENT).brokenAccuracy() > m.get(Guided.ERM).brokenAccuracy(),
                "randomising the patch removes the shortcut");
    }
}
