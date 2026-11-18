package io.github.zzuegg.brainiac.experiments.explainability;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.explainability.InterpretableNetworks.Guided;
import io.github.zzuegg.brainiac.experiments.explainability.InterpretableNetworks.GuidedScores;

import java.util.Map;

import org.junit.jupiter.api.Test;

class ExplanationGuidedTrainingTest {

    @Test
    void rightForTheRightReasonsIgnoresTheShortcut() {
        Map<Guided, GuidedScores> m = InterpretableNetworks.guided(new Guided[] {Guided.ERM, Guided.RRR_10_PERCENT, Guided.RRR_ALL,
            Guided.RRR_INTEGRATED}, 300, 1);
        m.forEach((k, s) -> System.out.printf("Guided %-15s shortcut-broken acc %.3f aligned acc %.3f patch saliency %.3f%n",
                k, s.brokenAccuracy(), s.alignedAccuracy(), s.patchSaliency()));
        assertTrue(m.get(Guided.RRR_ALL).patchSaliency() < m.get(Guided.ERM).patchSaliency(),
                "the penalty moves attribution off the patch");
        assertTrue(m.get(Guided.RRR_ALL).brokenAccuracy() > m.get(Guided.ERM).brokenAccuracy(),
                "and helps when the shortcut breaks");
    }
}
