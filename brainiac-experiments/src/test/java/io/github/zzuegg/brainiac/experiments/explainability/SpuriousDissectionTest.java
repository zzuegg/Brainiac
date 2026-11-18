package io.github.zzuegg.brainiac.experiments.explainability;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.explainability.ExplanationEvaluation.DissectionStudy;

import org.junit.jupiter.api.Test;

class SpuriousDissectionTest {

    @Test
    void tcavFlagsTheShortcutPatch() {
        DissectionStudy s = ExplanationEvaluation.dissection(true, 250, 0.05, 1);
        NetworkDissectionTest.print("spurious", s);
        double patchZ = s.tcavZ()[s.tcavZ().length - 1];
        assertTrue(s.accuracy() > s.brokenAccuracy(), "the model uses the shortcut");
        assertTrue(patchZ > 2, "TCAV flags the patch");
        int patchUnits = s.layers().stream()
                .mapToInt(l -> l.dissection().detectors("patch", ExplanationEvaluation.DETECTOR_IOU)).sum();
        assertTrue(patchUnits >= 2, "several units detect the patch");
    }
}
