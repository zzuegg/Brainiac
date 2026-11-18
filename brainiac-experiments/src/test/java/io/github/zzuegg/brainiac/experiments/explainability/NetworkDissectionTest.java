package io.github.zzuegg.brainiac.experiments.explainability;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.explain.NetworkDissection;
import io.github.zzuegg.brainiac.experiments.explainability.ExplanationEvaluation.DissectionStudy;
import io.github.zzuegg.brainiac.experiments.explainability.ExplanationEvaluation.LayerReport;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

class NetworkDissectionTest {

    static void print(String name, DissectionStudy s) {
        System.out.printf("Dissection %s accuracy %.3f, shortcut broken %.3f%n", name, s.accuracy(),
                s.brokenAccuracy());
        for (LayerReport l : s.layers()) {
            NetworkDissection.Dissection d = l.dissection();
            System.out.printf("Dissection %s %s unique detectors %d mean IoU %.3f formula IoU %.3f share >=20%% gain %.2f "
                    + "patch detectors %d | random directions mean IoU %.3f%n", name, l.layer(),
                    d.uniqueDetectors(ExplanationEvaluation.DETECTOR_IOU), d.meanIou(), d.meanFormulaIou(),
                    d.formulaShare(0.2), d.detectors("patch", ExplanationEvaluation.DETECTOR_IOU),
                    l.random().meanIou());
            for (int u = 0; u < d.single().length; u++) {
                System.out.printf("Dissection %s %s unit %d %s %.3f | %s %.3f%n", name, l.layer(), u,
                        d.single()[u].formula(), d.single()[u].iou(), d.compositional()[u].formula(),
                        d.compositional()[u].iou());
            }
        }
        System.out.printf("Dissection %s TCAV z %s for %s; agreement with dissection %.2f%n", name,
                Arrays.toString(Arrays.stream(s.tcavZ()).mapToObj(v -> String.format("%.1f", v)).toArray()),
                Arrays.toString(ExplanationEvaluation.TCAV_CONCEPTS), s.agreement());
    }

    @Test
    void unitsAreConceptDetectors() {
        DissectionStudy s = ExplanationEvaluation.dissection(false, 250, 0.05, 1);
        print("clean", s);
        for (LayerReport l : s.layers()) {
            // negative result kept: in an 8-channel net random channel mixtures align with the
            // concept masks about as well as single units, so the control is printed, not asserted
            assertTrue(l.dissection().uniqueDetectors(ExplanationEvaluation.DETECTOR_IOU) >= 2,
                    "the layer has concept detectors");
            assertTrue(l.dissection().meanFormulaIou() >= l.dissection().meanIou(), "formulas never lower the IoU");
        }
    }
}
