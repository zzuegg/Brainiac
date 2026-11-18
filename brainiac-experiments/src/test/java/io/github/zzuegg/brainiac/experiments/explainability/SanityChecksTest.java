package io.github.zzuegg.brainiac.experiments.explainability;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.explainability.ExplanationEvaluation.Method;
import io.github.zzuegg.brainiac.experiments.explainability.ExplanationEvaluation.SanityRow;
import io.github.zzuegg.brainiac.experiments.explainability.ExplanationEvaluation.SanityStudy;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class SanityChecksTest {

    @Test
    void gradientMapsChangeUnderRandomisation() {
        SanityStudy s = ExplanationEvaluation.sanity(300, 40, 1);
        System.out.printf("Sanity CNN accuracy %.3f, permuted-label CNN %.3f%n", s.accuracy(), s.shuffledAccuracy());
        Map<Method, SanityRow> by = new EnumMap<>(Method.class);
        for (SanityRow r : s.rows()) {
            by.put(r.method(), r);
            System.out.printf("Sanity %-20s pointing %.2f deletion AUC %.3f rank after top layer %.2f after all %.2f "
                    + "SSIM after all %.2f rank permuted labels %.2f%n", r.method(), r.pointing(), r.deletionAuc(),
                    r.topLayerRank(), r.allLayersRank(), r.allLayersSsim(), r.dataRank());
        }
        assertTrue(s.accuracy() > 0.4, "the CNN learned the task");
        assertTrue(by.get(Method.SALIENCY).allLayersRank() < by.get(Method.EDGE_DETECTOR).allLayersRank(),
                "saliency changes under randomisation more than a model-independent edge map");
        assertTrue(by.get(Method.EDGE_DETECTOR).pointing() > by.get(Method.RANDOM).pointing(),
                "the edge map still points at the object");
    }
}
