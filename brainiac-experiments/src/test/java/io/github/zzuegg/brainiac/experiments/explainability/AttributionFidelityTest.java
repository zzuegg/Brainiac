package io.github.zzuegg.brainiac.experiments.explainability;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.explainability.ExplanationEvaluation.FidelityRow;
import io.github.zzuegg.brainiac.experiments.explainability.ExplanationEvaluation.Method;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class AttributionFidelityTest {

    @Test
    void smoothingLowersSensitivity() {
        List<FidelityRow> rows = ExplanationEvaluation.fidelity(30, 1);
        Map<Method, FidelityRow> by = new EnumMap<>(Method.class);
        for (FidelityRow r : rows) {
            by.put(r.method(), r);
            System.out.printf("Fidelity %-20s infidelity %.3f max-sensitivity %.3f%n", r.method(), r.infidelity(),
                    r.maxSensitivity());
        }
        assertTrue(by.get(Method.SMOOTHGRAD).maxSensitivity() < by.get(Method.SALIENCY).maxSensitivity(),
                "SmoothGrad is less sensitive than saliency");
        assertTrue(by.get(Method.RANDOM).infidelity() > by.get(Method.SALIENCY).infidelity(),
                "random attributions are less faithful than saliency");
    }
}
