package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.techniques.UncertaintyBenchmark.Method;
import io.github.zzuegg.brainiac.experiments.techniques.UncertaintyBenchmark.Scores;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class UncertaintyOodTest {

    @Test
    void posteriorsAndSinglePassHeadsAgainstEnsemblesDropoutAndLaplace() {
        Map<Method, Scores> all = new EnumMap<>(Method.class);
        for (Method m : Method.values()) {
            Scores s = UncertaintyBenchmark.classification(m, 1);
            all.put(m, s);
            System.out.printf("OOD shapes %-17s acc %.3f NLL %.3f ECE %.3f AUROC %.3f passes %d%n", m, s.accuracy(),
                    s.nll(), s.ece(), s.oodAuroc(), s.forwardPasses());
        }
        Scores map = all.get(Method.MAP);
        for (Map.Entry<Method, Scores> e : all.entrySet()) {
            assertTrue(e.getValue().accuracy() > 0.75, e.getKey() + " learns the task: " + e.getValue());
            assertTrue(e.getValue().oodAuroc() > 0.6, e.getKey() + " flags OOD shapes: " + e.getValue());
        }
        assertTrue(all.get(Method.SWAG).nll() < map.nll(), "SWAG improves on MAP's NLL");
        assertTrue(all.get(Method.DEEP_ENSEMBLE).nll() < map.nll(), "the ensemble improves on MAP's NLL");
        assertTrue(all.get(Method.SNGP).oodAuroc() > map.oodAuroc(), "SNGP flags OOD better than MAP in one pass");
    }
}
