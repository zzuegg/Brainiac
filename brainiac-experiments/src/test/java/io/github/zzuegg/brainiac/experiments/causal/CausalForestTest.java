package io.github.zzuegg.brainiac.experiments.causal;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class CausalForestTest {

    @Test
    void causalForestRecoversTheStepEffectWithIntervals() {
        Map<String, Double> r = HeterogeneousEffects.causalForest(1500, 60, 1);
        String line = HeterogeneousEffectsTest.line("step effect:", r);
        System.out.println(line);
        // the effect ranges over [1, 4] with standard deviation ~0.8; a constant guess has PEHE ~0.8
        assertTrue(r.get("causal forest PEHE") < 0.75, line);
        assertTrue(r.get("causal forest 90% coverage") > 0.3, line);
    }
}
