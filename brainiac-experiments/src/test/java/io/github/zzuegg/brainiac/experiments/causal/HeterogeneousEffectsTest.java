package io.github.zzuegg.brainiac.experiments.causal;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.CausalScenarios;

import org.junit.jupiter.api.Test;

import java.util.Map;

class HeterogeneousEffectsTest {

    static String line(String title, Map<String, Double> r) {
        StringBuilder b = new StringBuilder(title);
        r.forEach((k, v) -> b.append(String.format(" %s %.3f |", k, v)));
        return b.toString();
    }

    @Test
    void xLearnerBeatsTheTLearnerWithFewTreated() {
        Map<String, Double> r = HeterogeneousEffects.metaLearners(CausalScenarios.Setting.UNBALANCED, 1000, 20, 150,
                1);
        String line = line("CATE PEHE, 10% treated:", r);
        System.out.printf("%s%n", line);
        assertTrue(r.get("X-learner (forest)") < r.get("T-learner (forest)"), line);
    }

    @Test
    void residualLearnersBeatTheTLearnerWhenTheEffectIsSimple() {
        Map<String, Double> r = HeterogeneousEffects.metaLearners(CausalScenarios.Setting.SIMPLE_EFFECT, 1000, 20,
                150, 1);
        String line = line("CATE PEHE, complex outcome / simple effect:", r);
        System.out.printf("%s%n", line);
        // With a 20-tree forest as the final stage, R and DR lose to T (noisy pseudo-outcomes); the
        // claim holds when the final stage matches the simple effect.
        double best = Math.min(r.get("R-learner (forest nuisances, ridge effect)"),
                r.get("DR-learner (forest nuisances, ridge effect)"));
        assertTrue(best < r.get("T-learner (forest)"), line);
    }
}
