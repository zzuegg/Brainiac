package io.github.zzuegg.brainiac.experiments.evolution;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class EvolvedPlasticityTest {

    @Test
    void evolutionDiscoversAHebbianRuleThatLearnsWithinALifetime() {
        EvolvedPlasticity.Outcome o = EvolvedPlasticity.run(1_500, 1);
        System.out.printf("evolved %.3f vs random %.3f, rule %s%n", o.evolvedAccuracy(), o.randomAccuracy(),
                java.util.Arrays.toString(o.rule()));
        assertTrue(o.evolvedAccuracy() > 0.9, "evolved " + o.evolvedAccuracy());
        assertTrue(o.evolvedAccuracy() > o.randomAccuracy() + 0.2);
        assertTrue(o.rule()[1] * Math.abs(o.rule()[0]) > 0, "correlation term must be Hebbian, not anti-Hebbian");
    }
}
