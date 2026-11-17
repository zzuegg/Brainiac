package io.github.zzuegg.brainiac.experiments.evolution;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class PopulationEnsemblesTest {

    @Test
    void ensembleAbstainsOnItsMistakesAndDisagreesOffTheData() {
        PopulationEnsembles.Outcome o = PopulationEnsembles.run(4, 1);
        System.out.println(o);
        assertTrue(o.members() >= 2);
        assertTrue(o.halfRisk() < 0.5 * o.fullRisk(), o.halfRisk() + " vs " + o.fullRisk());
        assertTrue(o.outOfDistributionDisagreement() > o.inDistributionDisagreement());
    }
}
