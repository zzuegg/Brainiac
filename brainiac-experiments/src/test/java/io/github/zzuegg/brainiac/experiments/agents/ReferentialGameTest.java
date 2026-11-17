package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ReferentialGameTest {

    @Test
    void agentsInventAWorkingCode() {
        ReferentialGame.Result r = ReferentialGame.run(1500, 1);
        System.out.println("referential game " + r);
        assertTrue(r.trainAccuracy() > 0.9, r.toString());
        assertTrue(r.testAccuracy() > 2 * r.chance(), r.toString());
        assertTrue(r.distinctMessages() > 30, r.toString());
        assertTrue(r.topographicSimilarity() > 0.2, r.toString());
    }
}
