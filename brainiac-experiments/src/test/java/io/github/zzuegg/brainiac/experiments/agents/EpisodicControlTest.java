package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class EpisodicControlTest {

    @Test
    void episodicMemoryLearnsCartPoleFromFewSteps() {
        EpisodicControl.Result r = EpisodicControl.run(1500, 1);
        System.out.println("NEC vs DQN " + r);
        assertTrue(r.nec() > 60, r.toString());
        assertTrue(r.nec() > r.dqn(), r.toString());
    }
}
