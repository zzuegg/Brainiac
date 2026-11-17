package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DreamingAgentTest {

    @Test
    void dreamerLearnsCartPoleInImagination() {
        DreamingAgent.DreamResult r = DreamingAgent.dreamer(3000, 1);
        System.out.println("dreamer " + r);
        assertTrue(r.after() > 2 * r.randomReturn(), r.toString());
    }

    @Test
    void jepaPredictsTheNextEmbeddingWithoutCollapse() {
        DreamingAgent.JepaResult r = DreamingAgent.jepa(2000, 400, 1);
        System.out.println("jepa " + r);
        assertTrue(r.modelError() < 0.5 * r.noChangeError(), r.toString());
        assertTrue(r.latentStd() > 0.3, r.toString());
    }
}
