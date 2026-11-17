package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.rl.ActiveInferenceAgent;
import io.github.zzuegg.brainiac.experimental.rl.CuedTMaze;
import org.junit.jupiter.api.Test;

class EpistemicForagingTest {

    @Test
    void agentReadsTheCueAndBelievesIt() {
        ActiveInferenceAgent agent = CuedTMaze.model(0.95, 3).agent(2, 1);
        agent.beginEpisode();
        assertEquals(3, agent.act(new float[] {1, 0, 0, 0, 0}, false)[0]);
        assertEquals(0.5, agent.belief()[0], 1e-9);
        assertEquals(1, agent.act(new float[] {0, 1, 0, 0, 0}, false)[0]);
        assertEquals(0.95, agent.belief()[3 * 2], 1e-9);
    }

    @Test
    void sophisticatedInferenceReadsTheCueClassicPoliciesGamble() {
        EpistemicForaging.Result r = EpistemicForaging.run(200, 1);
        System.out.println("active inference " + r);
        assertTrue(r.sophisticated().cueFirst() > 0.9, r.toString());
        assertTrue(r.sophisticated().rewardRate() > 0.85, r.toString());
        assertTrue(r.classic().rewardRate() < 0.7, r.toString());
    }
}
