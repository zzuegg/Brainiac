package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class StreamingRlTest {

    @Test
    void streamActorCriticLearnsWithoutReplay() {
        StreamingRl.Result r = StreamingRl.run(10_000, 1);
        System.out.println("stream AC " + r);
        assertTrue(r.lastEpisodes() > 2 * r.firstEpisodes(), r.toString());
        assertTrue(r.after() > 100, r.toString());
    }
}
