package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class NeverGiveUpExplorationTest {

    @Test
    void neverGiveUpExploresTheSparseMaze() {
        Map<String, NeverGiveUpExploration.Outcome> r = NeverGiveUpExploration.run(6000, 3);
        System.out.printf("W6 3.6 maze cells/goals: NGU %.1f/%.1f, Agent57 %.1f/%.1f vs none %.1f/%.1f, RND %.1f/%.1f,"
                + " ICM %.1f/%.1f, Go-Explore %.1f/%.1f%n", r.get("NGU").cells(), r.get("NGU").goals(),
                r.get("Agent57 (NGU)").cells(), r.get("Agent57 (NGU)").goals(), r.get("none").cells(),
                r.get("none").goals(), r.get("RND").cells(), r.get("RND").goals(), r.get("ICM").cells(),
                r.get("ICM").goals(), r.get("Go-Explore").cells(), r.get("Go-Explore").goals());
        assertTrue(r.get("NGU").cells() > r.get("none").cells(), r.toString());
        assertTrue(r.get("NGU").cells() > r.get("ICM").cells(), r.toString());
    }
}
