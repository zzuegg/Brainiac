package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.rl.LevelReplay;
import org.junit.jupiter.api.Test;

class LevelCurriculumTest {

    @Test
    void samplerPrefersHighScoresAndVisitsNewLevelsFirst() {
        LevelReplay s = LevelReplay.of(3, 1).replayProbability(0).staleness(0).temperature(0.1);
        for (int i = 0; i < 3; i++) {
            int level = s.next();
            assertEquals(i, level);
            s.update(level, i == 1 ? 1.0 : 0.01);
        }
        int ones = 0;
        for (int i = 0; i < 100; i++) {
            ones += s.next() == 1 ? 1 : 0;
        }
        assertTrue(ones > 90, "picked level 1 " + ones + " times");
        assertFalse(LevelCurriculum.solvable(new String[] {"S#....", "##....", "......", "......", "......",
            ".....G"}));
    }

    @Test
    void prioritizedLevelReplaySolvesMoreLevelsThanUniformSampling() {
        int plr = 0;
        int uniform = 0;
        for (int seed = 1; seed <= 4; seed++) {
            LevelCurriculum.Result r = LevelCurriculum.run(60, 1200, seed);
            System.out.println("PLR " + r);
            plr += r.plrSolved();
            uniform += r.uniformSolved();
        }
        assertTrue(plr > uniform + 10, plr + " vs " + uniform);
    }
}
