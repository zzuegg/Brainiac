package io.github.zzuegg.brainiac.experiments.evolution;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.rl.WindyReach;
import io.github.zzuegg.brainiac.rl.Environment;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class PoetCurriculumTest {

    @Test
    void windPushesAnIdleAgentBack() {
        WindyReach env = new WindyReach(2, -1, 20);
        env.reset(new SplittableRandom(1));
        Environment.Step s = null;
        for (int i = 0; i < 20; i++) {
            s = env.step(new float[] {0});
        }
        assertTrue(s.observation()[0] > 2, "drifted away from the target");
        assertTrue(s.truncated());
    }

    @Test
    void poetGrowsHarderEnvironmentsAndSolvesThem() {
        PoetCurriculum.Outcome o = PoetCurriculum.run(60, 2);
        System.out.printf("POET: %d environments, hardest target %.2f wind %.2f, poet %.3f, direct %.3f, %d transfers%n",
                o.environments(), o.hardest()[0], o.hardest()[1], o.poetScore(), o.directScore(), o.transfers());
        assertTrue(o.environments() >= 3, "environments " + o.environments());
        assertTrue(PoetCurriculum.difficulty(o.hardest()) > PoetCurriculum.difficulty(new double[] {1, 0}) + 1);
        assertTrue(o.poetScore() > -0.5, "poet agent score " + o.poetScore());
    }
}
