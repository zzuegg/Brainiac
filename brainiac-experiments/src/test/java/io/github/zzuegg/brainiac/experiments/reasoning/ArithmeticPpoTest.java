package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.PpoLmTrainer;
import org.junit.jupiter.api.Test;

/** RLHF-style PPO with a critic against GRPO, at equal samples per step. */
class ArithmeticPpoTest {

    @Test
    void gaeMatchesTheHandComputedRecursion() {
        // gamma 1, lambda 0.5, values 0.5 and 0.25, reward only at the end
        double[] a = PpoLmTrainer.gae(new double[] {0, 1}, new double[] {0.5, 0.25}, 1, 0.5);
        // delta1 = 1 - 0.25 = 0.75; delta0 = 0 + 0.25 - 0.5 = -0.25; a0 = -0.25 + 0.5 * 0.75
        assertArrayEquals(new double[] {0.125, 0.75}, a, 1e-12);
    }

    @Test
    void ppoWithACriticLearnsLikeGrpo() {
        ArithmeticTask task = ArithmeticTask.addition(4);
        long t0 = System.nanoTime();
        ArithmeticRl.Result ppo = ArithmeticRl.ppo(task, 300, 1);
        double ppoSeconds = (System.nanoTime() - t0) / 1e9;
        t0 = System.nanoTime();
        ArithmeticRl.Result grpo = ArithmeticRl.run(task, GrpoTrainer.Variant.GRPO, 300, 1);
        double grpoSeconds = (System.nanoTime() - t0) / 1e9;
        System.out.printf("PPO+critic greedy accuracy %.2f (%.1f s) vs GRPO %.2f (%.1f s), 128 samples/step%n",
                ppo.after(), ppoSeconds, grpo.after(), grpoSeconds);
        assertTrue(ppo.before() < 0.15 && ppo.after() > 0.5, ppo.toString());
    }
}
