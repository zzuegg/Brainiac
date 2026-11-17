package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import org.junit.jupiter.api.Test;

/** Critic-free REINFORCE-style recipes against GRPO, all at Adam 3e-3 and KL 0.05, seed 1. */
class ArithmeticRlVariantsTest {

    private static ArithmeticRl.Result run(GrpoTrainer.Variant v, long seed) {
        long t0 = System.nanoTime();
        ArithmeticRl.Result r = ArithmeticRl.run(ArithmeticTask.addition(4), v, 300, seed, 3e-3f, 0.05f);
        System.out.printf("%-12s seed %d: greedy accuracy %.2f -> %.2f, final reward %.3f, %.1f s (lr 3e-3, KL 0.05)%n",
                v, seed, r.before(), r.after(), r.finalReward(), (System.nanoTime() - t0) / 1e9);
        return r;
    }

    @Test
    void grpoBaselineAtTheSameSettings() {
        ArithmeticRl.Result r = run(GrpoTrainer.Variant.GRPO, 1);
        assertTrue(r.after() > 0.6, r.toString());
    }

    @Test
    void rlooLearnsWithALeaveOneOutBaseline() {
        ArithmeticRl.Result r = run(GrpoTrainer.Variant.RLOO, 1);
        assertTrue(r.after() > 0.6, r.toString());
    }

    @Test
    void remaxLearnsWithAGreedyBaseline() {
        ArithmeticRl.Result r = run(GrpoTrainer.Variant.REMAX, 1);
        assertTrue(r.after() > 0.6, r.toString());
    }

    @Test
    void reinforcePlusPlusLearnsWithGlobalNormalization() {
        ArithmeticRl.Result r = run(GrpoTrainer.Variant.REINFORCE_PP, 1);
        assertTrue(r.after() > 0.6, r.toString());
    }

    @Test
    void gspoLearnsWithSequenceRatios() {
        ArithmeticRl.Result r = run(GrpoTrainer.Variant.GSPO, 1);
        assertTrue(r.after() > 0.6, r.toString());
    }
}
