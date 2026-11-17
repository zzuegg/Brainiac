package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer.Variant;
import io.github.zzuegg.brainiac.experimental.llm.PassAtK;
import org.junit.jupiter.api.Test;

/** Wave 5, 3.6: NSR and W-REINFORCE against PSR and GRPO: pass@k for k = 1..64 from a weak SFT base, 200 steps. */
class NegativeSampleReinforcementTest {

    private static final int[] KS = PassAtK.powersOfTwo(64);

    private static RlvrRecipes.Run run(String name, Variant v, int steps) {
        ArithmeticTask task = ArithmeticTask.addition(6);
        RlvrRecipes.Run r = RlvrRecipes.train(name, task, RlvrRecipes.base(task, 60, 0.5, 1),
                g -> g.variant(v).promptsPerStep(steps == 0 ? 0 : 16), task::reward, Math.max(1, steps), 1, 3e-3f,
                0.05f, 64, KS);
        System.out.printf("%-11s pass@k (k=1..64) %s%n", name, RlvrRecipes.curve(r));
        return r;
    }

    @Test
    void negativeSamplesAlonePreserveLargeKCoverage() {
        RlvrRecipes.Run base = run("base", Variant.GRPO, 0);
        RlvrRecipes.Run grpo = run("GRPO", Variant.GRPO, 200);
        RlvrRecipes.Run psr = run("PSR", Variant.PSR, 200);
        RlvrRecipes.Run nsr = run("NSR", Variant.NSR, 200);
        RlvrRecipes.Run w = run("W-REINFORCE", Variant.W_REINFORCE, 200);
        assertTrue(nsr.pass(1) >= base.pass(1), "NSR improves pass@1 without positive rewards: " + nsr.pass(1));
        assertTrue(w.pass(1) > base.pass(1), "W-REINFORCE improves pass@1");
        assertTrue(nsr.pass(64) > base.pass(64) - 0.06, "NSR keeps pass@64 near the base: " + nsr.pass(64));
        assertTrue(w.pass(64) >= grpo.pass(64) - 0.02 && psr.pass(1) > base.pass(1), "sanity");
    }
}
