package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer.Variant;
import org.junit.jupiter.api.Test;

/** Wave 5, 3.2: policy gradient on the top-20% entropy tokens only, against all-token GRPO, from a weak SFT base. */
class ForkingTokensTest {

    @Test
    void twentyPercentOfTokensMatchAllTokenGrpo() {
        ArithmeticTask task = ArithmeticTask.addition(6);
        RlvrRecipes.Run all = RlvrRecipes.train("all tokens", task, RlvrRecipes.base(task, 60, 0.5, 1),
                g -> g.variant(Variant.GRPO), task::reward, 200, 1, 3e-3f, 0.05f, 64, 1, 64);
        RlvrRecipes.Run fork = RlvrRecipes.train("top 20%", task, RlvrRecipes.base(task, 60, 0.5, 1),
                g -> g.variant(Variant.GRPO).forkingTokens(0.2f), task::reward, 200, 1, 3e-3f, 0.05f, 64, 1, 64);
        for (RlvrRecipes.Run r : new RlvrRecipes.Run[] {all, fork}) {
            System.out.printf("%-10s greedy %.2f -> %.2f, pass@1 %.2f, pass@64 %.2f, tokens updated %.2f, final entropy %.3f%n",
                    r.name(), r.before(), r.after(), r.pass(1), r.pass(64), r.updatedFraction(), r.finalEntropy());
        }
        assertEquals(0.2, fork.updatedFraction(), 0.02);
        assertTrue(fork.after() >= fork.before(), "80/20 does not lose accuracy: " + fork.after());
        assertTrue(fork.after() > all.after() - 0.1, "within 0.1 of all-token GRPO: " + fork.after() + " vs " + all.after());
    }
}
