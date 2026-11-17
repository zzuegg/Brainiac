package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer.Variant;
import org.junit.jupiter.api.Test;

/** Wave 5, 3.4: GMPO against GRPO and GSPO, from scratch, 300 steps, seed 1. */
class GmpoTest {

    @Test
    void gmpoLearnsWithAWiderClipRange() {
        RlvrRecipes.Run[] runs = {
            RlvrRecipes.train("GRPO", ArithmeticTask.addition(4), g -> g.variant(Variant.GRPO), 300, 1, 0),
            RlvrRecipes.train("GSPO", ArithmeticTask.addition(4), g -> g.variant(Variant.GSPO), 300, 1, 0),
            RlvrRecipes.train("GMPO", ArithmeticTask.addition(4), g -> g.variant(Variant.GMPO), 300, 1, 0),
        };
        for (RlvrRecipes.Run r : runs) {
            System.out.printf("%-5s greedy %.2f, token ratio range [%.2f, %.2f], mean per-step ln(max/min) %.3f%n",
                    r.name(), r.after(), r.ratioMin(), r.ratioMax(), r.ratioSpread());
        }
        RlvrRecipes.Run grpo = runs[0];
        RlvrRecipes.Run gmpo = runs[2];
        assertTrue(gmpo.after() > 0.6, "GMPO " + gmpo.after());
        assertTrue(gmpo.ratioMax() < grpo.ratioMax(), "GMPO's largest ratio " + gmpo.ratioMax() + " vs " + grpo.ratioMax());
    }
}
