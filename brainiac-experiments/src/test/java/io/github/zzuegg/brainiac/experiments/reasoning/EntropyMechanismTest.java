package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer.EntropyControl;
import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer.Variant;
import java.util.function.UnaryOperator;
import org.junit.jupiter.api.Test;

/** Wave 5, 3.1: Clip-Cov and KL-Cov against GRPO and clip-higher (DAPO), from scratch, 300 steps, seed 1. */
class EntropyMechanismTest {

    private static RlvrRecipes.Run run(String name, UnaryOperator<GrpoTrainer> c) {
        RlvrRecipes.Run r = RlvrRecipes.train(name, ArithmeticTask.addition(4), c, 300, 1, 16, 1, 8);
        System.out.printf("%-8s policy entropy %.2f -> %.3f, greedy %.2f, pass@1 %.2f, pass@8 %.2f, tokens with gradient %.2f%n",
                name, r.entropy()[0], r.finalEntropy(), r.after(), r.pass(1), r.pass(8), r.updatedFraction());
        return r;
    }

    @Test
    void covarianceControlKeepsMoreEntropyThanGrpo() {
        RlvrRecipes.Run grpo = run("GRPO", g -> g.variant(Variant.GRPO));
        RlvrRecipes.Run dapo = run("DAPO", g -> g.variant(Variant.DAPO));
        RlvrRecipes.Run clipCov = run("Clip-Cov", g -> g.variant(Variant.GRPO)
                .entropyControl(EntropyControl.CLIP_COV, 0.2f, 1, 5, 1));
        RlvrRecipes.Run klCov = run("KL-Cov", g -> g.variant(Variant.GRPO)
                .entropyControl(EntropyControl.KL_COV, 0.2f, 0, 0, 1));
        for (RlvrRecipes.Run r : new RlvrRecipes.Run[] {grpo, dapo, clipCov, klCov}) {
            assertTrue(r.after() > 0.6, r.name() + " " + r.after());
        }
        assertTrue(clipCov.finalEntropy() > grpo.finalEntropy(), "Clip-Cov entropy " + clipCov.finalEntropy());
        assertTrue(klCov.finalEntropy() > grpo.finalEntropy(), "KL-Cov entropy " + klCov.finalEntropy());
        assertTrue(clipCov.updatedFraction() < 1, "Clip-Cov detaches some tokens");
    }
}
