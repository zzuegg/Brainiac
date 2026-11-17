package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer.Variant;
import io.github.zzuegg.brainiac.experimental.llm.PassAtK;
import org.junit.jupiter.api.Test;

/** Wave 5, 3.5: pass@k training (k = 4, analytic advantages) against pass@1 GRPO, from scratch, 300 steps. */
class PassAtKTrainingTest {

    @Test
    void passAtOneAdvantagesAreGrposPopulationStdAdvantages() {
        float[] r = {1, 0, 0, 1, 0, 0, 0, 0};
        float[] a = GrpoTrainer.passAtKAdvantages(r, 1, 0.5f);
        // mean 0.25, population std sqrt(0.25 * 0.75)
        double s = Math.sqrt(0.25 * 0.75);
        assertEquals((1 - 0.25) / s, a[0], 1e-5);
        assertEquals(-0.25 / s, a[1], 1e-5);
        assertArrayEquals(new float[8], GrpoTrainer.passAtKAdvantages(new float[8], 4, 0.5f));
        assertEquals(0.5, PassAtK.estimate(4, 1, 2), 1e-9);   // 1 - C(3,2)/C(4,2)
    }

    @Test
    void passAtKTrainingRaisesPassAtSixteen() {
        RlvrRecipes.Run p1 = RlvrRecipes.train("pass@1 GRPO", ArithmeticTask.addition(4), g -> g.variant(Variant.GRPO),
                300, 1, 32, 1, 16);
        RlvrRecipes.Run pk = RlvrRecipes.train("pass@4 training", ArithmeticTask.addition(4),
                g -> g.variant(Variant.GRPO).passAtK(4), 300, 1, 32, 1, 16);
        for (RlvrRecipes.Run r : new RlvrRecipes.Run[] {p1, pk}) {
            System.out.printf("%-15s pass@1 %.2f, pass@16 %.2f, greedy %.2f, final entropy %.3f%n",
                    r.name(), r.pass(1), r.pass(16), r.after(), r.finalEntropy());
        }
        assertTrue(pk.pass(16) > p1.pass(16), "pass@16 " + pk.pass(16) + " vs " + p1.pass(16));
        assertTrue(pk.finalEntropy() > p1.finalEntropy(), "pass@k training keeps exploring");
    }
}
