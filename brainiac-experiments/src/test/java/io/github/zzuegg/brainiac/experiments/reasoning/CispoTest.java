package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer.Variant;
import java.util.Arrays;
import org.junit.jupiter.api.Test;

/** Wave 5, 3.3: CISPO against GRPO and DAPO, from scratch, 300 steps, seed 1. */
class CispoTest {

    @Test
    void cispoLearnsAndKeepsEveryTokensGradient() {
        RlvrRecipes.Run[] runs = {
            RlvrRecipes.train("GRPO", ArithmeticTask.addition(4), g -> g.variant(Variant.GRPO), 300, 1, 0),
            RlvrRecipes.train("DAPO", ArithmeticTask.addition(4), g -> g.variant(Variant.DAPO), 300, 1, 0),
            RlvrRecipes.train("CISPO", ArithmeticTask.addition(4), g -> g.variant(Variant.CISPO), 300, 1, 0),
        };
        for (RlvrRecipes.Run r : runs) {
            System.out.printf("%-6s greedy accuracy every 60 steps %s, tokens outside the clip range %.3f%n",
                    r.name(), Arrays.toString(r.accuracyCurve()), r.clipFraction());
        }
        RlvrRecipes.Run cispo = runs[2];
        assertTrue(cispo.after() > 0.6, "CISPO " + cispo.after());
        assertTrue(cispo.updatedFraction() == 1, "CISPO never masks a token");
    }
}
