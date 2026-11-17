package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer.Variant;
import io.github.zzuegg.brainiac.experimental.llm.QuantizedRollouts;
import org.junit.jupiter.api.Test;

/**
 * Wave 5, 3.10: rollouts sampled from a 2-bit quantized copy of the policy, with and without
 * truncated importance sampling (cap 2), against on-policy GRPO; from scratch, 300 steps, seed 1.
 */
class RolloutMismatchTest {

    @Test
    void quantizedRolloutsStillTrainAndTisCorrectsTheirWeight() {
        QuantizedRollouts q2 = new QuantizedRollouts(2);
        RlvrRecipes.Run[] runs = {
            RlvrRecipes.train("on-policy", ArithmeticTask.addition(4), g -> g.variant(Variant.GRPO), 300, 1, 0),
            RlvrRecipes.train("2-bit", ArithmeticTask.addition(4), g -> g.variant(Variant.GRPO).behaviour(q2), 300, 1, 0),
            RlvrRecipes.train("2-bit+TIS", ArithmeticTask.addition(4),
                    g -> g.variant(Variant.GRPO).behaviour(q2).truncatedImportance(2), 300, 1, 0),
        };
        for (RlvrRecipes.Run r : runs) {
            System.out.printf("%-9s learner greedy %.2f, sampler reward %.2f, mean KL(sampler||learner) %.4f nats/token%n",
                    r.name(), r.after(), r.finalReward(), r.behaviourKl());
        }
        assertTrue(runs[1].behaviourKl() > 0.01, "2-bit rollouts are measurably off-policy");
        assertTrue(runs[0].behaviourKl() == 0, "on-policy has no mismatch");
        assertTrue(runs[1].after() > 0.5 && runs[2].after() > 0.5, "both still learn");
    }
}
