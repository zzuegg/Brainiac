package io.github.zzuegg.brainiac.experiments.generative;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Wave 5, 3.12: diffu-GRPO and JustGRPO against SFT only, masked diffusion LM on a+b, parallel decoding. */
class DiffusionArithmeticRlTest {

    @Test
    void rlImprovesTheDiffusionModel() {
        DiffusionArithmeticRl.Result r = DiffusionArithmeticRl.run(300, 150, 1);
        System.out.printf("accuracy on all 100 sums, parallel (left-to-right) decoding: SFT only %.2f (%.2f), diffu-GRPO"
                + " %.2f (%.2f), JustGRPO %.2f (%.2f)%n", r.sft(), r.sftLeftToRight(), r.diffuGrpo(),
                r.diffuGrpoLeftToRight(), r.justGrpo(), r.justGrpoLeftToRight());
        assertTrue(r.justGrpoLeftToRight() > r.sftLeftToRight() + 0.1, "JustGRPO improves its left-to-right policy");
        assertTrue(r.diffuGrpo() > r.sft() + 0.1, "diffu-GRPO improves parallel decoding");
    }
}
