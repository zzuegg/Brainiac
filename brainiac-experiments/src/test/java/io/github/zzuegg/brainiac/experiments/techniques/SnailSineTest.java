package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class SnailSineTest {

    @Test
    void snailRegressesSinesFromTheSupportSequence() {
        SineMetaRegression.Result snail = SineMetaRegression.run(SineMetaRegression.Learner.SNAIL, 700, 1);
        SineMetaRegression.Result maml = SineMetaRegression.run(SineMetaRegression.Learner.MAML, 1000, 1);
        System.out.printf("SNAIL sine 10-shot MSE %.3f vs MAML %.3f (untrained 4.10)%n", snail.mse(), maml.mse());
        assertTrue(snail.mse() < 0.7 * 4.1, "SNAIL beats an untrained net: " + snail);
    }
}
