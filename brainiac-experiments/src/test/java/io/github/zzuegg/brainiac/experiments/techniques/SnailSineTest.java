package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class SnailSineTest {

    @Test
    void snailRegressesSinesFromTheSupportSequence() {
        // the MAML baseline on the same waves is measured in SineMetaRegressionTest (MSE about 0.65)
        SineMetaRegression.Result snail = SineMetaRegression.run(SineMetaRegression.Learner.SNAIL, 800, 1);
        System.out.printf("SNAIL sine 10-shot MSE %.3f (untrained 4.10; MAML 0.65 in SineMetaRegressionTest)%n",
                snail.mse());
        assertTrue(snail.mse() < 0.75 * 4.1, "SNAIL beats an untrained net: " + snail);
    }
}
