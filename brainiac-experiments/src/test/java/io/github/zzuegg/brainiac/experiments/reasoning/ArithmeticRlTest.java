package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.GrpoTrainer;
import org.junit.jupiter.api.Test;

class ArithmeticRlTest {

    @Test
    void grpoLearnsAdditionFromRewardAlone() {
        ArithmeticRl.Result r = ArithmeticRl.run(ArithmeticTask.addition(4), GrpoTrainer.Variant.GRPO, 300, 1);
        System.out.println("GRPO " + r);
        assertTrue(r.before() < 0.15, r.toString());
        assertTrue(r.after() > 0.6, r.toString());
    }

    @Test
    void drGrpoLearns() {
        ArithmeticRl.Result r = ArithmeticRl.run(ArithmeticTask.addition(4), GrpoTrainer.Variant.DR_GRPO, 300, 2, 3e-3f, 0.05f);
        System.out.println("Dr.GRPO " + r);
        assertTrue(r.after() > 0.6, r.toString());
    }

    @Test
    void dapoLearns() {
        ArithmeticRl.Result r = ArithmeticRl.run(ArithmeticTask.addition(4), GrpoTrainer.Variant.DAPO, 300, 3);
        System.out.println("DAPO " + r);
        assertTrue(r.after() > 0.6, r.toString());
    }
}
