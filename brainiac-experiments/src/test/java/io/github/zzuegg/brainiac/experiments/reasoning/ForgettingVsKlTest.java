package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experiments.reasoning.ForgettingVsKl.Method;
import org.junit.jupiter.api.Test;

/** Wave 5, 3.18: SFT against GRPO towards the same new-task accuracy: forgetting and KL from the base. */
class ForgettingVsKlTest {

    @Test
    void forgettingFollowsKlFromTheBase() {
        ArithmeticTask oldTask = ArithmeticTask.multiplication(3);
        ArithmeticTask newTask = ArithmeticTask.addition(6);
        WindowLm base = ForgettingVsKl.base(oldTask, newTask, 150, 30, 1);
        ForgettingVsKl.Result[] rs = new ForgettingVsKl.Result[3];
        for (Method m : Method.values()) {
            ForgettingVsKl.Result r = ForgettingVsKl.run(base, oldTask, newTask, m, 0.65, 400, 1e-3f, 0, 1);
            rs[m.ordinal()] = r;
            System.out.printf("%-10s %3d steps: new task %.2f, old task %.2f -> %.2f (forgot %.2f), KL(base||model) on new"
                    + " task %.3f, on old %.3f%n", r.method(), r.steps(), r.newAccuracy(), r.baseOld(), r.oldAccuracy(),
                    r.forgetting(), r.klNew(), r.klOld());
        }
        ForgettingVsKl.Result sft = rs[Method.SFT.ordinal()];
        ForgettingVsKl.Result grpo = rs[Method.GRPO.ordinal()];
        assertTrue(grpo.klNew() < sft.klNew(), "GRPO ends closer to the base than SFT on annotator-style labels");
        assertTrue(grpo.forgetting() < sft.forgetting(), "and forgets less");
    }
}
