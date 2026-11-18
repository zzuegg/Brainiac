package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experiments.reasoning.SelfDistillationVsSft.Method;
import io.github.zzuegg.brainiac.experiments.reasoning.SelfDistillationVsSft.Result;

import org.junit.jupiter.api.Test;

class SelfDistillationVsSftTest {

    @Test
    void selfDistillationLearnsTheNewFormatWithLessForgetting() {
        ArithmeticTask mul = ArithmeticTask.multiplication(3);
        ArithmeticTask add = ArithmeticTask.addition(6);
        WindowLm base = SelfDistillationVsSft.base(mul, add, 300, 1);
        Result sft = SelfDistillationVsSft.run(base, mul, add, Method.SFT, 0.6, 300, 1);
        Result sdft = SelfDistillationVsSft.run(base, mul, add, Method.SDFT, 0.6, 300, 1);
        Result grpo = SelfDistillationVsSft.run(base, mul, add, Method.GRPO, 0.6, 150, 1);
        System.out.printf("padded addition (base in-context %.2f, without demo %.2f): SDFT %.2f in %d steps, old task "
                + "%.2f -> %.2f | SFT %.2f in %d steps, old %.2f | GRPO %.2f in %d steps, old %.2f%n",
                sdft.inContext(), SelfDistillationVsSft.paddedAccuracy(base, add, false, 0), sdft.newSkill(),
                sdft.steps(), sdft.oldBefore(), sdft.oldTask(), sft.newSkill(), sft.steps(), sft.oldTask(),
                grpo.newSkill(), grpo.steps(), grpo.oldTask());
        assertTrue(sdft.inContext() > 0.5, "the base copies the demonstrated format in context");
        assertTrue(sdft.newSkill() >= 0.6, "SDFT learns the skill without the demonstration: " + sdft);
        assertTrue(sdft.forgetting() <= sft.forgetting() + 0.02, "SDFT forgets no more than SFT");
    }
}
