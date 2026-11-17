package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.BudgetForcing;
import io.github.zzuegg.brainiac.experimental.llm.LogitMath;
import io.github.zzuegg.brainiac.experimental.llm.LogitsProcessor;

import org.junit.jupiter.api.Test;

class ThinkingBudgetTest {

    @Test
    void forcingLongerThinkingRaisesAccuracy() {
        ThinkingBudget.Result r = ThinkingBudget.run(400, 1);
        System.out.printf("Budget forcing: accuracy by thinking budget %s vs unconstrained %.2f (%.2f thoughts)%n",
                r.byBudget(), r.unconstrained(), r.unconstrainedThinking());
        assertTrue(r.byBudget().get(4) > r.unconstrained() + 0.3);
        assertTrue(r.byBudget().get(4) > r.byBudget().get(1));
        assertTrue(r.byBudget().get(4) >= 0.8);
    }

    @Test
    void processorSuppressesForcesAndWaits() {
        int end = 3;
        LogitsProcessor p = BudgetForcing.processor(1, end, 2, 4, 5);
        float[] l = {0, 0, 0, 9, 0, 0};
        p.process(new int[] {7, 1}, l);
        assertEquals(5, LogitMath.argmax(l), "an early end becomes the wait token");
        float[] m = {0, 1, 0, 0, 0, 0};
        BudgetForcing.processor(1, end, 2, 4, -1).process(new int[] {7, 1, 1, 1, 1}, m);
        assertEquals(end, LogitMath.argmax(m), "the end is forced at the budget");
        float[] a = {0, 1, 0, 0, 0, 0};
        p.process(new int[] {7, 1, end}, a);
        assertEquals(1, LogitMath.argmax(a), "answers are left alone");
    }
}
