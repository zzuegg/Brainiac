package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/** Wave 5, 3.9: LCPO (L1-Exact) against budget forcing on the bisection thinking task. */
class LengthControlledThinkingTest {

    @Test
    void lcpoFollowsTheBudgetInThePrompt() {
        LengthControlledThinking.Result r = LengthControlledThinking.run(300, 300, 0.5f, 1);
        System.out.printf("budget 0..4: LCPO accuracy %s at realized length %s (mean |error| %.2f); budget forcing %s;"
                + " unconstrained %.2f at %.2f thoughts%n", LengthControlledThinking.format(r.lcpoAccuracy()),
                LengthControlledThinking.format(r.lcpoLength()), r.lcpoLengthError(),
                LengthControlledThinking.format(r.forcedAccuracy()), r.unconstrainedAccuracy(),
                r.unconstrainedLength());
        assertTrue(r.lcpoLengthError() < 0.5, "LCPO thinks about as long as asked");
        assertTrue(r.lcpoAccuracy()[4] > r.unconstrainedAccuracy(), "a full budget beats the model's own habit");
        assertTrue(r.lcpoAccuracy()[4] > r.lcpoAccuracy()[1], "accuracy grows with the budget asked for");
    }
}
