package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ToolUseTest {

    @Test
    void toolformerKeepsUsefulCallsAndTheToolGeneralizes() {
        ToolUse.Result r = ToolUse.run(300, 300, 1);
        System.out.println(r);
        assertTrue(r.keptCorrect() >= 0.9 * r.kept(), "the filter keeps mostly right calls " + r);
        assertTrue(r.rightCallsKept() > 0.5, "and most of the right calls " + r);
        assertTrue(r.plainAccuracy() < 0.3, r.toString());
        assertTrue(r.toolAccuracy() > 0.7, r.toString());
    }
}
