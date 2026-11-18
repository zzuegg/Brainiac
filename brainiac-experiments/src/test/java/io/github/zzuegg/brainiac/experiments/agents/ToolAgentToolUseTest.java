package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 8, 8.4.16: RL for tool use (ToolRL, ReTool) against Toolformer and fixed policies. */
class ToolAgentToolUseTest {

    @Test
    void reinforcementLearnsWhenToCall() {
        Map<String, ToolAgentBench.ToolRow> r = ToolAgentBench.toolUse(500, 30, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format(" %s acc %.2f, longer %.2f, calls %.2f, needless %.2f, sums %.2f;",
                k,
                v.accuracy(), v.heldOutSize(), v.callsPerQuestion(), v.needlessCalls(), v.sumAccuracy())));
        System.out.printf("8.4.16 tool use:%s%n", sb);
        assertTrue(r.get("ToolRL (structured reward)").accuracy() > r.get("never call").accuracy() + 0.2,
                "tools help");
        assertTrue(r.get("ToolRL (structured reward)").accuracy() > r.get("GRPO without tools").accuracy() + 0.2,
                "RL with tools beats RL without");
        assertTrue(r.get("ToolRL (structured reward)").accuracy() > r.get("Toolformer SFT").accuracy() - 0.1,
                "RL roughly matches Toolformer");
    }
}
