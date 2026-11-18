package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 8, 8.4.17: Mixture-of-Agents against voting, debate and the best single agent. */
class ToolAgentMixtureTest {

    @Test
    void mixtureCombinesComplementarySkills() {
        Map<String, ToolAgentBench.MixtureRow> r = ToolAgentBench.mixture(300, 150, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format(" %s %.2f (%.0f tok);", k, v.accuracy(), v.tokens())));
        System.out.printf("8.4.17 accuracy (tokens/question):%s%n", sb);
        assertTrue(r.get("complementary: Mixture-of-Agents").accuracy()
                > r.get("complementary: majority vote").accuracy() + 0.2, "MoA beats the vote");
        assertTrue(r.get("complementary: Mixture-of-Agents").accuracy()
                > r.get("complementary: best single agent").accuracy() + 0.2, "MoA beats the best agent");
    }
}
