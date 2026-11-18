package io.github.zzuegg.brainiac.experiments.agents;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 9, 9.2.17: ArCHer with a slow actor, and with the full recipe. */
class ToolAgentArcherFullRecipeTest {

    @Test
    void archerSlowActorAndFullRecipe() {
        Map<String, ToolAgentBench.ArcherRow[]> r = ToolAgentBench.archerRecipe(6000, 2, 1,
                "ArCHer + slow actor, tau 0.01", "ArCHer, full recipe");
        System.out.printf("9.2.17 greedy/sampled success by interactions (see ToolAgentArcherRecipeTest for the"
                + " wave-8 baseline):%s%n", ToolAgentArcherRecipeTest.describe(r));
        assertEquals(2, r.size());
    }
}
