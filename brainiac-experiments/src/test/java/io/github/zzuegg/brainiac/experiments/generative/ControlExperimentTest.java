package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ControlExperimentTest {

    @Test
    void controlBranchFollowsTheMapAndKeepsTheBase() {
        Map<String, Double> m = ControlExperiment.run(800, 300, 1);
        StringBuilder line = new StringBuilder("ToyImages 4x4 occupancy control:");
        m.forEach((k, v) -> line.append(String.format(" %s %.3f;", k, v)));
        System.out.println(line);
        assertTrue(m.get("ControlNet IoU") > m.get("base IoU") + 0.2, "ControlNet follows the control map");
        assertTrue(m.get("ControlNet unconditional distance") <= m.get("base unconditional distance"),
                "the frozen base is untouched");
    }
}
