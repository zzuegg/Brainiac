package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class CycleGanTest {

    @Test
    void cycleGanKeepsTheContent() {
        Map<String, Double> m = ImageTranslation.unpaired(160, 1);
        StringBuilder line = new StringBuilder("squares -> crosses, unpaired:");
        m.forEach((k, v) -> line.append(String.format(" %s %.3f;", k, v)));
        System.out.println(line);
        assertTrue(m.get("CycleGAN cycle L1") < m.get("no cycle cycle L1"), "the cycle loss makes it invertible");
        assertTrue(m.get("CycleGAN target share") > 0.5, "CycleGAN draws crosses");
        assertTrue(m.get("CycleGAN centroid error") < m.get("no cycle centroid error"), "and keeps their place");
    }
}
