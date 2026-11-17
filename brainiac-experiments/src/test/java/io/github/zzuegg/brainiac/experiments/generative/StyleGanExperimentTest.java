package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class StyleGanExperimentTest {

    @Test
    void styleGeneratorVsPlainGenerator() {
        Map<String, StyleGanExperiment.Score> m = StyleGanExperiment.run(500, 0.5f, 1);
        StringBuilder line = new StringBuilder("ToyImages GANs (sliced W / distance / real share / classes / PPL):");
        m.forEach((k, v) -> line.append(String.format(" %s %.3f/%.3f/%.2f/%d/%.1f;", k, v.sliced(),
                v.images().distance(), v.images().realShare(), v.images().classes(), v.pathLength())));
        System.out.println(line);
        StyleGanExperiment.Score style = m.get("StyleGAN");
        assertTrue(style.images().distance() < 0.3, "the style generator draws shapes");
        assertTrue(style.sliced() < 3 * m.get("DCGAN").sliced(), "and is competitive with the plain generator");
        assertTrue(m.get("StyleGAN + PPL reg").pathLength() < style.pathLength(),
                "the regulariser smooths the W map");
    }
}
