package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class LatentDiffusionTest {
    static final int[] CHECKPOINTS = {150, 400, 800};

    @Test
    void latentAndRepresentationDiffusionVsPixels() {
        Map<String, List<GenerativeScores.ImageScore>> m = LatentDiffusion.run(600, CHECKPOINTS, 1);
        StringBuilder line = new StringBuilder("ToyImages diffusion (distance per pixel / real share) at 150/400/800 "
                + "steps:");
        m.forEach((k, v) -> {
            line.append(' ').append(k);
            v.forEach(s -> line.append(String.format(" %.3f/%.2f", s.distance(), s.realShare())));
            line.append(';');
        });
        System.out.println(line);
        double pixels = m.get("pixels").get(0).distance();
        assertTrue(m.get("LDM").get(0).distance() < pixels || m.get("RAE").get(0).distance() < pixels,
                "a latent space trains faster than pixels");
        assertTrue(m.get("RAE").get(2).distance() < m.get("pixels").get(2).distance(),
                "RAE latents beat pixels at the same budget");
    }
}
