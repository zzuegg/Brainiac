package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class VqGanExperimentTest {

    @Test
    void vqGanWithFewerCodesVsVqVae() {
        Map<String, Double> m = VqGanExperiment.run(200, 1);
        StringBuilder line = new StringBuilder("ToyImages 16-token autoencoders:");
        m.forEach((k, v) -> line.append(String.format(" %s %.3f;", k, v)));
        System.out.println(line);
        assertTrue(m.get("VQ-GAN 4 MSE") < 0.5, "VQ-GAN reconstructs");
        assertTrue(m.get("VQ-GAN 4 sharp") >= m.get("VQ-VAE 4 sharp") - 0.02, "VQ-GAN is at least as sharp");
    }
}
