package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.experimental.nn.HierarchicalVae;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class HierarchicalVaeExperimentTest {
    static final int STEPS = 800;

    @Test
    void iwaeAndLadderVsPlainVae() {
        HierarchicalVaeExperiment.Score plain = HierarchicalVaeExperiment.run(HierarchicalVae.plain(64, 64, 8, 1), 1f,
                1, STEPS, 1);
        HierarchicalVaeExperiment.Score beta = HierarchicalVaeExperiment.run(HierarchicalVae.plain(64, 64, 8, 1), 4f,
                1, STEPS, 1);
        HierarchicalVaeExperiment.Score iwae = HierarchicalVaeExperiment.run(HierarchicalVae.plain(64, 64, 8, 1), 1f,
                5, STEPS, 1);
        HierarchicalVaeExperiment.Score ladder = HierarchicalVaeExperiment.run(HierarchicalVae.ladder(64, 64, 8, 4, 1),
                1f, 1, STEPS, 1);
        System.out.printf("binary ToyImages (IWAE-500 NLL nats / active units / disentanglement gap): plain VAE "
                + "%.2f/%d/%.2f, beta-VAE(4) %.2f/%d/%.2f, IWAE(k=5) %.2f/%d/%.2f, ladder %.2f/%d/%.2f%n",
                plain.nll(), plain.activeUnits(), plain.disentanglement(), beta.nll(), beta.activeUnits(),
                beta.disentanglement(), iwae.nll(), iwae.activeUnits(), iwae.disentanglement(), ladder.nll(),
                ladder.activeUnits(), ladder.disentanglement());
        // 64 independent fair coins would cost 44.4 nats; the data set has 73 images (4.3 nats)
        assertTrue(plain.nll() < 30 && ladder.nll() < 30, "VAEs model the images");
        assertTrue(iwae.nll() < plain.nll() + 0.5, "IWAE at least matches the ELBO-trained VAE");
        assertTrue(beta.activeUnits() <= plain.activeUnits(), "beta-VAE uses no more units");
    }
}
