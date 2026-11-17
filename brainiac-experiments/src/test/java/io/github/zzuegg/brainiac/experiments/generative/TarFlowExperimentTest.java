package io.github.zzuegg.brainiac.experiments.generative;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class TarFlowExperimentTest {

    @Test
    void aTransformerFlowModelsPatchesAndDenoisesItsSamples() {
        Map<String, Double> r = TarFlowExperiment.run(600, 0.2f, 1);
        String line = String.format("ToyImages: bits/dim TarFlow %.3f vs RealNVP %.3f; nearest-real squared distance"
                + " TarFlow %.2f (raw %.2f), RealNVP %.2f, DDPM %.2f", r.get("TarFlow bpd"), r.get("RealNVP bpd"),
                r.get("TarFlow distance"), r.get("TarFlow raw distance"), r.get("RealNVP distance"),
                r.get("DDPM distance"));
        System.out.println(line);
        assertTrue(r.get("TarFlow bpd") < r.get("RealNVP bpd"), line);
        assertTrue(r.get("TarFlow distance") < r.get("TarFlow raw distance"), "denoising helps: " + line);
    }
}
