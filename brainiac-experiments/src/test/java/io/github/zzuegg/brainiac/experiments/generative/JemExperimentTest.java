package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class JemExperimentTest {

    @Test
    void jemClassifiesCalibratesAndSamples() {
        Map<String, Double> m = JemExperiment.run(400, 1);
        StringBuilder line = new StringBuilder("ShapeImages 8x8 JEM vs plain classifier vs EnergyRefiner:");
        m.forEach((k, v) -> line.append(String.format(" %s %.3f;", k, v)));
        System.out.println(line);
        assertTrue(m.get("JEM accuracy") > m.get("plain accuracy") - 0.05, "JEM keeps the classifier's accuracy");
        assertTrue(m.get("JEM OOD AUROC density") > 0.7, "JEM's density flags shuffled images");
        assertTrue(m.get("JEM buffer sample distance") < m.get("noise distance"),
                "persistent chains move towards the data");
    }
}
