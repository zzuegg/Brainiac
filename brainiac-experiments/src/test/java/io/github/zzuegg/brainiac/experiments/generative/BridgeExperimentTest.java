package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class BridgeExperimentTest {

    @Test
    void bridgesTransportMoonsToRingsAndImfShortensThePaths() {
        Map<String, double[]> m = BridgeExperiment.compare(600, 1);
        StringBuilder line = new StringBuilder("moons -> rings (transport cost / sliced W to rings):");
        m.forEach((k, v) -> line.append(String.format(" %s %.3f/%.3f;", k, v[0], v[1])));
        System.out.println(line);
        double[] bridge = m.get("bridge matching");
        double[] dsbm = m.get("DSBM (IMF)");
        double[] rf = m.get("rectified flow");
        assertTrue(bridge[1] < 0.15 && dsbm[1] < 0.2, "both bridges reach the rings");
        assertTrue(dsbm[0] < bridge[0], "iterative Markovian fitting shortens the transport");
        assertTrue(m.get("reflow")[0] < rf[0], "reflow, the noiseless case, shortens it too");
    }
}
