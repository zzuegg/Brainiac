package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class TopologicalAutoencodersTest {

    @Test
    void topologicalLossKeepsTheSpheresApart() {
        Map<String, double[]> r = TopologicalAutoencoders.nestedSpheres(80, 2000, 5.0, 1);
        System.out.printf("Nested spheres 10-D->2-D {trust, 5-NN acc, KL0.1, topo err, H0 bottleneck}: %s%n",
                ProbabilisticProgramsTest.line(r));
        assertTrue(r.get("topological AE")[2] < r.get("plain AE")[2], "the latent keeps the input density (KL_0.1)");
    }
}
