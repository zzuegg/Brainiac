package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class TransportFlowsTest {

    @Test
    void sinkhornCouplingsStraightenFlows() {
        Map<String, double[]> r = TransportFlows.flowPairing(150, 2);
        System.out.printf("Moons->rings CFM {1-step SW, 4-step SW, straightness, s}: %s%n",
                ProbabilisticProgramsTest.line(r));
        assertTrue(r.get("Sinkhorn n=1024")[2] < r.get("independent")[2], "Sinkhorn pairing straightens the flow");
        assertTrue(r.get("Sinkhorn n=1024")[0] < r.get("independent")[0], "better one-step samples");
    }
}
