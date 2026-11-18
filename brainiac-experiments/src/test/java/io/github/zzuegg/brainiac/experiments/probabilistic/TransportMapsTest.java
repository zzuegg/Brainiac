package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class TransportMapsTest {

    @Test
    void icnnLearnsTheMongeMap() {
        Map<String, double[]> g = TransportMaps.gaussian(300, 1);
        System.out.printf("Gaussian OT map {L2-UVP %%}: %s%n", ProbabilisticProgramsTest.line(g));
        Map<String, double[]> m = TransportMaps.moonsToRings(300, 2);
        System.out.printf("Moons->rings map {sliced W, cost}: %s%n", ProbabilisticProgramsTest.line(m));
        assertTrue(g.get("ICNN Monge map")[0] < 10, "the ICNN recovers the Gaussian map");
        assertTrue(m.get("ICNN Monge map")[0] < m.get("identity")[0], "the ICNN moves mass to the rings");
    }
}
