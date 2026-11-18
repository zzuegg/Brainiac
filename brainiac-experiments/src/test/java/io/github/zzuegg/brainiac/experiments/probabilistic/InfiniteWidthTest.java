package io.github.zzuegg.brainiac.experiments.probabilistic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class InfiniteWidthTest {

    @Test
    void wideNetworksFollowTheNtk() {
        Map<String, double[]> r = InfiniteWidth.ntkAgreement(new int[] {16, 64, 256, 1024, 2048}, 400, 1);
        System.out.printf("NTK agreement {RMS gap to NTK predictor, relative NTK change}: %s%n",
                ProbabilisticProgramsTest.line(r));
        assertTrue(r.get("width=2048")[0] < r.get("width=16")[0], "wider networks are closer to the NTK predictor");
        assertTrue(r.get("width=2048")[1] < r.get("width=16")[1], "the NTK moves less when wider");
    }

    @Test
    void kernelsOnSpiralsAndTheGap() {
        Map<String, double[]> s = InfiniteWidth.spirals(2);
        System.out.printf("Spirals test accuracy: %s%n", ProbabilisticProgramsTest.line(s));
        Map<String, double[]> g = InfiniteWidth.gapNll(3);
        System.out.printf("Gap NLL {data, gap}: %s%n", ProbabilisticProgramsTest.line(g));
        assertTrue(s.get("NTK (2 hidden)")[0] > 0.7, "NTK regression solves the spirals");
        assertTrue(Double.isFinite(g.get("NNGP GP")[1]));
    }
}
