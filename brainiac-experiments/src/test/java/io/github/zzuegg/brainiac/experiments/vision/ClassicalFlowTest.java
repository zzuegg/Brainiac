package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ClassicalFlowTest {

    @Test
    void hornSchunckTracksMovingShapes() {
        double[][] small = new double[3][];
        double[][] large = new double[3][];
        for (DensePrediction.FlowMethod m : DensePrediction.FlowMethod.values()) {
            small[m.ordinal()] = DensePrediction.classicalFlow(m, false, 0.02f, 1);
            large[m.ordinal()] = DensePrediction.classicalFlow(m, true, 0.02f, 1);
        }
        System.out.printf("flow EPE all/moving pixels, 0-4 px: zero %.3f/%.3f, Horn-Schunck %.3f/%.3f, Lucas-Kanade"
                + " %.3f/%.3f; 6-10 px: zero %.3f/%.3f, Horn-Schunck %.3f/%.3f, Lucas-Kanade %.3f/%.3f%n",
                small[0][0], small[0][1], small[1][0], small[1][1], small[2][0], small[2][1], large[0][0], large[0][1],
                large[1][0], large[1][1], large[2][0], large[2][1]);
        assertTrue(small[1][1] < small[0][1], "Horn-Schunck beats zero flow on the moving shapes");
        assertTrue(small[2][1] < small[0][1], "Lucas-Kanade beats zero flow on the moving shapes");
    }
}
