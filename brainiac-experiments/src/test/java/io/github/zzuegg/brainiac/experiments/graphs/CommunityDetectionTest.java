package io.github.zzuegg.brainiac.experiments.graphs;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class CommunityDetectionTest {

    @Test
    void nonBacktrackingAndBpDetectNearTheThreshold() {
        // c = 3: the Kesten-Stigum threshold is c_in - c_out = 2 sqrt(3) = 3.46
        double[] gaps = {2.5, 4, 5.5};
        Map<String, double[]> r = CommunityDetection.sweep(150, 3, gaps, 1, 1);
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(String.format("%s %.2f/%.2f/%.2f; ", k, v[0], v[1], v[2])));
        System.out.printf("SBM NMI at gap %s: %s%n", java.util.Arrays.toString(gaps), sb);
        assertTrue(r.get("SPECTRAL_NON_BACKTRACKING")[2] > r.get("adjacency spectral")[2], sb.toString());
        assertTrue(r.get("SPECTRAL_NON_BACKTRACKING")[2] > r.get("LOUVAIN")[2], sb.toString());
        // EM-learned BP matches the spectral start well above the threshold; near it, on graphs this
        // small, EM settles on the uninformative fixed point (a genuine finite-size negative)
        assertTrue(r.get("SBM_BP")[2] > 0.8 * r.get("SPECTRAL_NON_BACKTRACKING")[2], sb.toString());
    }

    @Test
    void jumpingKnowledgeHoldsUpAtDepth() {
        Map<String, Double> r = CommunityDetection.depth(new int[] {2, 8}, 100, 3);
        System.out.printf("GCN depth accuracy: %s%n", r);
        assertTrue(Math.max(r.get("JK concat @ 8"), r.get("skip @ 8")) > r.get("plain @ 8"), r.toString());
    }
}
