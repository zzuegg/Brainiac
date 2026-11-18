package io.github.zzuegg.brainiac.experiments.sensing;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class DirectionFindingTest {

    @Test
    void classical() {
        Map<String, double[]> r = DirectionFinding.classical(10, 100, 20, 1000, 3);
        String s = AdaptiveFiltersTest.line(r, "%.2f");
        System.out.printf("13.4.14 direction finding M=10 T=100 K=2 {rmse deg at -5/0/10/20 dB; resolution at 2/4/8/16"
                + " deg; count accuracy}: %s%n", s);
        assertTrue(r.get("rmse / MUSIC")[2] < 3 * r.get("rmse / CRB")[2] + 0.1, s);
        assertTrue(r.get("rmse / ESPRIT")[3] < 1, s);
        assertTrue(r.get("resolution / MUSIC")[1] > r.get("resolution / DELAY_AND_SUM")[1], s);
        assertTrue(r.get("coherent / MUSIC + FB spatial smoothing")[0] < r.get("coherent / MUSIC")[0], s);
        assertTrue(r.get("count / MDL")[3] > 0.9, s);
    }

    @Test
    void subspaceNet() {
        Map<String, double[]> r = DirectionFinding.learned(8, 20, 600, 20, 5);
        String s = AdaptiveFiltersTest.line(r, "%.2f");
        System.out.printf("13.4.15 SubspaceNet M=8 T=20 coherent K=2 {far rmse deg at 10 dB/0 dB/miscal 0.1;"
                + " near angle rmse deg/range rmse lambda}: %s%n", s);
        assertTrue(r.get("far / SubspaceNet + MUSIC")[0] < r.get("far / MUSIC (sample covariance)")[0], s);
        assertTrue(r.get("parameters")[0] <= 50_000, s);
    }
}
