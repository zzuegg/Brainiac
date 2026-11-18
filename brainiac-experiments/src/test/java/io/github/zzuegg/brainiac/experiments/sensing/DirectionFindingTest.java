package io.github.zzuegg.brainiac.experiments.sensing;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class DirectionFindingTest {

    @Test
    void classical() {
        Map<String, double[]> r = DirectionFinding.classical(10, 100, 30, 1500, 3);
        String s = AdaptiveFiltersTest.line(r, "%.2f");
        System.out.printf("13.4.14 direction finding M=10 T=100 K=2 {rmse deg at -5/0/10/20 dB; resolution at 2/4/8/16"
                + " deg; count accuracy}: %s%n", s);
        assertTrue(r.get("rmse / MUSIC")[2] < 3 * r.get("rmse / CRB")[2] + 0.1, s);
        assertTrue(r.get("rmse / ESPRIT")[3] < 1, s);
        assertTrue(r.get("resolution / MUSIC")[1] > r.get("resolution / DELAY_AND_SUM")[1], s);
        assertTrue(r.get("coherent / MUSIC + FB spatial smoothing")[0] < r.get("coherent / MUSIC")[0], s);
        assertTrue(r.get("count / MDL")[3] > 0.9, s);
    }
}
