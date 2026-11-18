package io.github.zzuegg.brainiac.experiments.causal;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class UpliftTest {

    @Test
    void upliftModelsBeatTheResponseModelOnQini() {
        Map<String, double[]> r = HeterogeneousEffects.uplift(3000, 10, 1, 1);
        StringBuilder line = new StringBuilder("uplift Qini/AUUC/profit/fraction:");
        r.forEach((k, v) -> line.append(String.format(" %s %.4f/%.4f/%.3f/%.2f |", k, v[0], v[1], v[2], v[3])));
        System.out.println(line);
        double response = r.get("response model (forest)")[0];
        assertTrue(r.get("oracle (true uplift)")[0] > response, line.toString());
        assertTrue(r.get("uplift forest (DDP)")[0] > response, line.toString());
        assertTrue(r.get("class transformation (forest)")[0] > response, line.toString());
        assertTrue(r.get("uplift forest (KL)")[2] > r.get("response model (forest)")[2], line.toString());
    }
}
