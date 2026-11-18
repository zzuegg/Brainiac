package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class SamplingBiasCorrectionTest {

    @Test
    void logQCorrectionRecoversHeadItems() {
        Map<String, double[]> r = SamplingBiasCorrection.run(1.2, 160, 1);
        StringBuilder b = new StringBuilder("Recall@20 all/head/tail/freq-TV (Zipf 1.2):");
        r.forEach((k, v) -> b.append(String.format(" %s %.3f/%.3f/%.3f/%.3f |", k, v[0], v[1], v[2], v[3])));
        String line = b.toString();
        System.out.printf("%s%n", line);
        double[] plain = r.get("in-batch softmax");
        double[] logq = r.get("logQ-corrected in-batch");
        assertTrue(logq[1] > plain[1], line);
        assertTrue(logq[3] < 0.5, line);
        assertTrue(r.get("full softmax (oracle)")[0] > 0.2, line);
    }
}
