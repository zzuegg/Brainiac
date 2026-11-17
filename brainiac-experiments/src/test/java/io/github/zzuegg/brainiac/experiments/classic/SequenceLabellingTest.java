package io.github.zzuegg.brainiac.experiments.classic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class SequenceLabellingTest {

    private static String line(Map<String, double[]> m, String a, String b) {
        StringBuilder sb = new StringBuilder();
        m.forEach((k, v) -> sb.append(String.format("%s %s %.3f %s %.3f; ", k, a, v[0], b, v[1])));
        return sb.toString();
    }

    @Test
    void hmmDecodesTheDishonestCasino() {
        Map<String, double[]> r = SequenceLabelling.casino(1);
        System.out.printf("casino: %s%n", line(r, "acc", "ll/step"));
        assertTrue(r.get("HMM true params")[0] > 0.75, line(r, "acc", "ll"));
        assertTrue(r.get("HMM Baum-Welch")[0] > 0.7, line(r, "acc", "ll"));
        assertTrue(r.get("HMM Baum-Welch")[1] > r.get("i.i.d. baseline")[1], line(r, "acc", "ll"));
        assertTrue(r.get("HMM true params")[0] > r.get("GRU tagger (supervised)")[0] - 0.03, line(r, "acc", "ll"));
    }

    @Test
    void gaussianHmmDecodesRegimes() {
        Map<String, double[]> r = SequenceLabelling.regimes(2);
        System.out.printf("regimes: %s%n", line(r, "acc", "ll/step"));
        assertTrue(r.get("HMM Baum-Welch")[0] > 0.85, line(r, "acc", "ll"));
        assertTrue(r.get("HMM Baum-Welch")[1] > r.get("i.i.d. baseline")[1] + 0.05, line(r, "acc", "ll"));
    }

    @Test
    void crfHeadNeverEmitsIllegalTransitions() {
        Map<String, double[]> r = SequenceLabelling.chunking(150, 3);
        System.out.printf("BIO chunking: %s%n", line(r, "span F1", "illegal rate"));
        assertTrue(r.get("GRU-CRF")[0] > 0.8, line(r, "f1", "illegal"));
        assertTrue(r.get("GRU-CRF")[1] < 0.002, line(r, "f1", "illegal"));
        assertTrue(r.get("GRU-softmax")[1] > 0.01, line(r, "f1", "illegal"));
        assertTrue(r.get("GRU-CRF")[0] > r.get("GRU-softmax")[0] + 0.1, line(r, "f1", "illegal"));
    }
}
