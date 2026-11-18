package io.github.zzuegg.brainiac.experiments.biology;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class SequenceAnalysisTest {

    static String line(Map<String, double[]> r) {
        StringBuilder b = new StringBuilder();
        r.forEach((k, v) -> {
            b.append(k).append(" {");
            for (int i = 0; i < v.length; i++) {
                b.append(String.format(i == 0 ? "%.3f" : ", %.3f", v[i]));
            }
            b.append("} | ");
        });
        return b.toString();
    }

    @Test
    void learnedScoresImproveAlignmentAtHighDivergence() {
        Map<String, double[]> a = SequenceAnalysis.alignment(new double[] {0.2, 0.4, 0.6}, 30, 12, 50, 1);
        System.out.printf("12.4.1 alignment accuracy at divergence {0.2, 0.4, 0.6}: %s%n", line(a));
        Map<String, double[]> h = SequenceAnalysis.homology(new double[] {0.5, 0.7, 0.8}, 6, 40, 50, 2);
        System.out.printf("12.4.1 homology search AUROC at divergence {0.5, 0.7, 0.8}: %s%n", line(h));
        assertTrue(a.get("SW learned matrix (soft-SW CRF)")[2] > a.get("edit distance")[2], line(a));
        assertTrue(a.get("learned matrix group recovery")[0] > 0, line(a));
        assertTrue(h.get("SW learned matrix")[1] > 0.8, line(h));
    }

    @Test
    void profileHmmsDetectFamilies() {
        Map<String, double[]> f = SequenceAnalysis.families(4, 6, 8, 0.2, 3);
        System.out.printf("12.4.2 family detection AUROC (identity 0.2, 6 members): %s%n", line(f));
        assertTrue(f.get("profile HMM (forward log-odds)")[0] > 0.8, line(f));
        assertTrue(f.get("profile HMM (forward log-odds)")[0] >= f.get("flat HMM (5 states)")[0], line(f));
    }

    @Test
    void motifFindersRecoverThePlantedPwm() {
        Map<String, double[]> m = SequenceAnalysis.motifs(200, 80, 10, 0.8, 4);
        System.out.printf("12.4.3 motifs {PWM corr, site F1, AUROC}: %s%n", line(m));
        assertTrue(m.get("MEME_EM")[0] > 0.6, line(m));
        assertTrue(m.get("MEME_EM")[2] > 0.6, line(m));
    }
}
