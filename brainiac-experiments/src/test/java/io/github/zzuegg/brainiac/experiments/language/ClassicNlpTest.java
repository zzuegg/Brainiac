package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class ClassicNlpTest {

    private static String join(String label, List<ClassicNlp.Row> rows) {
        StringBuilder b = new StringBuilder(label);
        String cond = "";
        for (ClassicNlp.Row r : rows) {
            if (!r.condition().equals(cond)) {
                cond = r.condition();
                b.append(" | ").append(cond).append(':');
            }
            b.append(String.format(" %s %.3f", r.method(), r.score()));
            if (!Double.isNaN(r.second())) {
                b.append(String.format("/%.3f", r.second()));
            }
            b.append(';');
        }
        return b.toString();
    }

    private static double find(List<ClassicNlp.Row> rows, String cond, String method) {
        return rows.stream().filter(r -> r.condition().equals(cond) && r.method().equals(method)).findFirst()
                .orElseThrow().score();
    }

    @Test
    void wordAlignment() {
        List<ClassicNlp.Row> rows = ClassicNlp.alignment(new int[] {500, 2000}, 200, 5, 150, 51);
        String line = join("11.5.7 AER", rows);
        System.out.println(line);
        assertTrue(find(rows, "2000 pairs", "HMM GROW_DIAG_FINAL") < find(rows, "2000 pairs", "Dice"), line);
    }

    @Test
    void spelling() {
        List<ClassicNlp.Row> rows = ClassicNlp.spelling(0.25, 300, 53);
        String line = join("11.5.8 word acc/typo acc", rows);
        System.out.println(line);
        assertTrue(find(rows, "typos", "noisy channel BRILL_MOORE") > find(rows, "typos", "uncorrected"), line);
    }

    @Test
    void sentenceEmbeddings() {
        List<ClassicNlp.Row> rows = ClassicNlp.sentenceEmbeddings(2000, 250, 55);
        String line = join("11.5.9 Spearman/R@1, align/unif", rows);
        System.out.println(line);
        assertTrue(find(rows, "STS-like", "SimCSE supervised") > 0.3, line);
    }
}
