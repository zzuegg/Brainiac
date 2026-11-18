package io.github.zzuegg.brainiac.experiments.classic;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class TsetlinMachineTest {

    private static String line(String head, List<ClassicLearners.TsetlinRow> rows) {
        StringBuilder b = new StringBuilder(head);
        rows.forEach(r -> b.append(String.format(" %s%s %.3f (%.0f ms%s);", r.method(),
                r.clauses() > 0 ? "/" + r.clauses() : "", r.accuracy(), r.millis(),
                r.rulesRecovered() >= 0 ? ", rules " + r.rulesRecovered() + "/4" : "")));
        return b.toString();
    }

    private static ClassicLearners.TsetlinRow row(List<ClassicLearners.TsetlinRow> rows, String m, int clauses) {
        return rows.stream().filter(r -> r.method().equals(m) && r.clauses() == clauses).findFirst().orElseThrow();
    }

    @Test
    void recoversNoisyXorRules() {
        List<ClassicLearners.TsetlinRow> rows = ClassicLearners.tsetlinXor(1);
        System.out.println(line("11.4.13 noisy XOR acc:", rows));
        ClassicLearners.TsetlinRow tm = rows.getFirst();
        assertTrue(tm.accuracy() > 0.95, "the machine solves XOR");
        assertTrue(tm.rulesRecovered() >= 3, "most planted conjunctions appear exactly as clauses");
        assertTrue(tm.accuracy() > row(rows, "logistic", 0).accuracy() + 0.3, "logistic regression cannot do XOR");
    }

    @Test
    void digitsAndTabularAgainstBaselines() {
        List<ClassicLearners.TsetlinRow> digits = ClassicLearners.tsetlinDigits(10, 1);
        System.out.println(line("11.4.13 digits acc:", digits));
        List<ClassicLearners.TsetlinRow> tab = ClassicLearners.tsetlinTabular(10, 1);
        System.out.println(line("11.4.13 tabular acc:", tab));
        assertTrue(row(digits, "tsetlin", 100).accuracy() > 0.5, "the standard machine learns digits");
        assertTrue(row(digits, "fuzzy", 10).accuracy() > 0.5, "the fuzzy machine learns digits with 10 clauses");
        assertTrue(row(tab, "tsetlin", 40).accuracy() > 0.6, "the thermometer-encoded machine learns the table");
    }
}
