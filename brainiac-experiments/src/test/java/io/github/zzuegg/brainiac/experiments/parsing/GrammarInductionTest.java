package io.github.zzuegg.brainiac.experiments.parsing;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class GrammarInductionTest {

    @Test
    void induction() {
        List<GrammarInduction.Row> rows = GrammarInduction.run(300, 60, 120, 61);
        StringBuilder line = new StringBuilder("11.5.10 unlabelled F1 (test NLL/word):");
        for (GrammarInduction.Row r : rows) {
            line.append(String.format(" %s %.3f", r.method(), r.f1()));
            if (!Double.isNaN(r.nll())) {
                line.append(String.format(" (%.2f)", r.nll()));
            }
            line.append(';');
        }
        System.out.println(line);
        double random = rows.get(2).f1();
        double best = rows.stream().filter(r -> r.method().contains("PCFG ") && !r.method().contains("treebank"))
                .mapToDouble(GrammarInduction.Row::f1).max().orElse(0);
        assertTrue(best > random, line.toString());
    }
}
