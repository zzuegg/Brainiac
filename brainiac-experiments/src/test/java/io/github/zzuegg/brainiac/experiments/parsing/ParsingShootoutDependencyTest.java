package io.github.zzuegg.brainiac.experiments.parsing;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class ParsingShootoutDependencyTest {

    static final ParsingShootout.Budget BUDGET = new ParsingShootout.Budget(300, 80, 12, 18, 250, 8);

    static String line(String title, Map<String, double[]> r) {
        StringBuilder s = new StringBuilder(title).append(':');
        r.forEach((k, v) -> {
            s.append(' ').append(k);
            for (double x : v) {
                s.append(String.format(" %.3f", x));
            }
            s.append(';');
        });
        return s.toString();
    }

    @Test
    void biaffineAlwaysTreesAndBeatsRightBranching() {
        Map<String, double[]> r = ParsingShootout.dependency(BUDGET, 1);
        String line = line("parsing dependency UAS, LAS, trees, PP, UAS long, ms", r);
        System.out.println(line);
        assertTrue(r.get("biaffine chu_liu_edmonds")[2] == 1.0, line);
        assertTrue(r.get("biaffine eisner")[2] == 1.0, line);
        assertTrue(r.get("biaffine chu_liu_edmonds")[0] > r.get("right-branching")[0], line);
    }
}
