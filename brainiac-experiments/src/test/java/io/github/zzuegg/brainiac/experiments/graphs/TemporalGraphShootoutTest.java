package io.github.zzuegg.brainiac.experiments.graphs;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.TemporalGraphs;
import org.junit.jupiter.api.Test;

import java.util.Map;

class TemporalGraphShootoutTest {

    static final TemporalGraphShootout.Budget BUDGET = new TemporalGraphShootout.Budget(40, 1500, 10, 12, 100);

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
    void edgeBankStrongUnderRandomWeakUnderHistoricalNegatives() {
        for (TemporalGraphs.Kind kind : new TemporalGraphs.Kind[] {TemporalGraphs.Kind.REPEATED,
            TemporalGraphs.Kind.RECENCY}) {
            Map<String, double[]> r = TemporalGraphShootout.compare(kind, BUDGET, 1);
            String line = line("temporal graph " + kind + " MRR/AUC random, MRR/AUC historical, ms", r);
            System.out.println(line);
            double[] bank = r.get("EdgeBank");
            assertTrue(bank[0] > bank[2], line);
            for (double[] v : r.values()) {
                assertTrue(v[0] > 0 && v[0] <= 1 && v[2] > 0 && v[2] <= 1, line);
            }
        }
    }
}
