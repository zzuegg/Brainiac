package io.github.zzuegg.brainiac.experiments.uncertainty;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class EnsembleShootoutSingleRunTest {

    static String line(List<EnsembleShootout.Row> rows) {
        StringBuilder b = new StringBuilder();
        for (EnsembleShootout.Row r : rows) {
            b.append(String.format(" %s acc %.3f nll %.3f near %.3f far %.3f dis %.3f;", r.method(), r.accuracy(),
                    r.nll(), r.nearAuroc(), r.farAuroc(), r.disagreement()));
        }
        return b.toString();
    }

    static EnsembleShootout.Row row(List<EnsembleShootout.Row> rows, String m) {
        return rows.stream().filter(r -> r.method().equals(m)).findFirst().orElseThrow();
    }

    @Test
    void snapshotAndFgeAgainstSingleDeepSwagAndDropout() {
        List<EnsembleShootout.Row> rows = EnsembleShootout.singleRun(450, 3, 1);
        System.out.println("11.4.6 equal steps (450):" + line(rows));
        for (EnsembleShootout.Row r : rows) {
            assertTrue(r.accuracy() > 0.4 && Double.isFinite(r.nll()), r.toString());
        }
        assertTrue(row(rows, "snapshot").disagreement() > 0, "snapshots differ");
    }

    @Test
    void mimoAndPackedAgainstBatchEnsemble() {
        List<EnsembleShootout.Row> rows = EnsembleShootout.insideOneNetwork(300, 1);
        StringBuilder b = new StringBuilder("11.4.7 300 steps:" + line(rows) + " cost:");
        rows.forEach(r -> b.append(String.format(" %s %.0f", r.method(), r.cost())));
        System.out.println(b);
        for (EnsembleShootout.Row r : rows) {
            assertTrue(r.accuracy() > 0.3, r.toString());
        }
        assertTrue(row(rows, "packed").disagreement() > 0, "packed members differ");
    }
}
