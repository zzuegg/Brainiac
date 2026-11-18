package io.github.zzuegg.brainiac.experiments.compression;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class NetworkSlimmingTest {

    @Test
    void slimmingAgainstFilterPruningAndThinNetworks() {
        List<CompactRow> rows = PruningShootout.cnn(new double[] {0.5, 0.7}, 40, 20, 3e-3f, 1);
        StringBuilder line = new StringBuilder("8.5.6 network slimming, shapes CNN acc @MACs (latency ms):");
        for (CompactRow r : rows) {
            line.append(String.format(" %s %.3f @%.0f (%.2f);", r.method(), r.metric(), r.cost(), r.millis()));
        }
        System.out.println(line);
        assertTrue(rows.get(0).metric() > 0.6, "dense CNN learns the shapes: " + rows.get(0));
        for (int i = 2; i < rows.size(); i += 4) {
            CompactRow slim = rows.get(i);
            assertTrue(slim.cost() < rows.get(0).cost(), "slimming removes multiply-adds: " + slim);
            assertTrue(slim.metric() > 0.3, "slimmed CNN still classifies: " + slim);
        }
    }
}
