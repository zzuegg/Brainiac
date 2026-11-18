package io.github.zzuegg.brainiac.experiments.compression;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class PruningCriteriaTest {

    static String join(String title, List<PruningShootout.Row> rows, boolean extra) {
        StringBuilder line = new StringBuilder(title);
        for (PruningShootout.Row r : rows) {
            line.append(extra ? String.format(" %s@%.3f %.3f (%.3f);", r.method(), r.sparsity(), r.accuracy(),
                    r.extra()) : String.format(" %s@%.3f %.3f;", r.method(), r.sparsity(), r.accuracy()));
        }
        return line.toString();
    }

    @Test
    void movementPruningDuringTransfer() {
        List<PruningShootout.Row> rows = PruningShootout.transfer(new double[] {0.5, 0.95}, 100, 4, 1);
        System.out.println(join("8.5.2 movement vs magnitude/Wanda (transfer), test acc:", rows, false));
        PruningShootout.Row movement95 = rows.get(5);
        assertEquals("movement", movement95.method());
        assertTrue(movement95.sparsity() > 0.9, "hard movement reaches its sparsity: " + movement95);
        assertTrue(movement95.accuracy() > 0.3, "the 95% sparse movement network classifies: " + movement95);
        PruningShootout.Row soft = rows.get(6);
        assertTrue(soft.sparsity() > 0.1 && soft.accuracy() > 0.3, "soft movement prunes and works: " + soft);
    }

    @Test
    void woodFisherAgainstOneShotCriteria() {
        List<PruningShootout.Row> rows = PruningShootout.oneShot(new double[] {0.7, 0.9}, 128, 30, 1);
        System.out.println(join("8.5.3 one-shot pruning, acc after recovery (right after pruning):", rows, true));
        for (int i = 0; i < rows.size(); i += 6) {
            PruningShootout.Row magnitude = rows.get(i);
            PruningShootout.Row woodFisher = rows.get(i + 5);
            assertTrue(woodFisher.extra() > magnitude.extra(),
                    "multi-step WoodFisher is at least as good as magnitude one-shot: " + woodFisher + " vs " + magnitude);
            assertTrue(woodFisher.sparsity() > magnitude.sparsity() - 0.02);
        }
    }
}
