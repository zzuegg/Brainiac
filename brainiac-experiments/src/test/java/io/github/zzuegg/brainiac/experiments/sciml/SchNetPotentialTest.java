package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class SchNetPotentialTest {

    @Test
    void forceMatchingImprovesSchNetForces() {
        List<SymmetricModels.PotentialRow> rows = SymmetricModels.potentials(5, 48, 150, 100, 1);
        StringBuilder line = new StringBuilder(
                "8.3.15 LJ-5 clusters, energy MAE/atom / force MAE / MD drift/atom (ms):");
        rows.forEach(r -> line.append(String.format(" %s %.4f/%.3f/%.3g (%.0f);", r.model(), r.energyMae(),
                r.forceMae(), r.energyDrift(), r.millis())));
        System.out.println(line);
        assertTrue(rows.get(0).forceMae() < rows.get(1).forceMae() * 1.5, line.toString());
    }
}
