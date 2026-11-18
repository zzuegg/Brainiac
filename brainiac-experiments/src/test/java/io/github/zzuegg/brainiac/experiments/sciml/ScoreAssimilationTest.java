package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class ScoreAssimilationTest {

    @Test
    void scoreBasedAssimilationReconstructsLorenz63() {
        List<DataAssimilation.FilterRow> rows = DataAssimilation.lorenz63(8, 1000, 20, 8, 1);
        StringBuilder line = new StringBuilder("8.3.16 Lorenz-63 x observed (sd 2), RMSE / spread ratio (ms):");
        rows.forEach(r -> line.append(String.format(" %s %.3f/%.2f (%.0f);", r.method(), r.rmse(), r.spreadRatio(),
                r.millis())));
        System.out.println(line);
        assertTrue(rows.get(0).rmse() < rows.get(rows.size() - 1).rmse(), line.toString());
    }
}
