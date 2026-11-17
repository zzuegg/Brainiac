package io.github.zzuegg.brainiac.experiments.sequence;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.nn.Mamba2Layer;
import io.github.zzuegg.brainiac.experimental.nn.S4dLayer;

import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class MixerShootoutTest {

    @Test
    void tasksProduceConsistentAnswers() {
        MixerShootout.Task recent = MixerShootout.recentRecall(2, 3, 4);
        int[][][] b = recent.batch(new SplittableRandom(1), 50);
        for (int i = 0; i < 50; i++) {
            int[] row = b[0][i];
            int q = row[row.length - 1];
            int last = -1;
            for (int p = 0; p < 4; p++) {
                if (row[2 * p] == q) {
                    last = row[2 * p + 1];
                }
            }
            assertEquals(last, b[1][0][i]);
        }
        int[][][] p = MixerShootout.parity(5).batch(new SplittableRandom(2), 20);
        for (int i = 0; i < 20; i++) {
            int x = 0;
            for (int bit : p[0][i]) {
                x ^= bit;
            }
            assertEquals(x, p[1][0][i]);
        }
    }

    @Test
    void everyMixerOfTheLineupTrainsThroughItsSpec() {
        List<MixerShootout.Result> results = MixerShootout.run(MixerShootout.lineup(),
                List.of(MixerShootout.recall(4, 4, 2)), 16, 2, 0.01f, 1);
        assertEquals(MixerShootout.lineup().size(), results.size());
        for (MixerShootout.Result r : results) {
            assertTrue(r.parameters() > 0, r.mixer());
            assertTrue(r.accuracy() >= 0 && r.accuracy() <= 1, r.mixer());
        }
        String table = MixerShootout.table(results);
        assertTrue(table.contains("Mamba-3 rotary"), table);
    }

    @Test
    void rotaryStateTracksParityWhereTimeInvariantConvolutionCannot() {
        MixerShootout.Task parity = MixerShootout.parity(6);
        List<MixerShootout.Result> r = MixerShootout.run(Map.of(
                "rotary", Mamba2Layer.of(2, 8, 8, 3, Mamba2Layer.Variant.ROTARY),
                "S4D", S4dLayer.of(8)), List.of(parity), 16, 300, 0.01f, 3);
        double rotary = r.stream().filter(x -> x.mixer().equals("rotary")).findFirst().orElseThrow().accuracy();
        double s4d = r.stream().filter(x -> x.mixer().equals("S4D")).findFirst().orElseThrow().accuracy();
        System.out.println(MixerShootout.table(r));
        assertTrue(rotary > 0.9, "rotary " + rotary);
        assertTrue(rotary > s4d + 0.2, "rotary " + rotary + " vs S4D " + s4d);
    }
}
