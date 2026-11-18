package io.github.zzuegg.brainiac.experiments.world;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class DreamerTricksTest {

    @Test
    void dreamerV3TricksAcrossRewardScales() {
        List<WorldModelShootout.TrickRow> rows = WorldModelShootout.dreamerTricks(
                new WorldModelShootout.Tricks[] {WorldModelShootout.Tricks.DEFAULT, WorldModelShootout.Tricks.ALL},
                new float[] {0.01f, 1f, 100f}, 400, 1);
        StringBuilder line = new StringBuilder("9.5.6 Dreamer return by reward scale:");
        for (WorldModelShootout.TrickRow r : rows) {
            line.append(String.format(" %s x%s %.1f;", r.config(), r.scale(), r.meanReturn()));
            assertTrue(Double.isFinite(r.meanReturn()), line.toString());
        }
        System.out.println(line);
    }
}
