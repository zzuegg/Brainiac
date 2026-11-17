package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.GridWorld;
import io.github.zzuegg.brainiac.rl.Rl;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EpisodicControlTest {

    @Test
    void keepsTheBestReturnPerStateAction() {
        GridWorld env = GridWorld.of("S.G").coordinates(true);
        EpisodicControl ec = EpisodicControl.of(env, 0, 1).discount(1);
        float[] start = {0, 0};
        ec.remember(List.of(start), List.of(1), List.of(0.5f));
        ec.remember(List.of(start), List.of(1), List.of(0.2f));
        assertEquals(0.5, ec.actionValues(start)[1], 1e-6);
        assertEquals(1, ec.memories());
        assertEquals(1, Math.round(ec.act(start, false)[0]));
    }

    @Test
    void memorySurvivesASession(@TempDir Path dir) throws Exception {
        GridWorld env = GridWorld.of("S..", "..G").coordinates(true);
        EpisodicControl ec = EpisodicControl.of(env, 0, 1);
        ec.fit(env, 400);
        double before = Rl.evaluate(ec, env, 3, 2);
        ec.save(dir.resolve("ec.bin"));
        EpisodicControl fresh = EpisodicControl.of(env, 0, 1);
        fresh.load(dir.resolve("ec.bin"));
        assertEquals(ec.memories(), fresh.memories());
        assertEquals(before, Rl.evaluate(fresh, env, 3, 2), 1e-6);
        assertTrue(before > 0.9, "learned return " + before);
    }

    @Test
    void randomProjectionEmbedsLargeObservations() {
        GridWorld env = GridWorld.of("S...#....", ".........", "........G");
        EpisodicControl ec = EpisodicControl.of(env, 8, 3).generalization(0.01, 0);
        ec.fit(env, 3000);
        assertTrue(Rl.evaluate(ec, env, 3, 4) > 0.8);
    }
}
