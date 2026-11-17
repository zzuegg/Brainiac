package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LeniaTest {

    @Test
    void kernelIsNormalizedAndGrowthPeaksAtMu() {
        Lenia l = Lenia.orbium(32);
        double sum = 0;
        for (double k : l.kernel()) {
            sum += k;
        }
        assertEquals(1, sum, 1e-9);
        assertEquals(1, l.growth(0.15), 1e-12);
        assertEquals(-1, l.growth(0.9), 1e-6);
    }

    @Test
    void uniformWorldsFollowTheGrowthFunction() {
        Lenia l = Lenia.orbium(32);
        double[] s = new double[32 * 32];
        java.util.Arrays.fill(s, 0.15);   // the measure equals the state, right at mu: grow by dt
        l.state(s);
        l.step();
        assertEquals(0.25, l.state()[77], 1e-9);
        l.state(new double[32 * 32]);
        l.step();
        assertEquals(0, l.mass(), 0);   // an empty world stays empty
    }

    @Test
    void stepsAreTranslationEquivariantOnTheTorus() {
        SplittableRandom r = new SplittableRandom(1);
        Lenia a = Lenia.of(24, 5, 0.2, 0.03, 0.1);
        a.soup(10, r);
        double[] start = a.state();
        double[] shifted = new double[start.length];
        for (int y = 0; y < 24; y++) {
            for (int x = 0; x < 24; x++) {
                shifted[((y + 7) % 24) * 24 + (x + 3) % 24] = start[y * 24 + x];
            }
        }
        Lenia b = Lenia.of(24, 5, 0.2, 0.03, 0.1);
        b.state(shifted);
        a.run(10, null, 1);
        b.run(10, null, 1);
        double[] sa = a.state();
        double[] sb = b.state();
        for (int y = 0; y < 24; y++) {
            for (int x = 0; x < 24; x++) {
                assertEquals(sa[y * 24 + x], sb[((y + 7) % 24) * 24 + (x + 3) % 24], 1e-9);
            }
        }
    }

    @Test
    void searchFindsSettingsWhereLifePersists() {
        List<double[]> found = Lenia.search(32, 6, 16, 120, new SplittableRandom(3));
        assertFalse(found.isEmpty(), "some random growth settings keep a soup alive");
        Lenia l = Lenia.of(32, 6, found.get(0)[0], found.get(0)[1], 0.1);
        l.soup(16, new SplittableRandom(4));
        List<float[]> frames = new ArrayList<>();
        l.run(30, frames, 3);
        assertEquals(10, frames.size());
        assertTrue(l.gif(frames, 2).length > 500);
    }
}
