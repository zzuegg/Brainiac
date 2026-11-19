package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class HotCacheTest {
    private final List<String> loaded = new ArrayList<>();
    private final List<String> unloaded = new ArrayList<>();

    private HotCache<String, String> cache(long budget, long warmup, long every, long swap) {
        return new HotCache<>(budget, k -> 10, k -> {
            loaded.add(k);
            return "copy of " + k;
        }, unloaded::add, warmup, every, swap);
    }

    @Test
    void fillsWithTheMostUsedAfterTheWarmup() {
        HotCache<String, String> c = cache(20, 6, 1000, 0);
        assertNull(c.use("a", 3));
        assertNull(c.use("b", 1));
        assertNull(c.use("c", 1));
        // the sixth use fills the budget of two: a (4 uses), then one of b and c
        assertEquals("copy of a", c.use("a", 1));
        assertEquals(2, c.size());
        assertEquals(20, c.bytes());
        assertTrue(c.holds("a"));
        assertEquals(1, c.hitRate(), 1e-9);
        assertNull(c.use("x", 1));
        assertEquals(0.5, c.hitRate(), 1e-9);
    }

    @Test
    void swapsInWhatBecameHotterWithinTheSwapLimit() {
        HotCache<String, String> c = cache(10, 2, 8, 10);
        c.use("a", 2);   // fills with a
        assertTrue(c.holds("a"));
        c.use("b", 8);   // refresh: b (8) beats a (1 * 1.25)
        assertTrue(c.holds("b"));
        assertFalse(c.holds("a"));
        assertEquals(List.of("copy of a"), unloaded);
        assertEquals(List.of("a", "b"), loaded);
    }

    @Test
    void aSmallSwapLimitDoesNotEmptyTheCache() {
        HotCache<String, String> c = cache(30, 3, 30, 10);
        c.use("a", 1);
        c.use("b", 1);
        c.use("c", 1);   // fills with a, b and c
        assertEquals(3, c.size());
        c.use("d", 10);
        c.use("e", 10);
        c.use("f", 10);  // refresh: d, e and f are hotter, but only one may load
        assertEquals(3, c.size());
        assertEquals(30, c.bytes());
        assertTrue(c.holds("d"));
        assertEquals(1, unloaded.size());
    }

    @Test
    void keepsAHeldItemOfAboutTheSameHeat() {
        HotCache<String, String> c = cache(10, 4, 4, 10);
        c.use("a", 4);   // fills with a; a's count halves to 2
        c.use("b", 2);
        c.use("a", 1);
        c.use("b", 1);   // refresh: a 3 * 1.25 against b 3
        assertTrue(c.holds("a"));
        assertTrue(unloaded.isEmpty());
    }

    @Test
    void copiesMadeElsewhereJoinWhenReady() {
        List<Runnable> queued = new ArrayList<>();
        HotCache<String, String> c = new HotCache<>(10, k -> 10, k -> "copy of " + k, unloaded::add, 1, 100, 0,
                queued::add);
        assertNull(c.use("a", 1));   // fills: the copy is on its way
        assertEquals(1, c.loading());
        assertNull(c.use("a", 1));
        queued.forEach(Runnable::run);
        assertEquals("copy of a", c.use("a", 1));
        assertEquals(0, c.loading());
        assertEquals(10, c.bytes());
    }

    @Test
    void clearFreesEverything() {
        HotCache<String, String> c = cache(30, 1, 100, 0);
        c.use("a", 1);
        c.clear();
        assertEquals(0, c.size());
        assertEquals(List.of("copy of a"), unloaded);
        assertThrows(IllegalArgumentException.class, () -> c.use("a", 0));
    }
}
