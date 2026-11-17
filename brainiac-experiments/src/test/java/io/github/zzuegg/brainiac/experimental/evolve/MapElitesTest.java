package io.github.zzuegg.brainiac.experimental.evolve;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class MapElitesTest {

    private static double rastrigin(double[] x) {
        double s = 10 * x.length;
        for (double v : x) {
            s += v * v - 10 * Math.cos(2 * Math.PI * v);
        }
        return s;
    }

    private static double[] random(SplittableRandom r) {
        double[] x = new double[6];
        for (int i = 0; i < x.length; i++) {
            x[i] = r.nextDouble(-0.5, 0.5);
        }
        return x;
    }

    private static MapElites<double[]> map(boolean isoLine) {
        // descriptor: the first two coordinates; quality: rastrigin of the rest
        MapElites<double[]> m = MapElites.of(MapElitesTest::random, MapElites.gaussian(0.15),
                x -> new MapElites.Evaluation(-rastrigin(java.util.Arrays.copyOfRange(x, 2, 6)),
                        new double[] {x[0], x[1]}))
                .grid(new int[] {10, 10}, new double[] {-2, -2}, new double[] {2, 2}).seed(4);
        if (isoLine) {
            m.crossover(MapElites.isoLine(0.05, 0.3));
        }
        return m;
    }

    @Test
    void illuminatesMoreOfTheSpaceThanRandomSampling() {
        MapElites<double[]> me = map(true).run(100, 6_000);
        MapElites<double[]> rnd = map(false);
        SplittableRandom r = new SplittableRandom(4);
        for (int i = 0; i < 6_100; i++) {
            rnd.insert(random(r));
        }
        assertTrue(me.coverage() > 0.9, "coverage " + me.coverage());
        assertTrue(me.coverage() > 2 * rnd.coverage(), me.coverage() + " vs random " + rnd.coverage());
        assertTrue(me.qdScore(-200) > rnd.qdScore(-200));
        assertEquals(6_100, me.evaluations());
    }

    @Test
    void anEliteIsOnlyReplacedByABetterOne() {
        MapElites<double[]> m = MapElites.<double[]>of(r -> new double[] {0}, (x, r) -> x,
                x -> new MapElites.Evaluation(x[1], new double[] {x[0]}))
                .grid(new int[] {2}, new double[] {0}, new double[] {1});
        m.run(0, 0);
        assertTrue(m.insert(new double[] {0.1, 1}));
        assertTrue(!m.insert(new double[] {0.2, 0.5}));
        assertTrue(m.insert(new double[] {0.3, 2}));
        assertTrue(!m.insert(new double[] {5, 9}));   // outside the grid
        List<MapElites.Elite<double[]>> e = m.elites();
        assertEquals(1, e.size());
        assertArrayEquals(new double[] {0.3, 2}, e.getFirst().genome());
    }

    @Test
    void archiveFindsNearestNeighbours() {
        BehaviourArchive a = new BehaviourArchive(1);
        for (int i = 0; i < 10; i++) {
            a.add(new double[] {i});
        }
        assertArrayEquals(new double[] {0.5, 0.5, 1.5}, a.nearestDistances(new double[] {4.5}, 3, null, false), 1e-12);
        assertEquals(0.5, a.novelty(new double[] {4.5}, 2, null), 1e-12);
    }
}
