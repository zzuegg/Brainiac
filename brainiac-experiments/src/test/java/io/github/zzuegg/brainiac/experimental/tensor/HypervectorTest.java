package io.github.zzuegg.brainiac.experimental.tensor;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class HypervectorTest {
    static final int D = 8192;

    @Test
    void randomVectorsAreNearlyOrthogonal() {
        SplittableRandom r = new SplittableRandom(1);
        Hypervector a = Hypervector.random(D, r);
        Hypervector b = Hypervector.random(D, r);
        assertEquals(0, a.similarity(b), 0.05);
        assertEquals(1, a.similarity(a), 0);
    }

    @Test
    void bindIsSelfInverseAndDissimilar() {
        SplittableRandom r = new SplittableRandom(2);
        Hypervector a = Hypervector.random(D, r);
        Hypervector b = Hypervector.random(D, r);
        Hypervector ab = a.bind(b);
        assertEquals(a, ab.bind(b));
        assertEquals(0, ab.similarity(a), 0.05);
        assertEquals(a, a.bind(Hypervector.zeros(D)));
    }

    @Test
    void bundleStaysSimilarToItsParts() {
        SplittableRandom r = new SplittableRandom(3);
        Hypervector a = Hypervector.random(D, r);
        Hypervector b = Hypervector.random(D, r);
        Hypervector c = Hypervector.random(D, r);
        Hypervector s = Hypervector.bundle(List.of(a, b, c));
        // majority of three: each part agrees on 3/4 of the bits -> similarity 0.5
        assertEquals(0.5, s.similarity(a), 0.05);
        assertEquals(0, s.similarity(Hypervector.random(D, r)), 0.05);
    }

    @Test
    void permuteIsAnInvertibleRotation() {
        Hypervector a = Hypervector.random(D, new SplittableRandom(4));
        Hypervector p = a.permute(5);
        assertEquals(a.get(10), p.get(15));
        assertEquals(a.get(D - 1), p.get(4));
        assertEquals(a, p.permute(-5));
        assertTrue(Math.abs(p.similarity(a)) < 0.05);
    }
}
