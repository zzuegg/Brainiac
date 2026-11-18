package io.github.zzuegg.brainiac.experiments.sequence;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class ConvexHullTest {

    @Test
    void hullOfASquareWithAnInnerPoint() {
        float[][] pts = {{0, 0}, {1, 0}, {0.5f, 0.5f}, {1, 1}, {0, 1}};
        assertArrayEquals(new int[] {0, 1, 3, 4}, Seq2SeqShootout.hull(pts));
    }

    @Test
    void pointerNetworkFindsConvexHulls() {
        double[] pointer = Seq2SeqShootout.convexHull(true, 300, 2, 8, 10);
        double[] tokens = Seq2SeqShootout.convexHull(false, 300, 2, 8, 10);
        System.out.printf("convex hull exact match at 8/10 points (trained on 6-8): pointer %.3f/%.3f, Bahdanau "
                + "seq2seq emitting indices %.3f/%.3f%n", pointer[0], pointer[1], tokens[0], tokens[1]);
        assertTrue(pointer[1] > tokens[1] + 0.1, "the pointer network handles unseen sizes");
    }
}
