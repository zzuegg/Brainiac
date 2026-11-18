package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class GraphAlgorithmsTest {

    @Test
    void maxFlowOfTheTextbookNetwork() {
        // CLRS figure 26.1: s=0, v1..v4 = 1..4, t=5, maximum flow 23
        int[] from = {0, 0, 1, 2, 2, 3, 3, 4, 4};
        int[] to = {1, 2, 3, 1, 4, 2, 5, 3, 5};
        double[] cap = {16, 13, 12, 4, 14, 9, 20, 7, 4};
        Graphs.Flow f = Graphs.maxFlow(6, from, to, cap, 0, 5);
        assertEquals(23, f.value(), 1e-9);
        // conservation at every inner node, and within capacity
        double[] net = new double[6];
        for (int e = 0; e < from.length; e++) {
            assertTrue(f.edgeFlow()[e] >= -1e-9 && f.edgeFlow()[e] <= cap[e] + 1e-9);
            net[from[e]] -= f.edgeFlow()[e];
            net[to[e]] += f.edgeFlow()[e];
        }
        for (int v = 1; v < 5; v++) {
            assertEquals(0, net[v], 1e-9);
        }
        // the cut's capacity equals the flow
        double cut = 0;
        for (int e = 0; e < from.length; e++) {
            if (f.sourceSide()[from[e]] && !f.sourceSide()[to[e]]) {
                cut += cap[e];
            }
        }
        assertEquals(23, cut, 1e-9);
    }

    @Test
    void neighbourhoodGraphs() {
        Tensor points = Tensor.of(Shape.of(4, 2), 0, 0, 1, 0, 0, 1, 5, 5);
        Tensor[] radius = Graphs.radiusGraph(points, 1.1f);
        // 0-1, 0-2 both ways; 1-2 is sqrt 2 apart; 3 is alone
        assertEquals(4, radius[0].shape().dim(0));
        Tensor[] knn = Graphs.knnGraph(points, 1);
        assertArrayEquals(new float[] {0, 1, 2, 3}, knn[1].toArray());
        assertEquals(1f, knn[0].toArray()[1] == 0f ? 1f : 0f);   // 1's nearest is 0
        Tensor[] nn = Linalg.nearestNeighbours(points, points, 2);
        assertArrayEquals(new float[] {0, 1}, java.util.Arrays.copyOf(nn[0].toArray(), 2), 1e-6f);
        Tensor a = Tensor.uniform(Shape.of(2, 5, 3), -1, 1, new SplittableRandom(1));
        Tensor b = Tensor.uniform(Shape.of(2, 4, 3), -1, 1, new SplittableRandom(2));
        float[] d = Linalg.squaredDistances(a, b).toArray();
        float[] av = a.toArray();
        float[] bv = b.toArray();
        for (int t = 0; t < 2; t++) {
            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 4; j++) {
                    double s = 0;
                    for (int c = 0; c < 3; c++) {
                        double diff = av[(t * 5 + i) * 3 + c] - bv[(t * 4 + j) * 3 + c];
                        s += diff * diff;
                    }
                    assertEquals(s, d[(t * 5 + i) * 4 + j], 1e-5);
                }
            }
        }
    }
}
