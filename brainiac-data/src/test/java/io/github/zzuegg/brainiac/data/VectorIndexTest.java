package io.github.zzuegg.brainiac.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class VectorIndexTest {

    @Test
    void flatIsExactForEveryMetric() {
        Tensor v = Tensor.of(Shape.of(3, 2), 1, 0, 0, 2, 3, 3);
        for (VectorIndex.Metric m : VectorIndex.Metric.values()) {
            VectorIndex idx = VectorIndex.flat(2, m, Device.cpu());
            assertEquals(0, idx.add(v.slice(0, 0, 2)));
            assertEquals(2, idx.add(v.slice(0, 2, 3)));
            List<VectorIndex.Hit> hits = idx.search(Tensor.of(new float[] {0.9f, 0.1f}), 2);
            int expected = switch (m) {
                case COSINE, EUCLIDEAN -> 0;
                case DOT -> 2;
            };
            assertEquals(expected, hits.getFirst().id(), m.toString());
            assertTrue(hits.get(0).score() >= hits.get(1).score());
        }
    }

    @Test
    void hnswFindsAlmostAllTrueNeighbours() {
        SplittableRandom r = new SplittableRandom(1);
        int n = 2000;
        int dim = 16;
        Tensor data = Tensor.normal(Shape.of(n, dim), 0, 1, r);
        VectorIndex exact = VectorIndex.flat(dim, VectorIndex.Metric.COSINE, Device.cpu());
        VectorIndex approx = VectorIndex.hnsw(dim, VectorIndex.Metric.COSINE, 16, 100, 50, 2);
        exact.add(data);
        approx.add(data);
        assertEquals(n, approx.size());
        int found = 0;
        int queries = 50;
        for (int q = 0; q < queries; q++) {
            Tensor query = Tensor.normal(Shape.of(dim), 0, 1, r);
            List<Integer> truth = exact.search(query, 10).stream().map(VectorIndex.Hit::id).toList();
            for (VectorIndex.Hit h : approx.search(query, 10)) {
                found += truth.contains(h.id()) ? 1 : 0;
            }
        }
        double recall = found / (queries * 10.0);
        assertTrue(recall > 0.9, "recall@10 " + recall);
    }
}
