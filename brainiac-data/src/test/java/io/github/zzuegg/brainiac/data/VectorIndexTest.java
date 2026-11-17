package io.github.zzuegg.brainiac.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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

    @Test
    void removedVectorsLeaveSearches() {
        SplittableRandom r = new SplittableRandom(4);
        Tensor data = Tensor.normal(Shape.of(300, 8), 0, 1, r);
        for (VectorIndex idx : List.of(VectorIndex.flat(8, VectorIndex.Metric.EUCLIDEAN, Device.cpu()),
                VectorIndex.hnsw(8, VectorIndex.Metric.EUCLIDEAN, 8, 50, 30, 2))) {
            idx.add(data);
            Tensor q = data.slice(0, 7, 8).reshape(Shape.of(8));
            assertEquals(7, idx.search(q, 1).getFirst().id());
            assertTrue(idx.remove(7));
            assertFalse(idx.remove(7));
            assertFalse(idx.contains(7));
            assertEquals(299, idx.size());
            List<VectorIndex.Hit> hits = idx.search(q, 5);
            assertEquals(5, hits.size());
            assertTrue(hits.stream().noneMatch(h -> h.id() == 7));
            assertEquals(300, idx.add(data.slice(0, 7, 8)));   // ids are not reused
            assertEquals(300, idx.search(q, 1).getFirst().id());
        }
    }

    @Test
    void storeKeepsValuesWithVectors() {
        VectorStore<String> store = new VectorStore<>(VectorIndex.flat(2, VectorIndex.Metric.COSINE, Device.cpu()));
        store.add(Tensor.of(Shape.of(2, 2), 1, 0, 0, 1), List.of("east", "north"));
        int west = store.add(Tensor.of(new float[] {-1, 0}), "west");
        assertEquals("east", store.search(Tensor.of(new float[] {1, 0.1f}), 1).getFirst().value());
        assertEquals("west", store.get(west));
        assertTrue(store.remove(0));
        assertEquals("north", store.search(Tensor.of(new float[] {1, 0.1f}), 1).getFirst().value());
        assertEquals(2, store.size());
    }
}
