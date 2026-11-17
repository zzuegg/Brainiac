package io.github.zzuegg.brainiac.experimental.evolve;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A growing set of behaviour descriptors with k-nearest-neighbour queries: the memory of
 * novelty search ("how unlike everything seen so far is this?") and a basic vector index
 * for curiosity bonuses and episodic memories.
 *
 * <p>Queries are exact (a linear scan with a bounded max-heap), which is fast enough for
 * the tens of thousands of low-dimensional descriptors evolutionary runs collect.
 */
public final class BehaviourArchive {

    private final List<double[]> points = new ArrayList<>();
    private final int dimension;

    /**
     * Creates an empty archive.
     *
     * @param dimension the length of every descriptor
     */
    public BehaviourArchive(int dimension) {
        if (dimension < 1) {
            throw new IllegalArgumentException("dimension must be positive");
        }
        this.dimension = dimension;
    }

    /**
     * Adds a descriptor.
     *
     * @param point the descriptor (copied)
     */
    public void add(double[] point) {
        check(point);
        points.add(point.clone());
    }

    /** @return the number of stored descriptors */
    public int size() {
        return points.size();
    }

    /** @return copies of the stored descriptors */
    public List<double[]> points() {
        return points.stream().map(double[]::clone).toList();
    }

    /**
     * The distances to the {@code k} nearest stored descriptors, nearest first.
     *
     * @param query the descriptor to look up
     * @param k how many neighbours
     * @param extra more descriptors to search besides the archive (such as the current
     *     population), or null
     * @param skipSelf whether to skip one exact match (the query itself, when it is among
     *     {@code extra})
     * @return up to {@code k} Euclidean distances
     */
    public double[] nearestDistances(double[] query, int k, List<double[]> extra, boolean skipSelf) {
        check(query);
        double[] heap = new double[k];   // max-heap of the k smallest squared distances
        int size = 0;
        boolean skipped = !skipSelf;
        List<List<double[]>> sources = extra == null ? List.of(points) : List.of(points, extra);
        for (List<double[]> source : sources) {
            for (double[] p : source) {
                if (!skipped && p == query) {
                    skipped = true;
                    continue;
                }
                double d = 0;
                for (int i = 0; i < dimension; i++) {
                    double t = p[i] - query[i];
                    d += t * t;
                }
                if (size < k) {
                    heap[size] = d;
                    siftUp(heap, size++);
                } else if (d < heap[0]) {
                    heap[0] = d;
                    siftDown(heap, size);
                }
            }
        }
        double[] out = Arrays.copyOf(heap, size);
        Arrays.sort(out);
        for (int i = 0; i < size; i++) {
            out[i] = Math.sqrt(out[i]);
        }
        return out;
    }

    /**
     * Novelty: the mean distance to the {@code k} nearest neighbours.
     *
     * @param query the descriptor
     * @param k how many neighbours
     * @param extra more descriptors to search, or null
     * @return the sparseness of the query's neighbourhood; 0 when there are no neighbours
     */
    public double novelty(double[] query, int k, List<double[]> extra) {
        double[] d = nearestDistances(query, k, extra, true);
        return d.length == 0 ? 0 : Arrays.stream(d).average().orElse(0);
    }

    private void check(double[] p) {
        if (p.length != dimension) {
            throw new IllegalArgumentException("expected a descriptor of length " + dimension + ", got " + p.length);
        }
    }

    private static void siftUp(double[] h, int i) {
        while (i > 0) {
            int parent = (i - 1) / 2;
            if (h[parent] >= h[i]) {
                return;
            }
            double t = h[parent];
            h[parent] = h[i];
            h[i] = t;
            i = parent;
        }
    }

    private static void siftDown(double[] h, int size) {
        int i = 0;
        while (true) {
            int l = 2 * i + 1;
            int r = l + 1;
            int m = i;
            if (l < size && h[l] > h[m]) {
                m = l;
            }
            if (r < size && h[r] > h[m]) {
                m = r;
            }
            if (m == i) {
                return;
            }
            double t = h[m];
            h[m] = h[i];
            h[i] = t;
            i = m;
        }
    }
}
