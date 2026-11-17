package io.github.zzuegg.brainiac.data;

import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Nearest-neighbour search over embedding vectors, for retrieval-augmented generation,
 * deduplication or k-NN classification. {@link #flat} compares a query with every vector
 * (one matrix product on the device: exact); {@link #hnsw} builds a hierarchical navigable
 * small-world graph on the host (Malkov and Yashunin, 2018): approximate, but it visits
 * only a small part of a large collection.
 */
public interface VectorIndex {

    /** How vectors are compared; higher is closer. */
    enum Metric {
        /** The cosine of the angle: vectors are normalized when added and queried. */
        COSINE,
        /** The dot product. */
        DOT,
        /** The negative squared Euclidean distance. */
        EUCLIDEAN
    }

    /**
     * A search result.
     *
     * @param id the vector's id, its position in the order of adding
     * @param score its similarity under the index's metric; higher is closer
     */
    record Hit(int id, float score) {
    }

    /**
     * Adds vectors.
     *
     * @param vectors {@code [n, dim]}
     * @return the id of the first; the others follow in order
     */
    int add(Tensor vectors);

    /**
     * Finds the closest vectors.
     *
     * @param query {@code [dim]}
     * @param k how many
     * @return up to {@code k} hits, closest first
     */
    List<Hit> search(Tensor query, int k);

    /** @return the number of vectors */
    int size();

    /**
     * An exact index that compares every vector.
     *
     * @param dim the vectors' width
     * @param metric the similarity
     * @param device where the vectors are kept and compared
     * @return the index
     */
    static VectorIndex flat(int dim, Metric metric, Device device) {
        return new Flat(dim, metric, device);
    }

    /**
     * An approximate graph index.
     *
     * @param dim the vectors' width
     * @param metric the similarity
     * @param links neighbours per node (M), such as 16
     * @param efConstruction candidates kept while inserting, such as 100
     * @param efSearch candidates kept while searching, at least {@code k}, such as 50
     * @param seed for the levels of new nodes
     * @return the index
     */
    static VectorIndex hnsw(int dim, Metric metric, int links, int efConstruction, int efSearch, long seed) {
        return new Hnsw(dim, metric, links, efConstruction, efSearch, seed);
    }

    /** Normalizes rows for {@link Metric#COSINE}. */
    private static float[] prepared(float[] v, Metric metric) {
        if (metric != Metric.COSINE) {
            return v;
        }
        double n = 0;
        for (float x : v) {
            n += (double) x * x;
        }
        float inv = n == 0 ? 0 : (float) (1 / Math.sqrt(n));
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i] * inv;
        }
        return out;
    }

    private static void checkWidth(Tensor t, int dim, int rank) {
        if (t.shape().rank() != rank || t.shape().dim(-1) != dim) {
            throw new IllegalArgumentException("expected " + (rank == 1 ? "[" : "[n, ") + dim + "], got " + t.shape());
        }
    }

    /** Every vector in one matrix; a query is one product. */
    final class Flat implements VectorIndex {
        private final int dim;
        private final Metric metric;
        private final Device device;
        private final List<Tensor> blocks = new ArrayList<>();
        private Tensor all;
        private Tensor squares;
        private int size;

        Flat(int dim, Metric metric, Device device) {
            this.dim = dim;
            this.metric = metric;
            this.device = device;
        }

        @Override
        public int add(Tensor vectors) {
            checkWidth(vectors, dim, 2);
            int n = vectors.shape().dim(0);
            float[] v = vectors.toArray();
            if (metric == Metric.COSINE) {
                for (int i = 0; i < n; i++) {
                    float[] row = prepared(Arrays.copyOfRange(v, i * dim, (i + 1) * dim), metric);
                    System.arraycopy(row, 0, v, i * dim, dim);
                }
            }
            blocks.add(Tensor.of(Shape.of(n, dim), v, device));
            all = null;
            int first = size;
            size += n;
            return first;
        }

        @Override
        public List<Hit> search(Tensor query, int k) {
            checkWidth(query, dim, 1);
            if (size == 0) {
                return List.of();
            }
            if (all == null) {
                all = blocks.size() == 1 ? blocks.getFirst() : Tensor.concat(0, blocks.toArray(Tensor[]::new));
                blocks.clear();
                blocks.add(all);
                squares = metric == Metric.EUCLIDEAN ? all.square().sum(1, false) : null;
            }
            float[] q = prepared(query.toArray(), metric);
            Tensor qt = Tensor.of(Shape.of(dim, 1), q, device).to(all.dtype());
            Tensor scores = all.matmul(qt).reshape(size);
            if (metric == Metric.EUCLIDEAN) {   // -|a - q|^2 = 2 a.q - |a|^2 - |q|^2
                double qq = 0;
                for (float x : q) {
                    qq += (double) x * x;
                }
                scores = scores.mul(2).sub(squares).sub((float) qq);
            }
            int kk = Math.min(k, size);
            Tensor[] top = scores.reshape(1, size).topk(kk);
            float[] s = top[0].toArray();
            float[] ids = top[1].toArray();
            List<Hit> out = new ArrayList<>(kk);
            for (int i = 0; i < kk; i++) {
                out.add(new Hit((int) ids[i], s[i]));
            }
            return out;
        }

        @Override
        public int size() {
            return size;
        }
    }

    /** A hierarchical navigable small-world graph on the host. */
    final class Hnsw implements VectorIndex {
        private final int dim;
        private final Metric metric;
        private final int links;
        private final int efConstruction;
        private final int efSearch;
        private final SplittableRandom random;
        private final double levelScale;
        private final List<float[]> vectors = new ArrayList<>();
        /** Per node, per level, its neighbours. */
        private final List<List<int[]>> graph = new ArrayList<>();
        private int entry = -1;
        private int top = -1;

        Hnsw(int dim, Metric metric, int links, int efConstruction, int efSearch, long seed) {
            if (links < 2 || efConstruction < 1 || efSearch < 1) {
                throw new IllegalArgumentException("links >= 2 and positive ef needed");
            }
            this.dim = dim;
            this.metric = metric;
            this.links = links;
            this.efConstruction = efConstruction;
            this.efSearch = efSearch;
            this.random = new SplittableRandom(seed);
            this.levelScale = 1 / Math.log(links);
        }

        private float score(float[] a, float[] b) {
            double s = 0;
            if (metric == Metric.EUCLIDEAN) {
                for (int i = 0; i < a.length; i++) {
                    double d = a[i] - b[i];
                    s -= d * d;
                }
            } else {
                for (int i = 0; i < a.length; i++) {
                    s += (double) a[i] * b[i];
                }
            }
            return (float) s;
        }

        @Override
        public int add(Tensor batch) {
            checkWidth(batch, dim, 2);
            int n = batch.shape().dim(0);
            float[] v = batch.toArray();
            int first = vectors.size();
            for (int i = 0; i < n; i++) {
                insert(prepared(Arrays.copyOfRange(v, i * dim, (i + 1) * dim), metric));
            }
            return first;
        }

        private void insert(float[] v) {
            int id = vectors.size();
            vectors.add(v);
            int level = (int) Math.floor(-Math.log(1 - random.nextDouble()) * levelScale);
            List<int[]> mine = new ArrayList<>();
            for (int l = 0; l <= level; l++) {
                mine.add(new int[0]);
            }
            graph.add(mine);
            if (entry < 0) {
                entry = id;
                top = level;
                return;
            }
            int current = entry;
            for (int l = top; l > level; l--) {
                current = greedy(v, current, l);
            }
            for (int l = Math.min(level, top); l >= 0; l--) {
                List<Hit> found = layer(v, List.of(current), efConstruction, l);
                int max = l == 0 ? 2 * links : links;
                int[] chosen = found.stream().limit(max).mapToInt(Hit::id).toArray();
                mine.set(l, chosen);
                for (int c : chosen) {
                    connect(c, id, l, max);
                }
                current = found.getFirst().id();
            }
            if (level > top) {
                top = level;
                entry = id;
            }
        }

        /** Adds an edge, keeping the node's closest {@code max} neighbours. */
        private void connect(int node, int other, int level, int max) {
            int[] old = graph.get(node).get(level);
            int[] grown = Arrays.copyOf(old, old.length + 1);
            grown[old.length] = other;
            if (grown.length > max) {
                float[] nv = vectors.get(node);
                grown = Arrays.stream(grown).boxed()
                        .sorted(Comparator.comparingDouble(x -> -score(nv, vectors.get(x))))
                        .limit(max).mapToInt(Integer::intValue).toArray();
            }
            graph.get(node).set(level, grown);
        }

        private int greedy(float[] q, int start, int level) {
            int current = start;
            float best = score(q, vectors.get(current));
            boolean moved = true;
            while (moved) {
                moved = false;
                for (int n : graph.get(current).get(level)) {
                    float s = score(q, vectors.get(n));
                    if (s > best) {
                        best = s;
                        current = n;
                        moved = true;
                    }
                }
            }
            return current;
        }

        /** Best-first search of one level; returns the closest {@code ef} found, closest first. */
        private List<Hit> layer(float[] q, List<Integer> starts, int ef, int level) {
            Set<Integer> seen = new HashSet<>(starts);
            PriorityQueue<Hit> candidates = new PriorityQueue<>(Comparator.comparingDouble(h -> -h.score()));
            PriorityQueue<Hit> results = new PriorityQueue<>(Comparator.comparingDouble(Hit::score));
            for (int s : starts) {
                Hit h = new Hit(s, score(q, vectors.get(s)));
                candidates.add(h);
                results.add(h);
            }
            while (!candidates.isEmpty()) {
                Hit c = candidates.poll();
                if (results.size() >= ef && c.score() < results.peek().score()) {
                    break;
                }
                List<int[]> levels = graph.get(c.id());
                if (level >= levels.size()) {
                    continue;
                }
                for (int n : levels.get(level)) {
                    if (!seen.add(n)) {
                        continue;
                    }
                    Hit h = new Hit(n, score(q, vectors.get(n)));
                    if (results.size() < ef || h.score() > results.peek().score()) {
                        candidates.add(h);
                        results.add(h);
                        if (results.size() > ef) {
                            results.poll();
                        }
                    }
                }
            }
            List<Hit> out = new ArrayList<>(results);
            out.sort(Comparator.comparingDouble(h -> -h.score()));
            return out;
        }

        @Override
        public List<Hit> search(Tensor query, int k) {
            checkWidth(query, dim, 1);
            if (entry < 0) {
                return List.of();
            }
            float[] q = prepared(query.toArray(), metric);
            int current = entry;
            for (int l = top; l > 0; l--) {
                current = greedy(q, current, l);
            }
            List<Hit> found = layer(q, List.of(current), Math.max(efSearch, k), 0);
            return found.subList(0, Math.min(k, found.size()));
        }

        @Override
        public int size() {
            return vectors.size();
        }
    }
}
