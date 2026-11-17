package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.experimental.tensor.Hypervector;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * The item (or "cleanup") memory of a vector-symbolic architecture: a fixed random
 * {@link Hypervector} per symbol, created on first use, and nearest-neighbour lookup that
 * turns a noisy result of binding and bundling back into the closest known symbol
 * (Kanerva, "Hyperdimensional Computing", 2009; survey <a href="https://arxiv.org/abs/2111.06077">2111.06077</a>).
 *
 * @param <T> the symbol type
 */
public final class HdcItemMemory<T> {
    private final int dims;
    private final SplittableRandom random;
    private final Map<T, Hypervector> items = new LinkedHashMap<>();

    /**
     * Creates an empty memory.
     *
     * @param dims the vector size, a multiple of 64
     * @param seed the seed for new symbols' vectors
     */
    public HdcItemMemory(int dims, long seed) {
        this.dims = dims;
        this.random = new SplittableRandom(seed);
    }

    /**
     * Returns a symbol's vector, creating a random one the first time.
     *
     * @param symbol the symbol
     * @return its vector
     */
    public Hypervector get(T symbol) {
        return items.computeIfAbsent(symbol, s -> Hypervector.random(dims, random));
    }

    /**
     * Stores a given vector for a symbol (for example a learned prototype).
     *
     * @param symbol the symbol
     * @param vector its vector
     */
    public void put(T symbol, Hypervector vector) {
        items.put(symbol, vector);
    }

    /**
     * Finds the most similar stored symbol.
     *
     * @param query a vector
     * @return the closest symbol, or {@code null} if the memory is empty
     */
    public T cleanup(Hypervector query) {
        T best = null;
        double sim = Double.NEGATIVE_INFINITY;
        for (Map.Entry<T, Hypervector> e : items.entrySet()) {
            double s = e.getValue().similarity(query);
            if (s > sim) {
                sim = s;
                best = e.getKey();
            }
        }
        return best;
    }

    /**
     * Ranks stored symbols by similarity to a query.
     *
     * @param query a vector
     * @param k how many to return
     * @return the {@code k} closest symbols, best first
     */
    public List<T> nearest(Hypervector query, int k) {
        List<T> keys = new ArrayList<>(items.keySet());
        keys.sort((a, b) -> Double.compare(items.get(b).similarity(query), items.get(a).similarity(query)));
        return keys.subList(0, Math.min(k, keys.size()));
    }

    /** @return the number of stored symbols */
    public int size() {
        return items.size();
    }

    /** @return the vector size */
    public int dims() {
        return dims;
    }
}
