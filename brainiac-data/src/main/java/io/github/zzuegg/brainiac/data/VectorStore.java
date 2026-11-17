package io.github.zzuegg.brainiac.data;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A {@link VectorIndex} that keeps a value with each vector, such as the text of a retrieved
 * passage or a memory's contents, for retrieval-augmented generation and k-NN memories.
 *
 * @param <T> the values
 */
public final class VectorStore<T> {

    /**
     * A search result.
     *
     * @param id the vector's id
     * @param score its similarity; higher is closer
     * @param value the value stored with it
     * @param <T> the values
     */
    public record Match<T>(int id, float score, T value) {
    }

    private final VectorIndex index;
    private final Map<Integer, T> values = new HashMap<>();

    /** @param index where the vectors go; it should be empty */
    public VectorStore(VectorIndex index) {
        this.index = Objects.requireNonNull(index, "index");
    }

    /**
     * @param vector {@code [dim]}
     * @param value what to store with it
     * @return its id
     */
    public int add(Tensor vector, T value) {
        return add(vector.reshape(Shape.of(1, vector.shape().dim(-1))), List.of(value));
    }

    /**
     * @param vectors {@code [n, dim]}
     * @param items one value per row
     * @return the id of the first; the others follow in order
     */
    public int add(Tensor vectors, List<? extends T> items) {
        if (vectors.shape().rank() != 2 || vectors.shape().dim(0) != items.size()) {
            throw new IllegalArgumentException(items.size() + " values for vectors " + vectors.shape());
        }
        int first = index.add(vectors);
        for (int i = 0; i < items.size(); i++) {
            values.put(first + i, items.get(i));
        }
        return first;
    }

    /**
     * @param query {@code [dim]}
     * @param k how many
     * @return up to {@code k} matches, closest first
     */
    public List<Match<T>> search(Tensor query, int k) {
        List<Match<T>> out = new ArrayList<>(k);
        for (VectorIndex.Hit h : index.search(query, k)) {
            out.add(new Match<>(h.id(), h.score(), values.get(h.id())));
        }
        return out;
    }

    /**
     * @param id a vector's id
     * @return its value, or {@code null} when it is not stored
     */
    public T get(int id) {
        return values.get(id);
    }

    /**
     * @param id a vector's id
     * @return whether it was there to remove
     */
    public boolean remove(int id) {
        values.remove(id);
        return index.remove(id);
    }

    /** @return the number of vectors stored */
    public int size() {
        return index.size();
    }

    /** @return the index underneath */
    public VectorIndex index() {
        return index;
    }
}
