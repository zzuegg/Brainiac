package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Where a decoder keeps the keys and values of many sequences, addressed by sequence and
 * slot: the extension point a pluggable KV cache needs. {@link PagedLlama} runs a core
 * {@code Llama}'s weights over any store; {@link PagedKvCache} is the paged block pool (vLLM's
 * PagedAttention) and {@link OffloadingKvCache} keeps only hot blocks resident and pages the
 * rest to a host tier.
 *
 * <p>A sequence's slots are numbered from 0 in the order they were appended. The decoder
 * {@linkplain #append appends} room for its new tokens, {@linkplain #write writes} each
 * layer's keys and values there, and asks for a {@linkplain #view view} of the slots its
 * queries may attend to. The core {@code Llama} keeps its cache private, so this interface is
 * the hook it lacks.
 */
public interface KvBlockStore {

    /**
     * Keys and values for one layer of one sequence, and which slots they came from.
     *
     * @param keys {@code [slots, kvHeads, headDim]}
     * @param values {@code [slots, kvHeads, headDim]}
     * @param slots the slot of each row, ascending
     */
    record View(Tensor keys, Tensor values, int[] slots) {
    }

    /** @return the number of layers the store holds keys for */
    int layers();

    /** @return a new, empty sequence id */
    int newSequence();

    /**
     * A new sequence holding the same slots as another, as beam search and prefix caching need.
     * A paged store shares the blocks and copies one only when either sequence writes into it.
     *
     * @param sequence the sequence to copy
     * @return the new sequence id
     */
    int fork(int sequence);

    /**
     * Forgets a sequence and gives its memory back.
     *
     * @param sequence the sequence
     */
    void free(int sequence);

    /**
     * @param sequence the sequence
     * @return its slots so far
     */
    int length(int sequence);

    /**
     * Forgets every slot from {@code length} on.
     *
     * @param sequence the sequence
     * @param length the slots to keep
     */
    void truncate(int sequence, int length);

    /**
     * @param sequence the sequence
     * @param steps new slots wanted
     * @return whether {@link #append} can make room for them now
     */
    boolean canAppend(int sequence, int steps);

    /**
     * Makes room for new slots at the end of a sequence.
     *
     * @param sequence the sequence
     * @param steps the new slots
     * @return the first new slot
     * @throws IllegalStateException if the store is full
     */
    int append(int sequence, int steps);

    /**
     * Writes one layer's keys and values into slots made by {@link #append}.
     *
     * @param sequence the sequence
     * @param layer the layer
     * @param from the first slot
     * @param keys {@code [steps, kvHeads, headDim]}
     * @param values {@code [steps, kvHeads, headDim]}
     */
    void write(int sequence, int layer, int from, Tensor keys, Tensor values);

    /**
     * The slots one layer's queries attend to. A full store returns every slot; an offloading
     * store may return a subset chosen from the queries.
     *
     * @param sequence the sequence
     * @param layer the layer
     * @param queries {@code [heads, steps, headDim]}, already rotated
     * @return the keys and values
     */
    View view(int sequence, int layer, Tensor queries);

    /** @return the bytes of key/value data held resident now (on the device) */
    long residentBytes();

    /** @return the most bytes ever held resident */
    long peakResidentBytes();
}
