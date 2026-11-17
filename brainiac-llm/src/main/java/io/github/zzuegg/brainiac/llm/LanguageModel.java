package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * A model that predicts the next token: what {@link Generator} needs. It keeps what it has
 * read so far in a {@link State} (a key/value cache), so each new token costs one step.
 */
public interface LanguageModel {

    /** @return the vocabulary size */
    int vocabSize();

    /**
     * Starts reading new sequences.
     *
     * @param batch the number of sequences read side by side
     * @return an empty state
     */
    State newState(int batch);

    /**
     * Reads more tokens and predicts the next one.
     *
     * @param state what was read before; updated
     * @param tokens {@code [batch][steps]} new tokens, the same count per sequence
     * @param padding {@code [batch][steps]}, {@code true} where a token is padding that
     *     nothing attends to (prompts of different lengths are padded on the left), or
     *     {@code null} for none
     * @return float32 logits {@code [batch, vocab]} for the token after the last step
     */
    Tensor next(State state, int[][] tokens, boolean[][] padding);

    /** What a model has read so far. */
    interface State {

        /** @return the steps read so far, padding included */
        int length();

        /**
         * Keeps some sequences, in a new order, for beam search: row {@code i} of the result
         * is row {@code rows[i]} of this state. A row may appear more than once.
         *
         * @param rows the rows to keep
         * @return the new state; this one is no longer used
         */
        State select(int[] rows);
    }
}
