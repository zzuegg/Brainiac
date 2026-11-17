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

    /**
     * Reads more tokens and returns the prediction after every one of them, as speculative
     * decoding needs to check several guessed tokens at once. By default one step at a time.
     *
     * @param state what was read before; updated
     * @param tokens {@code [batch][steps]} new tokens, without padding
     * @return float32 logits {@code [batch, steps, vocab]}
     */
    default Tensor nextAll(State state, int[][] tokens) {
        int batch = tokens.length;
        int steps = tokens[0].length;
        Tensor[] each = new Tensor[steps];
        for (int t = 0; t < steps; t++) {
            int[][] one = new int[batch][1];
            for (int b = 0; b < batch; b++) {
                one[b][0] = tokens[b][t];
            }
            each[t] = next(state, one, null).reshape(batch, 1, vocabSize());
        }
        return Tensor.concat(1, each);
    }

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

        /**
         * Forgets everything read after the first {@code length} steps, as when speculative
         * decoding rejects guessed tokens.
         *
         * @param length the steps to keep
         * @return the state, this one or a new one
         * @throws UnsupportedOperationException if the model cannot go back
         */
        default State truncate(int length) {
            throw new UnsupportedOperationException("this model's state cannot go back");
        }

        /**
         * Puts another state's sequences after this one's, as continuous batching does when
         * new requests join the ones being decoded. The shorter side is padded on the left,
         * so the result is as long as the longer one; each sequence keeps its positions.
         *
         * @param other a state of the same model; it is no longer used
         * @return the joined state; this one is no longer used
         * @throws UnsupportedOperationException if the model cannot join states
         */
        default State join(State other) {
            throw new UnsupportedOperationException("this model's states cannot be joined");
        }

        /**
         * Frees what this state holds (on a GPU, the key/value cache) now instead of when it
         * is garbage collected. The state is not used afterwards. By default nothing.
         */
        default void close() {
        }
    }
}
