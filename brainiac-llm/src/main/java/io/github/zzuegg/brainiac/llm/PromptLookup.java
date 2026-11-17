package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.Arrays;

/**
 * A draft for {@linkplain Generator#speculative speculative decoding} that costs nothing to
 * run (prompt lookup decoding, Saxena 2023): it finds the latest earlier place where the last
 * few tokens appeared, in the prompt or in the text written so far, and guesses the token that
 * followed there. Text that copies from its prompt (summaries, code edits, answers that quote
 * a document, extraction) is guessed many tokens at a time; elsewhere the guess is uniform and
 * almost always rejected, which costs about one ordinary step of the model.
 *
 * <p>Its guesses are certain, so the result still follows the model exactly: greedy decoding
 * gives the same tokens as without a draft.
 */
public final class PromptLookup implements LanguageModel {

    private static final float UNLIKELY = -1e4f;

    private final int vocab;
    private final int longest;
    private final int shortest;

    /**
     * A draft matching up to 3 tokens and at least 1.
     *
     * @param vocab the vocabulary size, the model's
     */
    public PromptLookup(int vocab) {
        this(vocab, 3, 1);
    }

    /**
     * @param vocab the vocabulary size, the model's
     * @param longest the most tokens to match; longer matches are tried first
     * @param shortest the fewest tokens to match
     */
    public PromptLookup(int vocab, int longest, int shortest) {
        if (vocab < 1) {
            throw new IllegalArgumentException("vocab must be positive, got " + vocab);
        }
        if (shortest < 1 || longest < shortest) {
            throw new IllegalArgumentException("need 1 <= shortest <= longest, got " + shortest + " and " + longest);
        }
        this.vocab = vocab;
        this.longest = longest;
        this.shortest = shortest;
    }

    @Override
    public int vocabSize() {
        return vocab;
    }

    @Override
    public State newState(int batch) {
        if (batch < 1) {
            throw new IllegalArgumentException("batch must be positive, got " + batch);
        }
        int[][] rows = new int[batch][16];
        return new Text(rows, new int[batch]);
    }

    @Override
    public Tensor next(State state, int[][] tokens, boolean[][] padding) {
        Text text = (Text) state;
        int batch = text.rows.length;
        if (tokens.length != batch) {
            throw new IllegalArgumentException("expected " + batch + " sequences, got " + tokens.length);
        }
        float[] logits = new float[batch * vocab];
        for (int b = 0; b < batch; b++) {
            for (int t = 0; t < tokens[b].length; t++) {
                if (padding == null || !padding[b][t]) {
                    text.append(b, tokens[b][t]);
                }
            }
            int guess = guess(text.rows[b], text.sizes[b]);
            if (guess >= 0) {
                Arrays.fill(logits, b * vocab, (b + 1) * vocab, UNLIKELY);
                logits[b * vocab + guess] = 0;
            }
        }
        text.steps += tokens[0].length;
        return Tensor.of(Shape.of(batch, vocab), logits);
    }

    /** The token after the latest earlier copy of the text's end, or -1 if there is none. */
    int guess(int[] text, int size) {
        for (int n = Math.min(longest, size - 1); n >= shortest; n--) {
            int tail = size - n;
            for (int start = tail - 1; start >= 0; start--) {
                if (Arrays.equals(text, start, start + n, text, tail, size)) {
                    int token = text[start + n];
                    return token < vocab ? token : -1;
                }
            }
        }
        return -1;
    }

    /** The tokens read so far, per sequence. */
    private static final class Text implements State {

        int[][] rows;
        int[] sizes;
        /** Steps read, padding included, as {@link State#length} counts them. */
        int steps;

        Text(int[][] rows, int[] sizes) {
            this.rows = rows;
            this.sizes = sizes;
        }

        void append(int b, int token) {
            if (sizes[b] == rows[b].length) {
                rows[b] = Arrays.copyOf(rows[b], rows[b].length * 2);
            }
            rows[b][sizes[b]++] = token;
        }

        @Override
        public int length() {
            return steps;
        }

        @Override
        public State select(int[] keep) {
            int[][] r = new int[keep.length][];
            int[] s = new int[keep.length];
            for (int i = 0; i < keep.length; i++) {
                r[i] = rows[keep[i]].clone();
                s[i] = sizes[keep[i]];
            }
            Text t = new Text(r, s);
            t.steps = steps;
            return t;
        }

        @Override
        public State truncate(int length) {
            if (length < 0 || length > steps) {
                throw new IllegalArgumentException("cannot keep " + length + " of " + steps + " steps");
            }
            // without padding every step is one token per sequence; padded rows were
            // shorter, by the same amount at every step after their padding
            int drop = steps - length;
            for (int b = 0; b < sizes.length; b++) {
                sizes[b] = Math.max(0, sizes[b] - drop);
            }
            steps = length;
            return this;
        }
    }
}
