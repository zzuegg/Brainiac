package io.github.zzuegg.brainiac.experimental.data;

import java.util.random.RandomGenerator;

/**
 * The fill-in-the-middle document transform (Bavarian et al., "Efficient Training of
 * Language Models to Fill in the Middle", <a href="https://arxiv.org/abs/2207.14255">2207.14255</a>),
 * the data-side companion of {@link SequencePacker}. A document is cut at two random points
 * into prefix, middle and suffix and rearranged with three sentinel tokens so that a plain
 * causal language model, trained with plain next-token loss, learns to write the middle
 * after seeing both ends:
 *
 * <pre>
 * PSM:  [PRE] prefix [SUF] suffix [MID] middle
 * SPM:  [PRE] [SUF] suffix [MID] prefix middle
 * </pre>
 *
 * <p>The paper's finding is that transforming about half of the documents this way costs
 * almost nothing in ordinary left-to-right quality ("FIM is free"). At inference,
 * {@link #infillPrompt} builds the prompt and the model's continuation is the middle.
 */
public final class FimTransform {
    /** Where the suffix goes. */
    public enum Mode {
        /** Prefix, suffix, middle. */
        PSM,
        /** Suffix, then prefix and middle as one continuation. */
        SPM
    }

    private final int prefixToken;
    private final int suffixToken;
    private final int middleToken;
    private final Mode mode;

    private FimTransform(int prefixToken, int suffixToken, int middleToken, Mode mode) {
        this.prefixToken = prefixToken;
        this.suffixToken = suffixToken;
        this.middleToken = middleToken;
        this.mode = mode;
    }

    /**
     * Creates the transform.
     *
     * @param prefixToken the {@code [PRE]} sentinel id
     * @param suffixToken the {@code [SUF]} sentinel id
     * @param middleToken the {@code [MID]} sentinel id
     * @param mode the arrangement
     * @return the transform
     */
    public static FimTransform of(int prefixToken, int suffixToken, int middleToken, Mode mode) {
        return new FimTransform(prefixToken, suffixToken, middleToken, mode);
    }

    /** @return the number of sentinel tokens a transformed document gains */
    public static int overhead() {
        return 3;
    }

    /**
     * Transforms a document at two uniformly random cut points.
     *
     * @param document the tokens
     * @param random the source of cut points
     * @return the rearranged document, {@link #overhead()} tokens longer
     */
    public int[] apply(int[] document, RandomGenerator random) {
        int a = random.nextInt(document.length + 1);
        int b = random.nextInt(document.length + 1);
        return apply(document, Math.min(a, b), Math.max(a, b));
    }

    /**
     * Transforms a document at given cut points.
     *
     * @param document the tokens
     * @param start where the middle starts
     * @param end where the middle ends (exclusive)
     * @return the rearranged document
     */
    public int[] apply(int[] document, int start, int end) {
        if (start < 0 || end < start || end > document.length) {
            throw new IllegalArgumentException("bad cut points " + start + ", " + end);
        }
        int[] prefix = java.util.Arrays.copyOfRange(document, 0, start);
        int[] middle = java.util.Arrays.copyOfRange(document, start, end);
        int[] suffix = java.util.Arrays.copyOfRange(document, end, document.length);
        int[] prompt = infillPrompt(prefix, suffix);
        int[] out = java.util.Arrays.copyOf(prompt, prompt.length + middle.length);
        System.arraycopy(middle, 0, out, prompt.length, middle.length);
        return out;
    }

    /**
     * The prompt for infilling: the model's continuation is the middle.
     *
     * @param prefix the text before the gap
     * @param suffix the text after it
     * @return the prompt
     */
    public int[] infillPrompt(int[] prefix, int[] suffix) {
        int[] out = new int[prefix.length + suffix.length + 3];
        int i = 0;
        if (mode == Mode.PSM) {
            out[i++] = prefixToken;
            System.arraycopy(prefix, 0, out, i, prefix.length);
            i += prefix.length;
            out[i++] = suffixToken;
            System.arraycopy(suffix, 0, out, i, suffix.length);
            i += suffix.length;
            out[i] = middleToken;
        } else {
            out[i++] = prefixToken;
            out[i++] = suffixToken;
            System.arraycopy(suffix, 0, out, i, suffix.length);
            i += suffix.length;
            out[i++] = middleToken;
            System.arraycopy(prefix, 0, out, i, prefix.length);
        }
        return out;
    }
}
