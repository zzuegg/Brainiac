package io.github.zzuegg.brainiac.experimental.llm;

import java.util.Arrays;
import java.util.random.RandomGenerator;

/**
 * The T5 span-corruption transform: random spans of a sequence are each replaced by one
 * sentinel token, and the target lists every sentinel followed by the span it replaced:
 *
 * <pre>
 *   tokens   a b c d e f g h
 *   input    a [S0] d e f [S1] h
 *   target   [S0] b c [S1] g
 * </pre>
 *
 * An encoder-decoder learns to read the corrupted input and write only the missing spans,
 * which costs fewer target tokens than reconstructing the whole sequence. This version uses a
 * fixed number of spans of fixed length, so that every example of a length has the same input
 * and target lengths (convenient for batching); spans never overlap or touch.
 *
 * <p>Paper: Raffel et al., "Exploring the Limits of Transfer Learning with a Unified
 * Text-to-Text Transformer" (JMLR 2020), <a href="https://arxiv.org/abs/1910.10683">arXiv:1910.10683</a>.
 */
public final class SpanCorruption {

    /**
     * A corrupted example.
     *
     * @param input the sequence with sentinels in place of the spans
     * @param target each sentinel followed by its span
     */
    public record Example(int[] input, int[] target) {
    }

    private final int firstSentinel;
    private final int spans;
    private final int spanLength;

    private SpanCorruption(int firstSentinel, int spans, int spanLength) {
        if (spans < 1 || spanLength < 1) {
            throw new IllegalArgumentException("need at least one span of at least one token");
        }
        this.firstSentinel = firstSentinel;
        this.spans = spans;
        this.spanLength = spanLength;
    }

    /**
     * Creates the transform.
     *
     * @param firstSentinel the id of sentinel 0; sentinel {@code k} is {@code firstSentinel + k}
     * @param spans the spans per sequence
     * @param spanLength the tokens per span
     * @return the transform
     */
    public static SpanCorruption of(int firstSentinel, int spans, int spanLength) {
        return new SpanCorruption(firstSentinel, spans, spanLength);
    }

    /**
     * @param length the sequence length
     * @return the corrupted input length
     */
    public int inputLength(int length) {
        return length - spans * (spanLength - 1);
    }

    /** @return the target length */
    public int targetLength() {
        return spans * (spanLength + 1);
    }

    /** @return the number of sentinel ids used */
    public int sentinels() {
        return spans;
    }

    /**
     * Corrupts at random span positions.
     *
     * @param tokens the sequence
     * @param random the source of span positions
     * @return the example
     */
    public Example apply(int[] tokens, RandomGenerator random) {
        int free = tokens.length - spans * spanLength - (spans - 1);
        if (free < 0) {
            throw new IllegalArgumentException("a sequence of " + tokens.length + " cannot hold " + spans
                    + " separated spans of " + spanLength);
        }
        // distribute the free tokens over spans + 1 gaps (stars and bars)
        int[] cuts = new int[spans];
        for (int i = 0; i < spans; i++) {
            cuts[i] = random.nextInt(free + 1);
        }
        Arrays.sort(cuts);
        int[] starts = new int[spans];
        for (int i = 0; i < spans; i++) {
            starts[i] = cuts[i] + i * (spanLength + 1);
        }
        return apply(tokens, starts);
    }

    /**
     * Corrupts at given span starts.
     *
     * @param tokens the sequence
     * @param starts increasing span starts, at least {@code spanLength + 1} apart
     * @return the example
     */
    public Example apply(int[] tokens, int[] starts) {
        if (starts.length != spans) {
            throw new IllegalArgumentException("expected " + spans + " starts, got " + starts.length);
        }
        int[] input = new int[inputLength(tokens.length)];
        int[] target = new int[targetLength()];
        int in = 0;
        int out = 0;
        int next = 0;
        for (int t = 0; t < tokens.length; ) {
            if (next < spans && t == starts[next]) {
                input[in++] = firstSentinel + next;
                target[out++] = firstSentinel + next;
                for (int k = 0; k < spanLength; k++) {
                    target[out++] = tokens[t++];
                }
                next++;
            } else {
                input[in++] = tokens[t++];
            }
        }
        if (next != spans || in != input.length) {
            throw new IllegalArgumentException("span starts " + Arrays.toString(starts) + " do not fit");
        }
        return new Example(input, target);
    }

    @Override
    public String toString() {
        return "SpanCorruption(" + spans + " x " + spanLength + ")";
    }
}
