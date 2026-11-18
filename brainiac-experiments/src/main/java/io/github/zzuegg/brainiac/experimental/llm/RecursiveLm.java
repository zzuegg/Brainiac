package io.github.zzuegg.brainiac.experimental.llm;

import java.util.Arrays;
import java.util.Objects;

/**
 * A recursive language model, RLM (Zhang, Kraska and Khattab, "Recursive language models",
 * <a href="https://arxiv.org/abs/2512.24601">arXiv:2512.24601</a>): the long context is never put
 * into the model's window. It is held in an environment, and the model is called on slices of
 * it; each sub-call's answer is written back as a short piece of context, and the model is called
 * again on the concatenated sub-answers, recursively, until the remaining context fits one call.
 * In the paper the root model decides how to slice (it writes code in a REPL); here the
 * decomposition is the fixed one such code most often is, contiguous slices of the window size,
 * and every call, leaf or combining, is the same model.
 */
public final class RecursiveLm {

    /** One model call on a context that fits the window. */
    @FunctionalInterface
    public interface Call {
        /**
         * @param query the question
         * @param context at most the window's worth of context
         * @return the answer
         */
        int[] answer(int[] query, int[] context);
    }

    /** Writes a sub-call's answer as context for the next level. */
    @FunctionalInterface
    public interface Encoder {
        /**
         * @param query the question
         * @param answer a sub-call's answer
         * @return the context tokens that stand for it
         */
        int[] encode(int[] query, int[] answer);
    }

    private final int window;
    private final Call call;
    private final Encoder encoder;
    private long calls;
    private int depth;

    private RecursiveLm(int window, Call call, Encoder encoder) {
        this.window = window;
        this.call = call;
        this.encoder = encoder;
    }

    /**
     * Creates a recursive model.
     *
     * @param window the most context tokens one call reads
     * @param call the model call
     * @param encoder writes a sub-answer back as context; must be shorter than {@code window}
     * @return the recursive model
     */
    public static RecursiveLm of(int window, Call call, Encoder encoder) {
        if (window < 2) {
            throw new IllegalArgumentException("window must be at least 2");
        }
        return new RecursiveLm(window, Objects.requireNonNull(call), Objects.requireNonNull(encoder));
    }

    /**
     * Answers a question about a context of any length.
     *
     * @param query the question
     * @param context the whole context
     * @return the answer
     */
    public int[] answer(int[] query, int[] context) {
        return answer(query, context, 1);
    }

    private int[] answer(int[] query, int[] context, int level) {
        depth = Math.max(depth, level);
        if (context.length <= window) {
            calls++;
            return call.answer(query, context);
        }
        int pieces = (context.length + window - 1) / window;
        int[] next = new int[0];
        for (int p = 0; p < pieces; p++) {
            int[] slice = Arrays.copyOfRange(context, p * window, Math.min(context.length, (p + 1) * window));
            int[] sub = answer(query, slice, level + 1);
            int[] enc = encoder.encode(query, sub);
            int[] grown = Arrays.copyOf(next, next.length + enc.length);
            System.arraycopy(enc, 0, grown, next.length, enc.length);
            next = grown;
        }
        if (next.length >= context.length) {
            throw new IllegalStateException("encoded sub-answers do not shrink the context");
        }
        return answer(query, next, level + 1);
    }

    /** @return model calls so far */
    public long calls() {
        return calls;
    }

    /** @return the deepest recursion level reached so far */
    public int depth() {
        return depth;
    }

    @Override
    public String toString() {
        return "RecursiveLm(window=" + window + ", calls=" + calls + ")";
    }
}
