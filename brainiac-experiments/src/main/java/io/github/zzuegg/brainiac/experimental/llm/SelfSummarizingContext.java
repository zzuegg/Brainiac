package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.llm.LanguageModel;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A context window that summarises itself instead of dropping what falls out of it, in the
 * spirit of recursive summarisation for long dialogue (Wang et al. 2023,
 * <a href="https://arxiv.org/abs/2308.15022">arXiv:2308.15022</a>) and MemGPT's eviction
 * summaries (<a href="https://arxiv.org/abs/2310.08560">arXiv:2310.08560</a>). When the
 * next tokens would overflow the window, the model itself reads the oldest part followed by
 * a summary instruction and writes a summary (greedily, up to an end token); the context
 * becomes {@code summary + recent tail}, and the model's state is rebuilt from it. Earlier
 * summaries are part of the oldest part, so they are summarised again: memory of the whole
 * stream in a bounded window.
 *
 * <p>{@link Mode#TRUNCATE} is the baseline that keeps only the tail. Works with any
 * {@link LanguageModel}; one sequence (batch 1).
 */
public final class SelfSummarizingContext {

    /** What happens when the window is full. */
    public enum Mode {
        /** The model summarises the old part. */
        SUMMARIZE,
        /** The old part is dropped. */
        TRUNCATE
    }

    private final LanguageModel model;
    private final int window;
    private final Mode mode;
    private int[] instruction = new int[0];
    private int end = -1;
    private int keep;
    private int maxSummary;
    private int[] context = new int[0];
    private LanguageModel.State state;
    private float[] lastLogits;
    private final List<int[]> summaries = new ArrayList<>();
    private long modelTokens;

    private SelfSummarizingContext(LanguageModel model, int window, Mode mode) {
        if (window < 2) {
            throw new IllegalArgumentException("window must be at least 2");
        }
        this.model = model;
        this.window = window;
        this.mode = mode;
        this.keep = window / 4;
        this.maxSummary = window / 4;
        this.state = model.newState(1);
    }

    /**
     * A context over a model.
     *
     * @param model the language model
     * @param window the most tokens the context holds
     * @param mode what to do when it is full
     * @return the context
     */
    public static SelfSummarizingContext of(LanguageModel model, int window, Mode mode) {
        return new SelfSummarizingContext(model, window, mode);
    }

    /**
     * @param prompt the instruction read after the old part, such as "Summarize the above:"
     * @param endToken the token that ends a summary
     * @param maxTokens the longest summary
     * @return this context
     */
    public SelfSummarizingContext summarizeWith(int[] prompt, int endToken, int maxTokens) {
        this.instruction = prompt.clone();
        this.end = endToken;
        this.maxSummary = maxTokens;
        return this;
    }

    /**
     * @param tokens the recent tokens kept verbatim when the window is compressed
     * @return this context
     */
    public SelfSummarizingContext keepRecent(int tokens) {
        this.keep = tokens;
        return this;
    }

    /**
     * Reads tokens, compressing the context first when they would not fit.
     *
     * @param tokens the tokens
     */
    public void append(int... tokens) {
        int at = 0;
        while (at < tokens.length) {
            if (context.length >= window) {
                compress();
            }
            int n = Math.min(tokens.length - at, window - context.length);
            int[] part = Arrays.copyOfRange(tokens, at, at + n);
            read(part);
            at += n;
        }
    }

    /**
     * Generates greedily, each token joining the context.
     *
     * @param tokens the tokens to generate
     * @return the generated tokens
     */
    public int[] generate(int tokens) {
        int[] out = new int[tokens];
        for (int i = 0; i < tokens; i++) {
            if (lastLogits == null) {
                throw new IllegalStateException("append something first");
            }
            out[i] = argmax(lastLogits);
            append(out[i]);
        }
        return out;
    }

    /** @return the greedy next token after the current context */
    public int next() {
        return argmax(lastLogits);
    }

    private void read(int[] tokens) {
        lastLogits = model.next(state, new int[][] {tokens}, null).toArray();
        modelTokens += tokens.length;
        int[] c = Arrays.copyOf(context, context.length + tokens.length);
        System.arraycopy(tokens, 0, c, context.length, tokens.length);
        context = c;
    }

    private void compress() {
        int tail = Math.min(keep, context.length);
        int[] old = Arrays.copyOf(context, context.length - tail);
        int[] recent = Arrays.copyOfRange(context, context.length - tail, context.length);
        int[] summary = new int[0];
        if (mode == Mode.SUMMARIZE && old.length > 0) {
            summary = summarize(old);
            summaries.add(summary);
        }
        context = new int[0];
        state = model.newState(1);
        lastLogits = null;
        int[] rebuilt = Arrays.copyOf(summary, summary.length + recent.length);
        System.arraycopy(recent, 0, rebuilt, summary.length, recent.length);
        if (rebuilt.length >= window) {
            throw new IllegalStateException("summary and recent tail fill the whole window; shorten them");
        }
        if (rebuilt.length > 0) {
            read(rebuilt);
        }
    }

    /** The model reads the old part and the instruction on a fresh state and writes. */
    private int[] summarize(int[] old) {
        LanguageModel.State s = model.newState(1);
        int[] input = Arrays.copyOf(old, old.length + instruction.length);
        System.arraycopy(instruction, 0, input, old.length, instruction.length);
        float[] logits = model.next(s, new int[][] {input}, null).toArray();
        modelTokens += input.length;
        int[] out = new int[maxSummary];
        int n = 0;
        while (n < maxSummary) {
            int t = argmax(logits);
            if (t == end) {
                break;
            }
            out[n++] = t;
            logits = model.next(s, new int[][] {{t}}, null).toArray();
            modelTokens++;
        }
        return Arrays.copyOf(out, n);
    }

    private static int argmax(float[] p) {
        int best = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] > p[best]) {
                best = i;
            }
        }
        return best;
    }

    /** @return the tokens in the window: the latest summary followed by recent tokens */
    public int[] context() {
        return context.clone();
    }

    /** @return the summaries written so far, oldest first */
    public List<int[]> summaries() {
        return summaries.stream().map(int[]::clone).toList();
    }

    /** @return the tokens the model has read, summaries and rebuilding included: the cost */
    public long modelTokens() {
        return modelTokens;
    }

    @Override
    public String toString() {
        return "SelfSummarizingContext(" + mode + ", " + context.length + "/" + window + ", " + summaries.size()
                + " summaries)";
    }
}
