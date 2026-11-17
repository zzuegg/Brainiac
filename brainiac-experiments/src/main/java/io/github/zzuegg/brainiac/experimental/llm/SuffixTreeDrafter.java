package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.llm.LanguageModel;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * SuffixDecoding: model-free speculation from a suffix tree over earlier outputs and the
 * current prompt. Agentic and tool-calling workloads repeat themselves (the same call
 * formats, the same reasoning steps, the same observations across requests), so the text
 * that followed the longest earlier match of what was just written is a good guess for what
 * comes next. The tree holds every suffix (up to {@code depth} tokens) of every finished
 * output, plus a per-request tree over the prompt and the text written so far; each node
 * counts how often its path occurred.
 *
 * <p>A draft starts at the node of the longest match of the context's end and follows the
 * most frequent child while that child's share of its parent's count stays above
 * {@code minShare}; its length is capped <em>adaptively</em> at {@code alpha} times the match
 * length (a long match is strong evidence, a one-token match is not), so weak guesses cost
 * few verified tokens. The target checks the whole draft in one pass and keeps the longest
 * agreeing prefix plus its own next token: greedy output is unchanged.
 *
 * <p>{@link #generate} runs this loop on any {@link LanguageModel} that can
 * {@linkplain LanguageModel.State#truncate go back}; the drafter is also a
 * {@link LanguageModel} itself (one-hot logits on its guess), so it drops into the core
 * {@code Generator.speculative} as a fixed-length draft, as the core {@code PromptLookup} does.
 *
 * <p>Oliaro et al., "SuffixDecoding: Extreme Speculative Decoding for Emerging AI
 * Applications" (2024), <a href="https://arxiv.org/abs/2411.04975">arXiv:2411.04975</a>.
 */
public final class SuffixTreeDrafter implements LanguageModel {

    /**
     * A generation.
     *
     * @param tokens the new tokens, equal to the target's greedy decoding
     * @param passes the target passes run
     * @param verified the tokens the target read in verification passes (its compute)
     * @param accepted the draft tokens kept
     */
    public record Result(int[] tokens, int passes, int verified, int accepted) {

        /** @return new tokens per target pass */
        public double tokensPerPass() {
            return tokens.length / (double) passes;
        }
    }

    /** A trie node: children by token and how often the path to it occurred. */
    private static final class Node {
        final Map<Integer, Node> children = new HashMap<>();
        int count;
    }

    private final int vocab;
    private final int depth;
    private final double alpha;
    private final double minShare;
    private final int maxDraft;
    private final Node global = new Node();

    /**
     * @param vocab the vocabulary size
     * @param depth the longest suffix stored and matched
     * @param alpha the most draft tokens per matched token
     * @param minShare the least share of its parent's count a followed child must have
     * @param maxDraft the most draft tokens per pass
     */
    public SuffixTreeDrafter(int vocab, int depth, double alpha, double minShare, int maxDraft) {
        if (vocab < 1 || depth < 1 || alpha <= 0 || maxDraft < 1) {
            throw new IllegalArgumentException("bad drafter settings");
        }
        this.vocab = vocab;
        this.depth = depth;
        this.alpha = alpha;
        this.minShare = minShare;
        this.maxDraft = maxDraft;
    }

    /**
     * Adds a finished output (or any reference text) to the global tree.
     *
     * @param tokens the text
     */
    public void add(int[] tokens) {
        insert(global, tokens, tokens.length);
    }

    private void insert(Node root, int[] tokens, int length) {
        for (int i = 0; i < length; i++) {
            Node n = root;
            n.count++;
            for (int j = i; j < Math.min(length, i + depth); j++) {
                n = n.children.computeIfAbsent(tokens[j], k -> new Node());
                n.count++;
            }
        }
    }

    /**
     * Guesses the continuation of a context from the global tree.
     *
     * @param context the prompt and the text so far
     * @return the draft, possibly empty
     */
    public int[] propose(int[] context) {
        return propose(context, context.length, null);
    }

    /** The draft from the longest match in the global and the request's own tree, the better scored. */
    private int[] propose(int[] context, int length, Node local) {
        int[] best = new int[0];
        double bestScore = -1;
        for (Node root : local == null ? List.of(global) : List.of(global, local)) {
            for (int m = Math.min(depth - 1, length); m >= 1; m--) {
                Node n = root;
                for (int j = length - m; j < length && n != null; j++) {
                    n = n.children.get(context[j]);
                }
                if (n == null || n.children.isEmpty()) {
                    continue;
                }
                int cap = Math.min(maxDraft, (int) Math.ceil(alpha * m));
                List<Integer> draft = new ArrayList<>();
                double score = 0;
                double prob = 1;
                while (draft.size() < cap && !n.children.isEmpty()) {
                    Map.Entry<Integer, Node> top = null;
                    for (Map.Entry<Integer, Node> e : n.children.entrySet()) {
                        if (top == null || e.getValue().count > top.getValue().count) {
                            top = e;
                        }
                    }
                    double share = top.getValue().count / (double) Math.max(1, n.count);
                    if (share < minShare) {
                        break;
                    }
                    prob *= share;
                    score += prob;   // expected accepted tokens along the path
                    draft.add(top.getKey());
                    n = top.getValue();
                }
                if (score > bestScore) {
                    bestScore = score;
                    best = draft.stream().mapToInt(Integer::intValue).toArray();
                }
                break;   // the longest match of this tree
            }
        }
        return best;
    }

    /**
     * Greedy generation verified by the target, adding the output to the global tree after.
     *
     * @param target the model; its state must support truncation
     * @param prompt the prompt
     * @param maxNew tokens to generate
     * @return the tokens and counts
     */
    public Result generate(LanguageModel target, int[] prompt, int maxNew) {
        Node local = new Node();
        int[] text = Arrays.copyOf(prompt, prompt.length + maxNew + maxDraft + 1);
        int size = prompt.length;
        insert(local, text, size);
        LanguageModel.State state = target.newState(1);
        float[] logits = target.next(state, new int[][] {prompt}, null).toArray();
        int passes = 1;
        int verified = 0;
        int accepted = 0;
        int pending = LogitMath.argmax(logits);
        int produced = 0;
        while (true) {
            text[size++] = pending;
            produced++;
            if (produced >= maxNew) {
                break;
            }
            int[] draft = propose(text, size, local);
            int[] read = new int[1 + draft.length];
            read[0] = pending;
            System.arraycopy(draft, 0, read, 1, draft.length);
            int base = state.length();
            float[] all = target.nextAll(state, new int[][] {read}).toArray();
            passes++;
            verified += read.length;
            int keep = 0;
            while (keep < draft.length && produced < maxNew
                    && LogitMath.argmax(Arrays.copyOfRange(all, keep * vocab, (keep + 1) * vocab)) == draft[keep]) {
                text[size++] = draft[keep];
                keep++;
                produced++;
            }
            accepted += keep;
            if (produced >= maxNew) {
                break;
            }
            pending = LogitMath.argmax(Arrays.copyOfRange(all, keep * vocab, (keep + 1) * vocab));
            state = state.truncate(base + 1 + keep);
            // grow the request's own tree with the newly accepted text
            local = new Node();
            insert(local, text, size);
        }
        int[] out = Arrays.copyOfRange(text, prompt.length, prompt.length + maxNew);
        add(out);
        return new Result(out, passes, verified, accepted);
    }

    // ---------------------------------------------------------------- LanguageModel (fixed-length draft)

    /** The text read so far. */
    private static final class Text implements State {
        int[] tokens = new int[16];
        int size;

        @Override
        public int length() {
            return size;
        }

        @Override
        public State select(int[] rows) {
            if (rows.length != 1 || rows[0] != 0) {
                throw new UnsupportedOperationException("one sequence only");
            }
            return this;
        }

        @Override
        public State truncate(int length) {
            size = length;
            return this;
        }
    }

    @Override
    public int vocabSize() {
        return vocab;
    }

    @Override
    public State newState(int batch) {
        if (batch != 1) {
            throw new IllegalArgumentException("one sequence only");
        }
        return new Text();
    }

    @Override
    public Tensor next(State state, int[][] tokens, boolean[][] padding) {
        Text t = (Text) state;
        for (int tok : tokens[0]) {
            if (t.size == t.tokens.length) {
                t.tokens = Arrays.copyOf(t.tokens, 2 * t.size);
            }
            t.tokens[t.size++] = tok;
        }
        Node local = new Node();
        insert(local, t.tokens, t.size);
        int[] d = propose(t.tokens, t.size, local);
        float[] logits = new float[vocab];
        if (d.length > 0) {
            Arrays.fill(logits, -1e4f);
            logits[d[0]] = 0;
        }
        return Tensor.of(Shape.of(1, vocab), logits);
    }

    @Override
    public String toString() {
        return "SuffixTreeDrafter(depth " + depth + ", alpha " + alpha + ", " + global.count + " suffixes)";
    }
}
