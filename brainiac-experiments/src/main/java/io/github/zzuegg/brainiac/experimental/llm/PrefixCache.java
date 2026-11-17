package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.llm.LanguageModel;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.function.Consumer;
import java.util.function.UnaryOperator;

/**
 * Reuses the key/value cache of prompt prefixes across requests, as SGLang's RadixAttention
 * and vLLM's automatic prefix caching do: requests that share a system prompt, few-shot
 * examples or an earlier chat turn read only the part that is new. Cached states hang on the
 * nodes of a radix tree over token ids; a lookup walks the tree as far as the prompt matches,
 * copies a state from below that point and {@linkplain LanguageModel.State#truncate cuts} it
 * back to the match. At most {@code capacity} states are kept, the least recently used going
 * first.
 *
 * <p>It works with any {@link LanguageModel} whose states can be copied and truncated. The
 * core {@code LanguageModel.State} has no copy or close, so both are given as functions: for
 * the core {@code Llama} a copy is {@code s -> s.select(new int[] {0})} (a gather of the
 * whole cache, see {@link #forCore}); for {@link PagedLlama} it is a copy-on-write
 * {@linkplain PagedLlama#fork fork} that shares the prefix's blocks and copies nothing.
 *
 * <p>Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs"
 * (2024), <a href="https://arxiv.org/abs/2312.07104">arXiv:2312.07104</a>.
 */
public final class PrefixCache {

    /**
     * The result of {@link #prefill}.
     *
     * @param state the state after the whole prompt, owned by the caller
     * @param logits the next-token logits {@code [1, vocab]}
     * @param reused the prompt tokens taken from the cache
     */
    public record Prefill(LanguageModel.State state, Tensor logits, int reused) {
    }

    private static final class Node {
        int[] edge;
        final Map<Integer, Node> children = new HashMap<>();
        LanguageModel.State state;
        long used;

        Node(int[] edge) {
            this.edge = edge;
        }
    }

    private final LanguageModel model;
    private final UnaryOperator<LanguageModel.State> copy;
    private final Consumer<LanguageModel.State> release;
    private final int capacity;
    private final Node root = new Node(new int[0]);
    private int entries;
    private long clock;
    private long reusedTokens;
    private long computedTokens;

    /**
     * @param model the model
     * @param copy makes an independent copy of a one-row state
     * @param release frees a state the cache drops
     * @param capacity the most states kept
     */
    public PrefixCache(LanguageModel model, UnaryOperator<LanguageModel.State> copy,
            Consumer<LanguageModel.State> release, int capacity) {
        if (capacity < 1) {
            throw new IllegalArgumentException("capacity must be positive");
        }
        this.model = Objects.requireNonNull(model, "model");
        this.copy = Objects.requireNonNull(copy, "copy");
        this.release = Objects.requireNonNull(release, "release");
        this.capacity = capacity;
    }

    /**
     * A cache for a model whose {@code select} leaves the old state usable, as the core
     * {@code Llama}'s does: copies are gathers, and dropped states are left to the collector.
     *
     * @param model the model
     * @param capacity the most states kept
     * @return the cache
     */
    public static PrefixCache forCore(LanguageModel model, int capacity) {
        return new PrefixCache(model, s -> s.select(new int[] {0}), s -> { }, capacity);
    }

    /**
     * A cache over a paged model: copies share blocks.
     *
     * @param model the model
     * @param capacity the most states kept
     * @return the cache
     */
    public static PrefixCache forPaged(PagedLlama model, int capacity) {
        return new PrefixCache(model, model::fork, model::release, capacity);
    }

    /** @return the states held */
    public int size() {
        return entries;
    }

    /** @return the prompt tokens served from the cache so far */
    public long reusedTokens() {
        return reusedTokens;
    }

    /** @return the prompt tokens the model had to read so far */
    public long computedTokens() {
        return computedTokens;
    }

    /**
     * Reads a prompt, reusing the longest cached prefix, and caches the state after it.
     *
     * @param prompt the prompt, at least one token
     * @return the state and next-token logits
     */
    public Prefill prefill(int[] prompt) {
        if (prompt.length == 0) {
            throw new IllegalArgumentException("empty prompt");
        }
        LanguageModel.State state = null;
        int matched = 0;
        Node[] found = new Node[1];
        int m = match(prompt, found);
        int keep = Math.min(m, prompt.length - 1);   // the last token is read again to get its logits
        Node source = keep > 0 ? stateBelow(found[0]) : null;
        if (source != null) {
            source.used = ++clock;
            state = copy.apply(source.state);
            if (state.length() > keep) {
                state = state.truncate(keep);
            }
            matched = keep;
        }
        if (state == null) {
            state = model.newState(1);
        }
        int[][] rest = {Arrays.copyOfRange(prompt, matched, prompt.length)};
        Tensor logits = model.next(state, rest, null);
        reusedTokens += matched;
        computedTokens += rest[0].length;
        insert(prompt, state);
        return new Prefill(state, logits, matched);
    }

    /**
     * Caches a copy of a state that has read exactly {@code tokens}.
     *
     * @param tokens what the state has read
     * @param state the state; the caller keeps it
     */
    public void insert(int[] tokens, LanguageModel.State state) {
        Node node = root;
        int i = 0;
        while (i < tokens.length) {
            Node child = node.children.get(tokens[i]);
            if (child == null) {
                child = new Node(Arrays.copyOfRange(tokens, i, tokens.length));
                node.children.put(tokens[i], child);
                node = child;
                i = tokens.length;
                break;
            }
            int common = 0;
            while (common < child.edge.length && i + common < tokens.length
                    && child.edge[common] == tokens[i + common]) {
                common++;
            }
            if (common < child.edge.length) {   // split the edge where the token ids part
                Node mid = new Node(Arrays.copyOf(child.edge, common));
                child.edge = Arrays.copyOfRange(child.edge, common, child.edge.length);
                mid.children.put(child.edge[0], child);
                node.children.put(mid.edge[0], mid);
                child = mid;
            }
            node = child;
            i += common;
        }
        if (node.state != null) {
            release.accept(node.state);
            entries--;
        }
        node.state = copy.apply(state);
        node.used = ++clock;
        entries++;
        while (entries > capacity) {
            evictOldest();
        }
    }

    /** The tokens of the prompt the tree holds; {@code found[0]} the node at or below that point. */
    private int match(int[] prompt, Node[] found) {
        Node node = root;
        int i = 0;
        found[0] = root;
        while (i < prompt.length) {
            Node child = node.children.get(prompt[i]);
            if (child == null) {
                break;
            }
            int common = 0;
            while (common < child.edge.length && i + common < prompt.length
                    && child.edge[common] == prompt[i + common]) {
                common++;
            }
            i += common;
            found[0] = child;   // everything below it shares the matched tokens
            if (common < child.edge.length) {
                break;
            }
            node = child;
        }
        return i;
    }

    /** The most recently used state at or below a node, or null. */
    private static Node stateBelow(Node from) {
        Node best = null;
        Deque<Node> todo = new ArrayDeque<>();
        todo.push(from);
        while (!todo.isEmpty()) {
            Node n = todo.pop();
            if (n.state != null && (best == null || n.used > best.used)) {
                best = n;
            }
            n.children.values().forEach(todo::push);
        }
        return best;
    }

    private void evictOldest() {
        Node oldest = null;
        Deque<Node> todo = new ArrayDeque<>();
        todo.push(root);
        while (!todo.isEmpty()) {
            Node n = todo.pop();
            if (n.state != null && (oldest == null || n.used < oldest.used)) {
                oldest = n;
            }
            n.children.values().forEach(todo::push);
        }
        if (oldest != null) {
            release.accept(oldest.state);
            oldest.state = null;
            entries--;
        }
    }

    @Override
    public String toString() {
        return "PrefixCache[" + entries + "/" + capacity + " states, " + reusedTokens + " reused, " + computedTokens
                + " computed]";
    }
}
