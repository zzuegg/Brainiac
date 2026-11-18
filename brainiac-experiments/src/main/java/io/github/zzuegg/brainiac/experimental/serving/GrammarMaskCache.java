package io.github.zzuegg.brainiac.experimental.serving;

import io.github.zzuegg.brainiac.llm.Constraint;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;

/**
 * A token-mask cache for grammar-constrained decoding, after XGrammar (Dong et al., MLSys 2025,
 * <a href="https://arxiv.org/abs/2411.15100">arXiv:2411.15100</a>). Checking every vocabulary
 * token against the grammar's pushdown automaton at every step is the cost of structured
 * output. XGrammar observes that most tokens' validity depends only on where the automaton
 * stands, so their masks can be computed once and reused, and it overlaps the remaining mask
 * work with the model's forward pass.
 *
 * <p>This cache works on any core {@link Constraint} (the GBNF {@code Grammar}'s pushdown
 * automaton, a regular expression or a JSON schema) through its matchers, which compare equal
 * exactly when they stand at the same automaton state (for a grammar: the same set of rule
 * stacks):
 * <ul>
 * <li>a mask is computed once per distinct state with the core {@link Constraint.Vocabulary}
 *     prefix-sharing scan and stored as a bitmask; every later visit is a lookup;</li>
 * <li>{@link #precompute} walks the states reachable from a start state ahead of time (the
 *     ahead-of-time cache of the paper; grammars with bounded nesting have few states);</li>
 * <li>{@link #prefetch} computes the next mask on another thread while the model runs.</li>
 * </ul>
 * The paper's further split of each state's tokens into context-independent ones (decided by
 * the top rule position alone, shared by every stack with that top) and context-dependent
 * ones needs the automaton's stack, which the core {@code Grammar} does not expose; with it,
 * masks would be shared between states that differ only below the top of the stack.
 */
public final class GrammarMaskCache {

    private final Constraint.Vocabulary vocabulary;
    private final Map<Constraint.Matcher, long[]> masks = new HashMap<>();
    private long hits;
    private long misses;

    private GrammarMaskCache(Constraint.Vocabulary vocabulary) {
        this.vocabulary = Objects.requireNonNull(vocabulary, "vocabulary");
    }

    /**
     * @param vocabulary the token texts
     * @return an empty cache
     */
    public static GrammarMaskCache of(Constraint.Vocabulary vocabulary) {
        return new GrammarMaskCache(vocabulary);
    }

    /**
     * The tokens whose whole text keeps the output acceptable.
     *
     * @param m where the output stands
     * @return per token whether it is allowed
     */
    public boolean[] allowed(Constraint.Matcher m) {
        long[] bits = bits(m);
        boolean[] ok = new boolean[vocabulary.size()];
        for (int t = 0; t < ok.length; t++) {
            ok[t] = (bits[t >>> 6] & (1L << t)) != 0;
        }
        return ok;
    }

    /**
     * The mask as a bitmask, {@code bit t of word t / 64}.
     *
     * @param m where the output stands
     * @return the bitmask (shared: do not modify)
     */
    public synchronized long[] bits(Constraint.Matcher m) {
        long[] b = masks.get(m);
        if (b != null) {
            hits++;
            return b;
        }
        misses++;
        boolean[] ok = vocabulary.allowed(m);
        b = new long[(ok.length + 63) / 64];
        for (int t = 0; t < ok.length; t++) {
            if (ok[t]) {
                b[t >>> 6] |= 1L << t;
            }
        }
        masks.put(m, b);
        return b;
    }

    /**
     * Fills the cache with every state reachable from {@code start} by whole tokens, up to a
     * number of states.
     *
     * @param start the start state
     * @param maxStates the most states visited
     * @return the states cached
     */
    public int precompute(Constraint.Matcher start, int maxStates) {
        Deque<Constraint.Matcher> todo = new ArrayDeque<>();
        todo.add(start);
        int visited = 0;
        java.util.Set<Constraint.Matcher> seen = new java.util.HashSet<>();
        seen.add(start);
        while (!todo.isEmpty() && visited < maxStates) {
            Constraint.Matcher m = todo.poll();
            visited++;
            boolean[] ok = allowed(m);
            for (int t = 0; t < ok.length; t++) {
                if (ok[t]) {
                    Constraint.Matcher n = vocabulary.advance(m, t);
                    if (n != null && seen.add(n)) {
                        todo.add(n);
                    }
                }
            }
        }
        hits = 0;
        misses = 0;
        return visited;
    }

    /**
     * Starts computing a mask on another thread, to overlap with the model's forward pass.
     *
     * @param m where the output will stand
     * @param executor the executor
     * @return the mask, when ready
     */
    public CompletableFuture<long[]> prefetch(Constraint.Matcher m, Executor executor) {
        return CompletableFuture.supplyAsync(() -> bits(m), executor);
    }

    /** @return lookups answered from the cache since the last precompute */
    public long hits() {
        return hits;
    }

    /** @return lookups that had to scan the vocabulary since the last precompute */
    public long misses() {
        return misses;
    }

    /** @return the states cached */
    public int states() {
        return masks.size();
    }

    @Override
    public String toString() {
        return "GrammarMaskCache(" + masks.size() + " states, " + hits + " hits, " + misses + " misses)";
    }
}
