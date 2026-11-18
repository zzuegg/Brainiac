package io.github.zzuegg.brainiac.experimental.evolve;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;
import java.util.function.BiFunction;

/**
 * Weighted bottom-up enumeration for straight-line programs over a list DSL: a pool of
 * values (one list per example), grown cheapest first by applying each primitive to the
 * cheapest unexpanded value. A value's weight is its parent's weight plus the primitive's
 * cost plus an optional per-value cost; with unit costs and no value cost this is the
 * breadth-first search of {@link GuidedEnumerator}. Observational equivalence drops a value
 * whose outputs on the examples equal those of one already in the pool.
 *
 * <p>The two hooks are what the guided searches plug in: {@link Bustle} sets a value cost
 * from a learned usefulness model, {@link Probe} sets primitive costs from a grammar it
 * learns during the search and uses the {@link Listener} to restart on partial solutions.
 * Cost is counted, as in {@link GuidedEnumerator}, in primitive applications to the whole
 * example set.
 *
 * <p>The search follows the bottom-up enumerators of BUSTLE (Odena et al., ICLR 2021,
 * <a href="https://arxiv.org/abs/2007.14381">arXiv:2007.14381</a>) and Probe (Barke, Peleg
 * and Polikarpova, OOPSLA 2020), restricted to unary primitives.
 */
public final class BottomUpSynthesis {

    /** A per-value cost added to the weight of every new value (higher is tried later). */
    @FunctionalInterface
    public interface ValueCost {
        /**
         * @param value the value, one list per example
         * @param inputs the example inputs
         * @param outputs the expected outputs
         * @return the extra weight, at least 0
         */
        double cost(int[][] value, int[][] inputs, int[][] outputs);
    }

    /** Sees every new value; returning {@code true} stops the search. */
    @FunctionalInterface
    public interface Listener {
        /**
         * @param program the program that made the value
         * @param value the value
         * @return whether to stop
         */
        boolean onValue(List<String> program, int[][] value);
    }

    /**
     * A search outcome.
     *
     * @param program the program found, or {@code null}
     * @param evaluated primitive applications spent
     * @param stopped whether the listener stopped the search
     */
    public record Result(List<String> program, long evaluated, boolean stopped) {
        /** @return whether a program was found */
        public boolean solved() {
            return program != null;
        }
    }

    private record Entry(double weight, long order, int[][] value, List<String> program) {
    }

    private final List<String> primitives;
    private final BiFunction<String, int[], int[]> semantics;

    /**
     * Creates the search for a DSL.
     *
     * @param primitives the primitive names
     * @param semantics applies a primitive to a list; {@code null} means the program fails
     */
    public BottomUpSynthesis(List<String> primitives, BiFunction<String, int[], int[]> semantics) {
        this.primitives = List.copyOf(primitives);
        this.semantics = semantics;
    }

    /** @return the primitive names */
    public List<String> primitives() {
        return primitives;
    }

    /**
     * Applies a primitive to every example.
     *
     * @param op the primitive
     * @param value one list per example
     * @return the results, or {@code null} if it fails on any example
     */
    public int[][] apply(String op, int[][] value) {
        int[][] out = new int[value.length][];
        for (int e = 0; e < value.length; e++) {
            out[e] = semantics.apply(op, value[e]);
            if (out[e] == null) {
                return null;
            }
        }
        return out;
    }

    /**
     * Unguided search: unit primitive costs, no value cost.
     *
     * @param inputs the example inputs
     * @param outputs the expected outputs
     * @param maxLength the longest program
     * @param budget the most primitive applications
     * @return the outcome
     */
    public Result search(int[][] inputs, int[][] outputs, int maxLength, long budget) {
        return search(inputs, outputs, maxLength, null, null, null, budget);
    }

    /**
     * Weighted bottom-up search.
     *
     * @param inputs the example inputs
     * @param outputs the expected outputs
     * @param maxLength the longest program
     * @param opCosts a cost per primitive of {@link #primitives()} ({@code null}: all 1)
     * @param valueCost a cost per new value, or {@code null}
     * @param listener sees every new value, or {@code null}
     * @param budget the most primitive applications
     * @return the first program found that reproduces every example
     */
    public Result search(int[][] inputs, int[][] outputs, int maxLength, double[] opCosts, ValueCost valueCost,
            Listener listener, long budget) {
        PriorityQueue<Entry> queue = new PriorityQueue<>((a, b) -> a.weight() != b.weight()
                ? Double.compare(a.weight(), b.weight()) : Long.compare(a.order(), b.order()));
        Set<String> seen = new HashSet<>();
        seen.add(Arrays.deepToString(inputs));
        long order = 0;
        queue.add(new Entry(0, order++, inputs, List.of()));
        long cost = 0;
        while (!queue.isEmpty()) {
            Entry e = queue.poll();
            if (e.program().size() >= maxLength) {
                continue;
            }
            for (int k = 0; k < primitives.size(); k++) {
                if (++cost > budget) {
                    return new Result(null, budget, false);
                }
                String op = primitives.get(k);
                int[][] out = apply(op, e.value());
                if (out == null) {
                    continue;
                }
                List<String> program = new ArrayList<>(e.program());
                program.add(op);
                if (Arrays.deepEquals(out, outputs)) {
                    return new Result(List.copyOf(program), cost, false);
                }
                if (!seen.add(Arrays.deepToString(out))) {
                    continue;
                }
                if (listener != null && listener.onValue(program, out)) {
                    return new Result(null, cost, true);
                }
                double w = e.weight() + (opCosts == null ? 1 : opCosts[k])
                        + (valueCost == null ? 0 : valueCost.cost(out, inputs, outputs));
                queue.add(new Entry(w, order++, out, program));
            }
        }
        return new Result(null, cost, false);
    }
}
