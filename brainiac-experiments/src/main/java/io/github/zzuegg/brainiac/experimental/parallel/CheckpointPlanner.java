package io.github.zzuegg.brainiac.experimental.parallel;

import io.github.zzuegg.brainiac.autograd.Recompute;
import io.github.zzuegg.brainiac.autograd.Var;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.function.Function;

/**
 * Plans activation checkpointing over a linear chain of operations: which outputs to keep for
 * the backward pass and which to recompute, trading peak memory against extra compute.
 *
 * <p>The chain is a list of {@link Node}s, the outputs of consecutive operations, each with its
 * size and the time its operation takes. A plan keeps some outputs; every maximal run of dropped
 * outputs is recomputed once during the backward pass from the kept output before it, by
 * re-running the run's operations and the operation that produces the next kept output (exactly
 * what the core {@link Recompute} does for a block). So
 * <pre>
 * peak  = sum(kept bytes) + max over runs (bytes of the run and its closing output)
 * extra = sum over runs (time of the run's operations and the closing one)
 * </pre>
 * Strategies:
 * <ul>
 *   <li>{@link #none()}: keep everything;</li>
 *   <li>{@link #everyKth(int)}: keep every k-th output; {@code k = sqrt(n)} is Chen et al.'s
 *       {@code O(sqrt n)} schedule, {@code k} = operations per block is whole-block
 *       {@link Recompute};</li>
 *   <li>{@link #selective(Set)}: drop only outputs of chosen kinds, such as the attention-score
 *       region that is large but cheap to recompute (Korthikanti et al.);</li>
 *   <li>{@link #optimal(long)}: the cheapest plan within a memory budget, from an exact dynamic
 *       programme over the kept outputs that keeps the Pareto front of (kept bytes, largest
 *       recomputed run, recompute time); the single-recomputation case of Checkmate's integer
 *       programme, solved exactly on a chain;</li>
 *   <li>{@link #revolveForwards(int, int)}: Griewank and Walther's binomial checkpointing
 *       (Revolve), which may recompute repeatedly, for a uniform chain with a fixed number of
 *       checkpoint slots.</li>
 * </ul>
 * {@link #run} executes a plan on real operations by wrapping each recomputed run in core
 * {@link Recompute}.
 *
 * <p>Papers: Chen et al., <a href="https://arxiv.org/abs/1604.06174">arXiv:1604.06174</a>;
 * Griewank and Walther, "Algorithm 799: Revolve" (2000); Korthikanti et al.,
 * <a href="https://arxiv.org/abs/2205.05198">arXiv:2205.05198</a>; Jain et al., "Checkmate",
 * <a href="https://arxiv.org/abs/1910.02653">arXiv:1910.02653</a>.
 */
public final class CheckpointPlanner {

    /**
     * One operation's output.
     *
     * @param name a label
     * @param kind the operation's kind, such as {@code matmul} or {@code attention}
     * @param bytes the output's size
     * @param cost the operation's time (any unit)
     */
    public record Node(String name, String kind, long bytes, double cost) {
    }

    /**
     * A plan and its cost.
     *
     * @param name a label
     * @param keep which outputs are kept
     * @param peakBytes peak activation memory during the backward pass
     * @param extraCost recompute time
     */
    public record Plan(String name, boolean[] keep, long peakBytes, double extraCost) {

        @Override
        public String toString() {
            return String.format("%s: peak %d B, extra %.4g", name, peakBytes, extraCost);
        }
    }

    private final List<Node> chain;
    private final long totalBytes;
    private final double totalCost;

    /**
     * Creates a planner.
     *
     * @param chain the operations' outputs in order; the last is always kept
     */
    public CheckpointPlanner(List<Node> chain) {
        if (chain.isEmpty()) {
            throw new IllegalArgumentException("empty chain");
        }
        this.chain = List.copyOf(chain);
        long b = 0;
        double c = 0;
        for (Node n : chain) {
            b += n.bytes();
            c += n.cost();
        }
        this.totalBytes = b;
        this.totalCost = c;
    }

    /** @return the forward time of the whole chain */
    public double forwardCost() {
        return totalCost;
    }

    /** @return the chain */
    public List<Node> chain() {
        return chain;
    }

    /**
     * Evaluates a keep mask.
     *
     * @param name a label
     * @param keep one flag per output; the last is forced to true
     * @return the plan
     */
    public Plan evaluate(String name, boolean[] keep) {
        int n = chain.size();
        boolean[] k = keep.clone();
        k[n - 1] = true;
        long kept = 0;
        long worstRun = 0;
        double extra = 0;
        int i = 0;
        while (i < n) {
            if (k[i]) {
                kept += chain.get(i).bytes();
                i++;
                continue;
            }
            int a = i;
            while (!k[i]) {
                i++;
            }
            // run a..i-1 dropped, i kept and recomputed with it
            long bytes = 0;
            for (int j = a; j <= i; j++) {
                bytes += chain.get(j).bytes();
                extra += chain.get(j).cost();
            }
            worstRun = Math.max(worstRun, bytes);
        }
        return new Plan(name, k, kept + worstRun, extra);
    }

    /** @return the plan that keeps every output */
    public Plan none() {
        boolean[] k = new boolean[chain.size()];
        Arrays.fill(k, true);
        return evaluate("no checkpointing", k);
    }

    /**
     * Keeps every {@code k}-th output (and the last).
     *
     * @param k the spacing, at least 1
     * @return the plan
     */
    public Plan everyKth(int k) {
        boolean[] keep = new boolean[chain.size()];
        for (int i = k - 1; i < keep.length; i += k) {
            keep[i] = true;
        }
        return evaluate("every " + k, keep);
    }

    /** @return Chen et al.'s plan, keeping every {@code ceil(sqrt n)}-th output */
    public Plan sqrt() {
        Plan p = everyKth((int) Math.ceil(Math.sqrt(chain.size())));
        return new Plan("sqrt(n)", p.keep(), p.peakBytes(), p.extraCost());
    }

    /**
     * Drops the outputs of chosen kinds and keeps the rest.
     *
     * @param drop the kinds to recompute
     * @return the plan
     */
    public Plan selective(Set<String> drop) {
        boolean[] keep = new boolean[chain.size()];
        for (int i = 0; i < keep.length; i++) {
            keep[i] = !drop.contains(chain.get(i).kind());
        }
        return evaluate("selective " + drop, keep);
    }

    /** A partial plan in the Pareto search: output {@code node} kept. */
    private record State(long kept, long maxRun, double cost, int node, State previous) {
    }

    private List<State> front;

    /**
     * The exact Pareto front of plans over (kept bytes, largest recomputed run, recompute
     * time), by dynamic programming over the kept outputs with dominance pruning.
     */
    private List<State> front() {
        if (front != null) {
            return front;
        }
        int n = chain.size();
        long[] b = new long[n];
        double[] c = new double[n];
        for (int i = 0; i < n; i++) {
            b[i] = chain.get(i).bytes();
            c[i] = chain.get(i).cost();
        }
        List<List<State>> at = new ArrayList<>();
        State start = new State(0, 0, 0, -1, null);
        for (int i = 0; i < n; i++) {
            List<State> cand = new ArrayList<>();
            long runBytes = 0;
            double runCost = 0;
            for (int j = i - 1; j >= -1; j--) {
                if (j < i - 1) {
                    runBytes += b[j + 1];
                    runCost += c[j + 1];
                }
                boolean run = j < i - 1;
                long withClosing = run ? runBytes + b[i] : 0;
                double add = run ? runCost + c[i] : 0;
                List<State> from = j < 0 ? List.of(start) : at.get(j);
                for (State st : from) {
                    cand.add(new State(st.kept() + b[i], Math.max(st.maxRun(), withClosing), st.cost() + add, i, st));
                }
            }
            at.add(prune(cand));
        }
        front = at.get(n - 1);
        return front;
    }

    /** Keeps the states no other state beats in all three of kept bytes, largest run and cost. */
    private static List<State> prune(List<State> cand) {
        cand.sort(java.util.Comparator.comparingLong(State::kept).thenComparingLong(State::maxRun)
                .thenComparingDouble(State::cost));
        List<State> keep = new ArrayList<>();
        // staircase of (largest run -> least cost) over the states kept so far, which all keep
        // no more bytes than the current one
        java.util.TreeMap<Long, Double> stairs = new java.util.TreeMap<>();
        for (State s : cand) {
            java.util.Map.Entry<Long, Double> e = stairs.floorEntry(s.maxRun());
            if (e != null && e.getValue() <= s.cost() + 1e-15) {
                continue;
            }
            keep.add(s);
            stairs.put(s.maxRun(), s.cost());
            java.util.Map.Entry<Long, Double> next = stairs.higherEntry(s.maxRun());
            while (next != null && next.getValue() >= s.cost()) {
                stairs.remove(next.getKey());
                next = stairs.higherEntry(s.maxRun());
            }
        }
        if (keep.size() > 4000) {
            // safety valve for very long chains: an evenly spread subset ordered by peak
            keep.sort(java.util.Comparator.comparingLong(st -> st.kept() + st.maxRun()));
            List<State> thin = new ArrayList<>();
            for (int i = 0; i < keep.size(); i += keep.size() / 2000 + 1) {
                thin.add(keep.get(i));
            }
            keep = thin;
        }
        return keep;
    }

    private Plan plan(String name, State s) {
        boolean[] keep = new boolean[chain.size()];
        for (State x = s; x != null && x.node() >= 0; x = x.previous()) {
            keep[x.node()] = true;
        }
        return evaluate(name, keep);
    }

    /**
     * The plan with the least recompute time whose peak fits the budget, from the exact Pareto
     * front of all keep masks.
     *
     * @param budget the peak-memory budget in bytes
     * @return the plan, or {@code null} if no plan fits
     */
    public Plan optimal(long budget) {
        State best = null;
        for (State s : front()) {
            if (s.kept() + s.maxRun() <= budget && (best == null || s.cost() < best.cost()
                    || (s.cost() == best.cost() && s.kept() + s.maxRun() < best.kept() + best.maxRun()))) {
                best = s;
            }
        }
        return best == null ? null : plan("optimal (budget " + budget + ")", best);
    }

    /**
     * The Pareto front of peak memory against recompute time over all keep masks.
     *
     * @return the non-dominated plans, increasing in peak memory
     */
    public List<Plan> optimalFront() {
        List<State> f = new ArrayList<>(front());
        f.sort(java.util.Comparator.comparingLong(st -> st.kept() + st.maxRun()));
        List<Plan> out = new ArrayList<>();
        double bestCost = Double.POSITIVE_INFINITY;
        for (State s : f) {
            if (s.cost() < bestCost - 1e-15) {
                bestCost = s.cost();
                out.add(plan("optimal", s));
            }
        }
        return out;
    }

    /**
     * Revolve's minimal number of forward steps to reverse a uniform chain of {@code steps}
     * operations with {@code slots} checkpoint slots, by the binomial recursion
     * {@code F(n, s) = min_j (j + F(n - j, s - 1) + F(j, s))}, {@code F(n, 1) = n (n - 1) / 2}.
     * The forward pass itself ({@code n - 1} steps before the first reversal) is included, so
     * {@code F(n, n) = n - 1}.
     *
     * @param steps {@code n}
     * @param slots {@code s}, at least 1
     * @return forward steps
     */
    public static long revolveForwards(int steps, int slots) {
        long[][] f = new long[steps + 1][slots + 1];
        for (int n = 1; n <= steps; n++) {
            f[n][1] = (long) n * (n - 1) / 2;
            for (int s = 2; s <= slots; s++) {
                if (n == 1) {
                    f[n][s] = 0;
                    continue;
                }
                long best = Long.MAX_VALUE;
                for (int j = 1; j < n; j++) {
                    best = Math.min(best, j + f[n - j][s - 1] + f[j][s]);
                }
                f[n][s] = best;
            }
        }
        return f[steps][slots];
    }

    /**
     * Runs a chain of operations under a plan: each recomputed run, with its closing operation,
     * goes through core {@link Recompute}, which keeps only its input and output.
     *
     * @param ops the operations, one per chain node
     * @param parameters each operation's parameters
     * @param keep the plan's keep mask
     * @param x the chain's input
     * @return the chain's output
     */
    public static Var run(List<Function<Var, Var>> ops, List<List<Var>> parameters, boolean[] keep, Var x) {
        int n = ops.size();
        Var h = x;
        int i = 0;
        while (i < n) {
            if (keep[i] && (i == 0 || keep[i - 1])) {
                h = ops.get(i).apply(h);
                i++;
                continue;
            }
            int a = i;
            while (i < n - 1 && !keep[i]) {
                i++;
            }
            int lo = a;
            int hi = i;
            List<Var> ps = new ArrayList<>();
            for (int j = lo; j <= hi; j++) {
                ps.addAll(parameters.get(j));
            }
            Function<Var, Var> segment = v -> {
                Var y = v;
                for (int j = lo; j <= hi; j++) {
                    y = ops.get(j).apply(y);
                }
                return y;
            };
            h = Recompute.apply(segment, h, ps);
            i++;
        }
        return h;
    }
}
