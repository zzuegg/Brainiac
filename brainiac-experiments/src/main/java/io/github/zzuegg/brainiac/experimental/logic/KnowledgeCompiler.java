package io.github.zzuegg.brainiac.experimental.logic;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Knowledge compilation of CNF formulas into tractable circuits, with weighted model counting
 * (Bryant, "Graph-based algorithms for Boolean function manipulation", IEEE TC 1986; Darwiche,
 * "A compiler for deterministic, decomposable negation normal form", AAAI 2002; Sang, Beame &amp;
 * Kautz, "Performing Bayesian inference by weighted model counting", AAAI 2005; Chavira &amp;
 * Darwiche, "On probabilistic inference by weighted model counting", AIJ 2008).
 *
 * <ul>
 *   <li>{@link #compileObdd}: a reduced ordered BDD built by conjoining clause BDDs with a memoised
 *       apply, under a static or min-fill variable order. Canonical, but its size depends on the
 *       order.</li>
 *   <li>{@link #compileDnnf}: decision-DNNF from the trace of an exhaustive DPLL with unit
 *       propagation, decomposition into variable-disjoint components (AND nodes) and a cache of
 *       compiled components keyed by their residual clauses (the c2d / sharpSAT recipe).</li>
 *   <li>{@link Circuit}: both compile to one circuit form (literals, decomposable ANDs,
 *       deterministic ORs), which is smoothed so that weighted model counting is a single
 *       bottom-up pass and its gradient with respect to every literal weight is one backward
 *       pass. Marginals follow from the gradient; semantic loss is {@code -log WMC}.</li>
 *   <li>{@link #encodeBayesNet}: a Bayesian network as weighted CNF (indicator variables per
 *       value, parameter variables per CPT entry; zero entries become clauses and one entries
 *       drop their parameter, which is how WMC exploits determinism).</li>
 * </ul>
 * Literals use the DIMACS convention.
 */
public final class KnowledgeCompiler {

    private KnowledgeCompiler() {
    }

    // ---- circuits -----------------------------------------------------------------------------

    /** A smooth, decomposable, deterministic circuit over variables {@code 1..n}. */
    public static final class Circuit {

        private static final int TRUE = 0;
        private static final int FALSE = 1;
        private static final int LIT = 2;
        private static final int AND = 3;
        private static final int OR = 4;
        private static final int FREE = 5;

        private final int variables;
        private final List<int[]> nodes = new ArrayList<>();
        private int root;
        private int logicalSize;
        private double millis;

        private Circuit(int variables) {
            this.variables = variables;
        }

        private int add(int... node) {
            nodes.add(node);
            return nodes.size() - 1;
        }

        /** @return the number of variables */
        public int variables() {
            return variables;
        }

        /** @return the circuit size before smoothing: nodes plus edges */
        public int size() {
            return logicalSize;
        }

        /** @return the compile time in milliseconds */
        public double millis() {
            return millis;
        }

        /**
         * Weighted model count.
         *
         * @param pos weight of {@code +v} at index {@code v-1}
         * @param neg weight of {@code -v} at index {@code v-1}
         * @return the WMC
         */
        public double wmc(double[] pos, double[] neg) {
            return values(pos, neg)[root];
        }

        /** @return the number of models */
        public double modelCount() {
            double[] one = new double[variables];
            Arrays.fill(one, 1);
            return wmc(one, one);
        }

        private double[] values(double[] pos, double[] neg) {
            double[] val = new double[nodes.size()];
            for (int i = 0; i < nodes.size(); i++) {
                int[] n = nodes.get(i);
                switch (n[0]) {
                    case TRUE -> val[i] = 1;
                    case FALSE -> val[i] = 0;
                    case LIT -> val[i] = n[1] > 0 ? pos[n[1] - 1] : neg[-n[1] - 1];
                    case FREE -> val[i] = pos[n[1] - 1] + neg[n[1] - 1];
                    case AND -> {
                        double p = 1;
                        for (int k = 1; k < n.length; k++) {
                            p *= val[n[k]];
                        }
                        val[i] = p;
                    }
                    default -> {
                        double s = 0;
                        for (int k = 1; k < n.length; k++) {
                            s += val[n[k]];
                        }
                        val[i] = s;
                    }
                }
            }
            return val;
        }

        /**
         * The WMC and its gradient with respect to every literal weight.
         *
         * @param pos positive weights
         * @param neg negative weights
         * @param dPos receives {@code dWMC / dpos}
         * @param dNeg receives {@code dWMC / dneg}
         * @return the WMC
         */
        public double gradient(double[] pos, double[] neg, double[] dPos, double[] dNeg) {
            double[] val = values(pos, neg);
            double[] adj = new double[nodes.size()];
            adj[root] = 1;
            Arrays.fill(dPos, 0);
            Arrays.fill(dNeg, 0);
            for (int i = nodes.size() - 1; i >= 0; i--) {
                if (adj[i] == 0) {
                    continue;
                }
                int[] n = nodes.get(i);
                switch (n[0]) {
                    case LIT -> {
                        if (n[1] > 0) {
                            dPos[n[1] - 1] += adj[i];
                        } else {
                            dNeg[-n[1] - 1] += adj[i];
                        }
                    }
                    case FREE -> {
                        dPos[n[1] - 1] += adj[i];
                        dNeg[n[1] - 1] += adj[i];
                    }
                    case AND -> {
                        for (int k = 1; k < n.length; k++) {
                            double others = 1;
                            for (int j = 1; j < n.length; j++) {
                                if (j != k) {
                                    others *= val[n[j]];
                                }
                            }
                            adj[n[k]] += adj[i] * others;
                        }
                    }
                    case OR -> {
                        for (int k = 1; k < n.length; k++) {
                            adj[n[k]] += adj[i];
                        }
                    }
                    default -> {
                    }
                }
            }
            return val[root];
        }

        /**
         * Marginals {@code P(v = true)} under independent literal weights (normalised by the WMC).
         *
         * @param pos positive weights
         * @param neg negative weights
         * @return per variable the marginal
         */
        public double[] marginals(double[] pos, double[] neg) {
            double[] dp = new double[variables];
            double[] dn = new double[variables];
            double z = gradient(pos, neg, dp, dn);
            double[] out = new double[variables];
            for (int v = 0; v < variables; v++) {
                out[v] = pos[v] * dp[v] / z;
            }
            return out;
        }

        /** Smooths the circuit (every OR child mentions the same variables) and sets the root. */
        private void finish(int rootNode, long t0) {
            logicalSize = 0;
            List<BitSet> vars = new ArrayList<>();
            for (int[] n : nodes) {
                logicalSize += n[0] == AND || n[0] == OR ? n.length : 1;
                BitSet b = new BitSet();
                switch (n[0]) {
                    case LIT -> b.set(Math.abs(n[1]));
                    case AND, OR -> {
                        for (int k = 1; k < n.length; k++) {
                            b.or(vars.get(n[k]));
                        }
                    }
                    default -> {
                    }
                }
                vars.add(b);
            }
            Map<Integer, Integer> freeNode = new HashMap<>();
            int count = nodes.size();
            int[] remap = new int[count];
            for (int i = 0; i < count; i++) {
                remap[i] = i;
            }
            for (int i = 0; i < count; i++) {
                int[] n = nodes.get(i);
                if (n[0] != OR) {
                    continue;
                }
                int[] smoothed = n.clone();
                for (int k = 1; k < n.length; k++) {
                    BitSet missing = (BitSet) vars.get(i).clone();
                    missing.andNot(vars.get(n[k]));
                    if (!missing.isEmpty()) {
                        smoothed[k] = wrap(n[k], missing, freeNode);
                    }
                }
                // the wrappers were appended after i: move this OR to the end to keep topological order
                int moved = add(smoothed);
                nodes.set(i, new int[] {AND, moved});
                vars.add(vars.get(i));
            }
            BitSet all = new BitSet();
            all.set(1, variables + 1);
            BitSet missing = (BitSet) all.clone();
            missing.andNot(vars.get(rootNode));
            int r = missing.isEmpty() ? rootNode : wrap(rootNode, missing, freeNode);
            // re-establish topological order: ORs replaced by a forward reference are rebuilt
            topologicalRebuild(r);
            millis = (System.nanoTime() - t0) / 1e6;
        }

        private int wrap(int child, BitSet missing, Map<Integer, Integer> freeNode) {
            int[] and = new int[missing.cardinality() + 2];
            and[0] = AND;
            and[1] = child;
            int k = 2;
            for (int v = missing.nextSetBit(0); v >= 0; v = missing.nextSetBit(v + 1)) {
                Integer f = freeNode.get(v);
                if (f == null) {
                    f = add(FREE, v);
                    freeNode.put(v, f);
                }
                and[k++] = f;
            }
            return add(and);
        }

        /** Rewrites the node list in a depth-first post-order from the root. */
        private void topologicalRebuild(int r) {
            List<int[]> out = new ArrayList<>();
            int[] newId = new int[nodes.size()];
            Arrays.fill(newId, -1);
            ArrayList<int[]> stack = new ArrayList<>();
            stack.add(new int[] {r, 0});
            while (!stack.isEmpty()) {
                int[] top = stack.get(stack.size() - 1);
                int[] n = nodes.get(top[0]);
                boolean internal = n[0] == AND || n[0] == OR;
                if (internal && top[1] + 1 < n.length) {
                    int c = n[++top[1]];
                    if (newId[c] < 0) {
                        stack.add(new int[] {c, 0});
                    }
                    continue;
                }
                stack.remove(stack.size() - 1);
                if (newId[top[0]] >= 0) {
                    continue;
                }
                int[] copy = n.clone();
                if (internal) {
                    for (int k = 1; k < copy.length; k++) {
                        copy[k] = newId[copy[k]];
                    }
                }
                out.add(copy);
                newId[top[0]] = out.size() - 1;
            }
            nodes.clear();
            nodes.addAll(out);
            root = newId[r];
        }
    }

    // ---- decision-DNNF ---------------------------------------------------------------------------

    /**
     * Compiles a CNF to decision-DNNF by exhaustive DPLL with components and caching.
     *
     * @param variables the number of variables
     * @param clauses DIMACS clauses
     * @return the circuit
     */
    public static Circuit compileDnnf(int variables, List<int[]> clauses) {
        long t0 = System.nanoTime();
        Circuit c = new Circuit(variables);
        int t = c.add(Circuit.TRUE);
        int f = c.add(Circuit.FALSE);
        Map<String, Integer> cache = new HashMap<>();
        int r = dnnf(c, clauses, cache, t, f);
        c.finish(r, t0);
        return c;
    }

    private static int dnnf(Circuit c, List<int[]> clauses, Map<String, Integer> cache, int t, int f) {
        // unit propagation
        List<Integer> units = new ArrayList<>();
        List<int[]> cur = clauses;
        while (true) {
            int unit = 0;
            for (int[] cl : cur) {
                if (cl.length == 0) {
                    return f;
                }
                if (cl.length == 1) {
                    unit = cl[0];
                    break;
                }
            }
            if (unit == 0) {
                break;
            }
            units.add(unit);
            cur = condition(cur, unit);
            if (cur == null) {
                return f;
            }
        }
        List<Integer> parts = new ArrayList<>();
        for (int u : units) {
            parts.add(c.add(Circuit.LIT, u));
        }
        for (List<int[]> comp : components(cur)) {
            String key = key(comp);
            Integer node = cache.get(key);
            if (node == null) {
                int v = branchVariable(comp);
                List<int[]> pos = condition(comp, v);
                List<int[]> neg = condition(comp, -v);
                int hi = pos == null ? f : dnnf(c, pos, cache, t, f);
                int lo = neg == null ? f : dnnf(c, neg, cache, t, f);
                List<Integer> alts = new ArrayList<>();
                if (hi != f) {
                    alts.add(c.add(Circuit.AND, c.add(Circuit.LIT, v), hi));
                }
                if (lo != f) {
                    alts.add(c.add(Circuit.AND, c.add(Circuit.LIT, -v), lo));
                }
                if (alts.isEmpty()) {
                    node = f;
                } else if (alts.size() == 1) {
                    node = alts.get(0);
                } else {
                    node = c.add(Circuit.OR, alts.get(0), alts.get(1));
                }
                cache.put(key, node);
            }
            if (node == f) {
                return f;
            }
            parts.add(node);
        }
        if (parts.isEmpty()) {
            return t;
        }
        if (parts.size() == 1) {
            return parts.get(0);
        }
        int[] and = new int[parts.size() + 1];
        and[0] = Circuit.AND;
        for (int i = 0; i < parts.size(); i++) {
            and[i + 1] = parts.get(i);
        }
        return c.add(and);
    }

    /** Clauses under {@code lit = true}; null on an empty clause. */
    private static List<int[]> condition(List<int[]> clauses, int lit) {
        List<int[]> out = new ArrayList<>(clauses.size());
        for (int[] cl : clauses) {
            boolean sat = false;
            int keep = 0;
            for (int l : cl) {
                if (l == lit) {
                    sat = true;
                    break;
                }
                if (l != -lit) {
                    keep++;
                }
            }
            if (sat) {
                continue;
            }
            if (keep == 0) {
                return null;
            }
            if (keep == cl.length) {
                out.add(cl);
            } else {
                int[] n = new int[keep];
                int k = 0;
                for (int l : cl) {
                    if (l != -lit) {
                        n[k++] = l;
                    }
                }
                out.add(n);
            }
        }
        return out;
    }

    private static List<List<int[]>> components(List<int[]> clauses) {
        Map<Integer, Integer> parent = new HashMap<>();
        for (int[] cl : clauses) {
            int first = Math.abs(cl[0]);
            parent.putIfAbsent(first, first);
            for (int l : cl) {
                int v = Math.abs(l);
                parent.putIfAbsent(v, v);
                union(parent, first, v);
            }
        }
        Map<Integer, List<int[]>> groups = new java.util.TreeMap<>();
        for (int[] cl : clauses) {
            groups.computeIfAbsent(find(parent, Math.abs(cl[0])), k -> new ArrayList<>()).add(cl);
        }
        return new ArrayList<>(groups.values());
    }

    private static int find(Map<Integer, Integer> p, int x) {
        while (p.get(x) != x) {
            p.put(x, p.get(p.get(x)));
            x = p.get(x);
        }
        return x;
    }

    private static void union(Map<Integer, Integer> p, int a, int b) {
        int ra = find(p, a);
        int rb = find(p, b);
        if (ra != rb) {
            p.put(Math.max(ra, rb), Math.min(ra, rb));
        }
    }

    private static String key(List<int[]> clauses) {
        List<String> parts = new ArrayList<>();
        for (int[] cl : clauses) {
            int[] s = cl.clone();
            Arrays.sort(s);
            parts.add(Arrays.toString(s));
        }
        parts.sort(null);
        return String.join("", parts);
    }

    private static int branchVariable(List<int[]> clauses) {
        Map<Integer, Integer> count = new HashMap<>();
        for (int[] cl : clauses) {
            for (int l : cl) {
                count.merge(Math.abs(l), 1, Integer::sum);
            }
        }
        int best = 0;
        int bc = -1;
        for (Map.Entry<Integer, Integer> e : count.entrySet()) {
            if (e.getValue() > bc || (e.getValue() == bc && e.getKey() < best)) {
                bc = e.getValue();
                best = e.getKey();
            }
        }
        return best;
    }

    // ---- OBDD ----------------------------------------------------------------------------------

    /** A BDD node table with hash-consing and a memoised apply. */
    private static final class Bdd {
        final List<int[]> nodes = new ArrayList<>(); // {var level, low, high}
        final Map<Long, Integer> unique = new HashMap<>();
        final Map<Long, Integer> andMemo = new HashMap<>();
        final Map<Long, Integer> orMemo = new HashMap<>();
        final int levels;

        Bdd(int levels) {
            this.levels = levels;
            nodes.add(new int[] {levels, 0, 0}); // 0 = false
            nodes.add(new int[] {levels, 1, 1}); // 1 = true
        }

        int mk(int level, int lo, int hi) {
            if (lo == hi) {
                return lo;
            }
            long k = ((long) level * 1_000_003L + lo) * 2_000_029L + hi;
            Integer id = unique.get(k);
            if (id != null) {
                int[] n = nodes.get(id);
                if (n[0] == level && n[1] == lo && n[2] == hi) {
                    return id;
                }
            }
            nodes.add(new int[] {level, lo, hi});
            unique.put(k, nodes.size() - 1);
            return nodes.size() - 1;
        }

        int apply(int a, int b, boolean and) {
            if (and) {
                if (a == 0 || b == 0) {
                    return 0;
                }
                if (a == 1) {
                    return b;
                }
                if (b == 1 || a == b) {
                    return a;
                }
            } else {
                if (a == 1 || b == 1) {
                    return 1;
                }
                if (a == 0) {
                    return b;
                }
                if (b == 0 || a == b) {
                    return a;
                }
            }
            long k = (long) Math.min(a, b) * 4_000_037L + Math.max(a, b);
            Map<Long, Integer> memo = and ? andMemo : orMemo;
            Integer m = memo.get(k);
            if (m != null) {
                return m;
            }
            int[] na = nodes.get(a);
            int[] nb = nodes.get(b);
            int lv = Math.min(na[0], nb[0]);
            int alo = na[0] == lv ? na[1] : a;
            int ahi = na[0] == lv ? na[2] : a;
            int blo = nb[0] == lv ? nb[1] : b;
            int bhi = nb[0] == lv ? nb[2] : b;
            int r = mk(lv, apply(alo, blo, and), apply(ahi, bhi, and));
            memo.put(k, r);
            return r;
        }
    }

    /**
     * A min-fill variable order on the primal graph of the clauses.
     *
     * @param variables the number of variables
     * @param clauses the clauses
     * @return the variables (1-based) in elimination order
     */
    public static int[] minFillOrder(int variables, List<int[]> clauses) {
        List<BitSet> adj = new ArrayList<>();
        for (int v = 0; v <= variables; v++) {
            adj.add(new BitSet());
        }
        for (int[] cl : clauses) {
            for (int a : cl) {
                for (int b : cl) {
                    if (Math.abs(a) != Math.abs(b)) {
                        adj.get(Math.abs(a)).set(Math.abs(b));
                    }
                }
            }
        }
        boolean[] done = new boolean[variables + 1];
        int[] order = new int[variables];
        for (int i = 0; i < variables; i++) {
            int best = -1;
            long bestFill = Long.MAX_VALUE;
            for (int v = 1; v <= variables; v++) {
                if (done[v]) {
                    continue;
                }
                long fill = 0;
                int[] nb = adj.get(v).stream().filter(u -> !done[u]).toArray();
                for (int x = 0; x < nb.length; x++) {
                    for (int y = x + 1; y < nb.length; y++) {
                        if (!adj.get(nb[x]).get(nb[y])) {
                            fill++;
                        }
                    }
                }
                if (fill < bestFill) {
                    bestFill = fill;
                    best = v;
                }
            }
            done[best] = true;
            order[i] = best;
            int[] nb = adj.get(best).stream().filter(u -> !done[u]).toArray();
            for (int x : nb) {
                for (int y : nb) {
                    if (x != y) {
                        adj.get(x).set(y);
                    }
                }
            }
        }
        return order;
    }

    /**
     * Compiles a CNF to a reduced OBDD and converts it to a circuit.
     *
     * @param variables the number of variables
     * @param clauses DIMACS clauses
     * @param order the variable order (1-based), or null for index order
     * @return the circuit; its {@link Circuit#size} is the OBDD's node count
     */
    public static Circuit compileObdd(int variables, List<int[]> clauses, int[] order) {
        long t0 = System.nanoTime();
        int[] level = new int[variables + 1];
        int[] varAt = new int[variables];
        for (int i = 0; i < variables; i++) {
            int v = order == null ? i + 1 : order[i];
            level[v] = i;
            varAt[i] = v;
        }
        Bdd b = new Bdd(variables);
        int acc = 1;
        for (int[] cl : clauses) {
            int[] lits = cl.clone();
            Integer[] boxed = Arrays.stream(lits).boxed().toArray(Integer[]::new);
            Arrays.sort(boxed, (x, y) -> Integer.compare(level[Math.abs(y)], level[Math.abs(x)]));
            int c = 0;
            for (int l : boxed) {
                int lv = level[Math.abs(l)];
                c = l > 0 ? b.apply(b.mk(lv, 0, 1), c, false) : b.apply(b.mk(lv, 1, 0), c, false);
            }
            acc = b.apply(acc, c, true);
            if (acc == 0) {
                break;
            }
        }
        // count reachable nodes and convert to a decision circuit
        Circuit circ = new Circuit(variables);
        int t = circ.add(Circuit.TRUE);
        int f = circ.add(Circuit.FALSE);
        Map<Integer, Integer> conv = new HashMap<>();
        conv.put(0, f);
        conv.put(1, t);
        int r = convert(b, acc, varAt, circ, conv);
        circ.finish(r, t0);
        circ.logicalSize = conv.size();
        return circ;
    }

    private static int convert(Bdd b, int node, int[] varAt, Circuit c, Map<Integer, Integer> conv) {
        Integer done = conv.get(node);
        if (done != null) {
            return done;
        }
        int[] n = b.nodes.get(node);
        int v = varAt[n[0]];
        int lo = convert(b, n[1], varAt, c, conv);
        int hi = convert(b, n[2], varAt, c, conv);
        int f = conv.get(0);
        int out;
        if (lo == f) {
            out = c.add(Circuit.AND, c.add(Circuit.LIT, v), hi);
        } else if (hi == f) {
            out = c.add(Circuit.AND, c.add(Circuit.LIT, -v), lo);
        } else {
            out = c.add(Circuit.OR, c.add(Circuit.AND, c.add(Circuit.LIT, v), hi),
                c.add(Circuit.AND, c.add(Circuit.LIT, -v), lo));
        }
        conv.put(node, out);
        return out;
    }

    // ---- Bayesian networks as weighted CNF -----------------------------------------------------------

    /**
     * A Bayesian network as weighted CNF.
     *
     * @param variables CNF variables
     * @param clauses the clauses
     * @param pos weight of every positive literal
     * @param neg weight of every negative literal
     * @param indicator per network variable and value, the indicator's CNF variable
     */
    public record WeightedCnf(int variables, List<int[]> clauses, double[] pos, double[] neg, int[][] indicator) {
    }

    /**
     * Encodes a Bayesian network (Chavira &amp; Darwiche's encoding with determinism).
     *
     * @param cardinality values per network variable
     * @param parents parents per variable
     * @param cpt CPTs, row-major with the first parent slowest and the child fastest
     * @return the weighted CNF
     */
    public static WeightedCnf encodeBayesNet(int[] cardinality, int[][] parents, double[][] cpt) {
        int n = cardinality.length;
        int[][] ind = new int[n][];
        int next = 1;
        List<int[]> cs = new ArrayList<>();
        for (int v = 0; v < n; v++) {
            ind[v] = new int[cardinality[v]];
            for (int x = 0; x < cardinality[v]; x++) {
                ind[v][x] = next++;
            }
            cs.add(ind[v].clone());
            for (int a = 0; a < cardinality[v]; a++) {
                for (int c = a + 1; c < cardinality[v]; c++) {
                    cs.add(new int[] {-ind[v][a], -ind[v][c]});
                }
            }
        }
        List<double[]> paramWeights = new ArrayList<>();
        for (int v = 0; v < n; v++) {
            int rows = cpt[v].length / cardinality[v];
            for (int row = 0; row < rows; row++) {
                int[] cfg = new int[parents[v].length];
                int rem = row;
                for (int k = parents[v].length - 1; k >= 0; k--) {
                    cfg[k] = rem % cardinality[parents[v][k]];
                    rem /= cardinality[parents[v][k]];
                }
                for (int x = 0; x < cardinality[v]; x++) {
                    double p = cpt[v][row * cardinality[v] + x];
                    int[] body = new int[cfg.length + 1];
                    for (int k = 0; k < cfg.length; k++) {
                        body[k] = ind[parents[v][k]][cfg[k]];
                    }
                    body[cfg.length] = ind[v][x];
                    if (p == 0) {
                        int[] cl = new int[body.length];
                        for (int k = 0; k < body.length; k++) {
                            cl[k] = -body[k];
                        }
                        cs.add(cl);
                    } else if (p < 1) {
                        int theta = next++;
                        paramWeights.add(new double[] {theta, p});
                        int[] cl = new int[body.length + 1];
                        for (int k = 0; k < body.length; k++) {
                            cl[k] = -body[k];
                            cs.add(new int[] {-theta, body[k]});
                        }
                        cl[body.length] = theta;
                        cs.add(cl);
                    }
                }
            }
        }
        int vars = next - 1;
        double[] pos = new double[vars];
        double[] neg = new double[vars];
        Arrays.fill(pos, 1);
        Arrays.fill(neg, 1);
        for (double[] pw : paramWeights) {
            pos[(int) pw[0] - 1] = pw[1];
        }
        return new WeightedCnf(vars, cs, pos, neg, ind);
    }

    /**
     * Posterior marginals of every network variable given evidence, from one WMC gradient pass.
     *
     * @param enc the encoding
     * @param circuit the compiled encoding
     * @param evidence per network variable the observed value or -1
     * @return per variable and value the posterior probability
     */
    public static double[][] marginals(WeightedCnf enc, Circuit circuit, int[] evidence) {
        double[] pos = enc.pos().clone();
        double[] neg = enc.neg().clone();
        for (int v = 0; v < evidence.length; v++) {
            if (evidence[v] >= 0) {
                for (int x = 0; x < enc.indicator()[v].length; x++) {
                    if (x != evidence[v]) {
                        pos[enc.indicator()[v][x] - 1] = 0;
                    }
                }
            }
        }
        double[] dp = new double[pos.length];
        double[] dn = new double[pos.length];
        double z = circuit.gradient(pos, neg, dp, dn);
        double[][] out = new double[evidence.length][];
        for (int v = 0; v < evidence.length; v++) {
            out[v] = new double[enc.indicator()[v].length];
            for (int x = 0; x < out[v].length; x++) {
                int i = enc.indicator()[v][x] - 1;
                out[v][x] = pos[i] * dp[i] / z;
            }
        }
        return out;
    }
}
