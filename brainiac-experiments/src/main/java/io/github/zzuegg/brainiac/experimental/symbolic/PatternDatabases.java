package io.github.zzuegg.brainiac.experimental.symbolic;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.SplittableRandom;
import java.util.function.ToIntFunction;

/**
 * Pattern database heuristics (Culberson &amp; Schaeffer, Comput. Intell. 1998; Edelkamp,
 * "Planning with pattern databases", ECP 2001; Haslum et al., "Domain-independent construction
 * of pattern database heuristics for cost-optimal planning", AAAI 2007).
 *
 * <p>A pattern database projects a task onto a subset of its multi-valued variables (the
 * pattern), solves the projected (abstract) state space exactly by a backward Dijkstra from the
 * abstract goal states, and stores every abstract goal distance; looking up the projection of a
 * state gives an admissible and consistent heuristic. The multi-valued view ({@link SasView}) is
 * built from the task's mutex groups: every group becomes a variable whose values are its facts
 * plus "none of them".
 *
 * <ul>
 *   <li>{@link #build}: one PDB.</li>
 *   <li>{@link #canonical}: the canonical heuristic of a pattern collection: two patterns are
 *       additive when no action changes variables of both, and the estimate is the maximum,
 *       over the maximal cliques of additive patterns, of the clique's sum.</li>
 *   <li>{@link #ipdb}: iPDB hill climbing: starting from one pattern per goal variable, add the
 *       one-variable extension (a causally relevant variable) that raises the canonical
 *       heuristic on the most sampled states, until no extension helps enough.</li>
 *   <li>{@link #slidingTileAdditive}: disjoint additive PDBs for the sliding-tile puzzle (Korf
 *       &amp; Felner, AIJ 2002): only moves of a pattern's tiles cost, so the PDBs of disjoint tile
 *       sets add up admissibly.</li>
 * </ul>
 */
public final class PatternDatabases {

    private PatternDatabases() {
    }

    /** The multi-valued (SAS+) view of a STRIPS task built from its mutex groups. */
    public static final class SasView {

        private final StripsTask task;
        private final int[] domain;
        private final int[] factVar;
        private final int[] factValue;
        private final int[][] varFacts;
        private final int[][] preVar;
        private final int[][] preVal;
        private final int[][] effVar;
        private final int[][] effVal;
        private final int[] goalVal;

        /**
         * Builds the view: declared groups in order, each keeping its facts not covered by an
         * earlier group; every remaining fact becomes a binary variable.
         *
         * @param task the task
         */
        public SasView(StripsTask task) {
            this.task = task;
            int nf = task.facts();
            factVar = new int[nf];
            factValue = new int[nf];
            Arrays.fill(factVar, -1);
            List<int[]> vars = new ArrayList<>();
            for (int[] g : task.mutexGroups()) {
                int[] left = Arrays.stream(g).filter(f -> factVar[f] < 0).distinct().toArray();
                if (left.length == 0) {
                    continue;
                }
                for (int i = 0; i < left.length; i++) {
                    factVar[left[i]] = vars.size();
                    factValue[left[i]] = i;
                }
                vars.add(left);
            }
            for (int f = 0; f < nf; f++) {
                if (factVar[f] < 0) {
                    factVar[f] = vars.size();
                    factValue[f] = 0;
                    vars.add(new int[] {f});
                }
            }
            varFacts = vars.toArray(new int[0][]);
            domain = new int[varFacts.length];
            for (int v = 0; v < domain.length; v++) {
                domain[v] = varFacts[v].length + 1;
            }
            int na = task.actions().size();
            preVar = new int[na][];
            preVal = new int[na][];
            effVar = new int[na][];
            effVal = new int[na][];
            for (StripsTask.Action a : task.actions()) {
                Map<Integer, Integer> pre = new HashMap<>();
                for (int p : a.pre()) {
                    pre.put(factVar[p], factValue[p]);
                }
                Map<Integer, Integer> eff = new HashMap<>();
                for (int d : a.del()) {
                    // a delete of a precondition empties its variable unless an add refills it
                    if (pre.containsKey(factVar[d]) && pre.get(factVar[d]) == factValue[d]) {
                        eff.put(factVar[d], varFacts[factVar[d]].length);
                    }
                }
                for (int ad : a.add()) {
                    eff.put(factVar[ad], factValue[ad]);
                }
                preVar[a.id()] = pre.keySet().stream().mapToInt(Integer::intValue).toArray();
                preVal[a.id()] = Arrays.stream(preVar[a.id()]).map(pre::get).toArray();
                effVar[a.id()] = eff.keySet().stream().mapToInt(Integer::intValue).toArray();
                effVal[a.id()] = Arrays.stream(effVar[a.id()]).map(eff::get).toArray();
            }
            goalVal = new int[varFacts.length];
            Arrays.fill(goalVal, -1);
            for (int g : task.goal()) {
                goalVal[factVar[g]] = factValue[g];
            }
        }

        /** @return the number of variables */
        public int variables() {
            return domain.length;
        }

        /**
         * @param v a variable
         * @return its domain size (including "none")
         */
        public int domain(int v) {
            return domain[v];
        }

        /**
         * @param state a STRIPS state
         * @return the value of every variable
         */
        public int[] project(BitSet state) {
            int[] s = new int[domain.length];
            for (int v = 0; v < domain.length; v++) {
                s[v] = varFacts[v].length;
                for (int i = 0; i < varFacts[v].length; i++) {
                    if (state.get(varFacts[v][i])) {
                        s[v] = i;
                        break;
                    }
                }
            }
            return s;
        }

        /** @return the variables that have a goal value */
        public int[] goalVariables() {
            List<Integer> out = new ArrayList<>();
            for (int v = 0; v < goalVal.length; v++) {
                if (goalVal[v] >= 0) {
                    out.add(v);
                }
            }
            return out.stream().mapToInt(Integer::intValue).toArray();
        }

        /** @return the task */
        public StripsTask task() {
            return task;
        }
    }

    /** One pattern database. */
    public static final class Pdb {

        private final SasView view;
        private final int[] pattern;
        private final long[] radix;
        private final double[] distance;
        private final double millis;

        private Pdb(SasView view, int[] pattern, long[] radix, double[] distance, double millis) {
            this.view = view;
            this.pattern = pattern;
            this.radix = radix;
            this.distance = distance;
            this.millis = millis;
        }

        /** @return the pattern (sorted variables) */
        public int[] pattern() {
            return pattern.clone();
        }

        /** @return the number of abstract states (the table size) */
        public int size() {
            return distance.length;
        }

        /** @return build time in milliseconds */
        public double millis() {
            return millis;
        }

        /**
         * @param values a full SAS+ state
         * @return the abstract goal distance
         */
        public double lookup(int[] values) {
            long idx = 0;
            for (int i = 0; i < pattern.length; i++) {
                idx += radix[i] * values[pattern[i]];
            }
            return distance[(int) idx];
        }

        /**
         * @param state a STRIPS state
         * @return the abstract goal distance
         */
        public double lookup(BitSet state) {
            return lookup(view.project(state));
        }

        boolean affects(int action) {
            for (int v : view.effVar[action]) {
                if (Arrays.binarySearch(pattern, v) >= 0) {
                    return true;
                }
            }
            return false;
        }
    }

    /**
     * Builds a pattern database.
     *
     * @param view the SAS+ view
     * @param pattern the variables
     * @param maxSize the largest table allowed
     * @return the PDB, or null if the abstract space is larger than {@code maxSize}
     */
    public static Pdb build(SasView view, int[] pattern, int maxSize) {
        long t0 = System.nanoTime();
        int[] p = Arrays.stream(pattern).distinct().sorted().toArray();
        long size = 1;
        long[] radix = new long[p.length];
        for (int i = 0; i < p.length; i++) {
            radix[i] = size;
            size *= view.domain[p[i]];
            if (size > maxSize) {
                return null;
            }
        }
        int n = (int) size;
        int[] local = new int[view.domain.length];
        Arrays.fill(local, -1);
        for (int i = 0; i < p.length; i++) {
            local[p[i]] = i;
        }
        // abstract operators: projected preconditions and effects
        List<int[][]> ops = new ArrayList<>();
        List<Double> cost = new ArrayList<>();
        for (StripsTask.Action a : view.task.actions()) {
            List<int[]> pre = new ArrayList<>();
            List<int[]> eff = new ArrayList<>();
            for (int i = 0; i < view.preVar[a.id()].length; i++) {
                int l = local[view.preVar[a.id()][i]];
                if (l >= 0) {
                    pre.add(new int[] {l, view.preVal[a.id()][i]});
                }
            }
            for (int i = 0; i < view.effVar[a.id()].length; i++) {
                int l = local[view.effVar[a.id()][i]];
                if (l >= 0) {
                    eff.add(new int[] {l, view.effVal[a.id()][i]});
                }
            }
            if (eff.isEmpty()) {
                continue;
            }
            ops.add(new int[][] {pre.stream().flatMapToInt(Arrays::stream).toArray(),
                eff.stream().flatMapToInt(Arrays::stream).toArray()});
            cost.add(a.cost());
        }
        // reverse edges
        List<List<long[]>> rev = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            rev.add(null);
        }
        int[] vals = new int[p.length];
        for (int s = 0; s < n; s++) {
            int rem = s;
            for (int i = 0; i < p.length; i++) {
                vals[i] = rem % view.domain[p[i]];
                rem /= view.domain[p[i]];
            }
            for (int o = 0; o < ops.size(); o++) {
                int[] pre = ops.get(o)[0];
                boolean ok = true;
                for (int k = 0; k < pre.length; k += 2) {
                    if (vals[pre[k]] != pre[k + 1]) {
                        ok = false;
                        break;
                    }
                }
                if (!ok) {
                    continue;
                }
                int[] eff = ops.get(o)[1];
                long t = s;
                for (int k = 0; k < eff.length; k += 2) {
                    t += (eff[k + 1] - vals[eff[k]]) * radix[eff[k]];
                }
                if (t == s) {
                    continue;
                }
                List<long[]> r = rev.get((int) t);
                if (r == null) {
                    r = new ArrayList<>(2);
                    rev.set((int) t, r);
                }
                r.add(new long[] {s, Double.doubleToLongBits(cost.get(o))});
            }
        }
        double[] dist = new double[n];
        Arrays.fill(dist, Double.POSITIVE_INFINITY);
        PriorityQueue<double[]> pq = new PriorityQueue<>((a, b) -> Double.compare(a[0], b[0]));
        for (int s = 0; s < n; s++) {
            int rem = s;
            boolean goal = true;
            for (int i = 0; i < p.length; i++) {
                int v = rem % view.domain[p[i]];
                rem /= view.domain[p[i]];
                if (view.goalVal[p[i]] >= 0 && view.goalVal[p[i]] != v) {
                    goal = false;
                }
            }
            if (goal) {
                dist[s] = 0;
                pq.add(new double[] {0, s});
            }
        }
        while (!pq.isEmpty()) {
            double[] e = pq.poll();
            int s = (int) e[1];
            if (e[0] > dist[s]) {
                continue;
            }
            List<long[]> r = rev.get(s);
            if (r == null) {
                continue;
            }
            for (long[] edge : r) {
                double nd = dist[s] + Double.longBitsToDouble(edge[1]);
                int from = (int) edge[0];
                if (nd < dist[from]) {
                    dist[from] = nd;
                    pq.add(new double[] {nd, from});
                }
            }
        }
        return new Pdb(view, p, radix, dist, (System.nanoTime() - t0) / 1e6);
    }

    /**
     * A single PDB as an evaluator.
     *
     * @param pdb the PDB
     * @return the evaluator
     */
    public static StateSpaceSearch.Evaluator evaluator(Pdb pdb) {
        return (s, d) -> StateSpaceSearch.Evaluation.of(pdb.lookup(s));
    }

    /** The canonical heuristic of a pattern collection. */
    public static final class Canonical implements StateSpaceSearch.Evaluator {

        private final SasView view;
        private final List<Pdb> pdbs;
        private final List<int[]> cliques;

        Canonical(SasView view, List<Pdb> pdbs) {
            this.view = view;
            this.pdbs = List.copyOf(pdbs);
            int k = pdbs.size();
            boolean[][] additive = new boolean[k][k];
            for (int i = 0; i < k; i++) {
                for (int j = i + 1; j < k; j++) {
                    boolean ok = true;
                    for (StripsTask.Action a : view.task.actions()) {
                        if (pdbs.get(i).affects(a.id()) && pdbs.get(j).affects(a.id())) {
                            ok = false;
                            break;
                        }
                    }
                    additive[i][j] = ok;
                    additive[j][i] = ok;
                }
            }
            this.cliques = new ArrayList<>();
            bronKerbosch(new ArrayList<>(), allIndices(k), new ArrayList<>(), additive, cliques);
        }

        private static List<Integer> allIndices(int k) {
            List<Integer> out = new ArrayList<>();
            for (int i = 0; i < k; i++) {
                out.add(i);
            }
            return out;
        }

        private static void bronKerbosch(List<Integer> r, List<Integer> p, List<Integer> x, boolean[][] adj,
                List<int[]> out) {
            if (p.isEmpty() && x.isEmpty()) {
                out.add(r.stream().mapToInt(Integer::intValue).toArray());
                return;
            }
            for (Integer v : new ArrayList<>(p)) {
                List<Integer> r2 = new ArrayList<>(r);
                r2.add(v);
                List<Integer> p2 = new ArrayList<>();
                List<Integer> x2 = new ArrayList<>();
                for (int u : p) {
                    if (adj[v][u]) {
                        p2.add(u);
                    }
                }
                for (int u : x) {
                    if (adj[v][u]) {
                        x2.add(u);
                    }
                }
                bronKerbosch(r2, p2, x2, adj, out);
                p.remove(v);
                x.add(v);
            }
        }

        /** @return the PDBs */
        public List<Pdb> pdbs() {
            return pdbs;
        }

        /** @return the total number of table entries */
        public long totalSize() {
            long s = 0;
            for (Pdb p : pdbs) {
                s += p.size();
            }
            return s;
        }

        /** @return the total build time in milliseconds */
        public double totalMillis() {
            double s = 0;
            for (Pdb p : pdbs) {
                s += p.millis();
            }
            return s;
        }

        /**
         * @param values a full SAS+ state
         * @return the canonical estimate
         */
        public double value(int[] values) {
            double[] h = new double[pdbs.size()];
            for (int i = 0; i < h.length; i++) {
                h[i] = pdbs.get(i).lookup(values);
            }
            double best = 0;
            for (int[] c : cliques) {
                double s = 0;
                for (int i : c) {
                    s += h[i];
                }
                best = Math.max(best, s);
            }
            return best;
        }

        @Override
        public StateSpaceSearch.Evaluation evaluate(BitSet state, Object data) {
            return StateSpaceSearch.Evaluation.of(value(view.project(state)));
        }
    }

    /**
     * The canonical heuristic of a pattern collection.
     *
     * @param view the SAS+ view
     * @param patterns the patterns
     * @param maxSize the largest PDB allowed (larger ones are skipped)
     * @return the heuristic
     */
    public static Canonical canonical(SasView view, List<int[]> patterns, int maxSize) {
        List<Pdb> pdbs = new ArrayList<>();
        for (int[] p : patterns) {
            Pdb pdb = build(view, p, maxSize);
            if (pdb != null) {
                pdbs.add(pdb);
            }
        }
        return new Canonical(view, pdbs);
    }

    /**
     * iPDB hill climbing in pattern space.
     *
     * @param view the SAS+ view
     * @param maxPdbSize the largest PDB
     * @param maxCollectionSize the largest total number of entries
     * @param samples sampled states per step
     * @param minImprovement the fewest improved samples to accept a step
     * @param maxSteps the most hill-climbing steps
     * @param seed the sampling seed
     * @return the canonical heuristic of the final collection
     */
    public static Canonical ipdb(SasView view, int maxPdbSize, long maxCollectionSize, int samples,
            int minImprovement, int maxSteps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Map<String, Pdb> cache = new HashMap<>();
        List<Pdb> collection = new ArrayList<>();
        for (int v : view.goalVariables()) {
            Pdb p = cached(view, new int[] {v}, maxPdbSize, cache);
            if (p != null) {
                collection.add(p);
            }
        }
        Canonical current = new Canonical(view, collection);
        for (int step = 0; step < maxSteps; step++) {
            double h0 = current.value(view.project(view.task.initialState()));
            List<int[]> sample = new ArrayList<>();
            for (int i = 0; i < samples; i++) {
                sample.add(view.project(randomWalk(view.task, (int) Math.max(1, 2 * h0), r)));
            }
            double[] base = new double[sample.size()];
            for (int i = 0; i < base.length; i++) {
                base[i] = current.value(sample.get(i));
            }
            Pdb bestCand = null;
            int bestCount = -1;
            for (Pdb pdb : collection) {
                for (int v : relevantExtensions(view, pdb.pattern)) {
                    int[] np = Arrays.copyOf(pdb.pattern, pdb.pattern.length + 1);
                    np[pdb.pattern.length] = v;
                    Pdb cand = cached(view, np, maxPdbSize, cache);
                    if (cand == null || current.totalSize() + cand.size() > maxCollectionSize) {
                        continue;
                    }
                    boolean present = false;
                    for (Pdb q : collection) {
                        if (Arrays.equals(q.pattern, cand.pattern)) {
                            present = true;
                        }
                    }
                    if (present) {
                        continue;
                    }
                    List<Pdb> trial = new ArrayList<>(collection);
                    trial.add(cand);
                    Canonical t = new Canonical(view, trial);
                    int count = 0;
                    for (int i = 0; i < base.length; i++) {
                        if (t.value(sample.get(i)) > base[i] + 1e-9) {
                            count++;
                        }
                    }
                    if (count > bestCount) {
                        bestCount = count;
                        bestCand = cand;
                    }
                }
            }
            if (bestCand == null || bestCount < minImprovement) {
                break;
            }
            collection.add(bestCand);
            current = new Canonical(view, collection);
        }
        return current;
    }

    private static Pdb cached(SasView view, int[] pattern, int maxSize, Map<String, Pdb> cache) {
        int[] p = Arrays.stream(pattern).distinct().sorted().toArray();
        String key = Arrays.toString(p);
        if (cache.containsKey(key)) {
            return cache.get(key);
        }
        Pdb pdb = build(view, p, maxSize);
        cache.put(key, pdb);
        return pdb;
    }

    private static int[] relevantExtensions(SasView view, int[] pattern) {
        BitSet out = new BitSet();
        for (StripsTask.Action a : view.task.actions()) {
            boolean affects = false;
            for (int v : view.effVar[a.id()]) {
                if (Arrays.binarySearch(pattern, v) >= 0) {
                    affects = true;
                    break;
                }
            }
            if (!affects) {
                continue;
            }
            for (int v : view.preVar[a.id()]) {
                out.set(v);
            }
            for (int v : view.effVar[a.id()]) {
                out.set(v);
            }
        }
        for (int v : pattern) {
            out.clear(v);
        }
        return out.stream().toArray();
    }

    private static BitSet randomWalk(StripsTask task, int maxLength, SplittableRandom r) {
        BitSet s = task.initialState();
        int len = r.nextInt(maxLength + 1);
        for (int i = 0; i < len; i++) {
            List<StripsTask.Action> app = task.applicable(s);
            if (app.isEmpty()) {
                break;
            }
            s = app.get(r.nextInt(app.size())).apply(s);
        }
        return s;
    }

    // ---- sliding-tile additive PDBs -------------------------------------------------------------

    /**
     * Disjoint additive pattern databases for the sliding-tile puzzle (board layout of
     * {@code PlanningMaps}: tile {@code t} belongs at cell {@code t-1}, blank last).
     *
     * @param side the board side
     * @param partition disjoint tile sets
     * @return the heuristic: the sum of every set's PDB value
     */
    public static ToIntFunction<int[]> slidingTileAdditive(int side, int[][] partition) {
        int cells = side * side;
        List<byte[]> tables = new ArrayList<>();
        for (int[] tiles : partition) {
            int k = tiles.length;
            int size = (int) Math.pow(cells, k);
            byte[] d = new byte[size];
            Arrays.fill(d, (byte) -1);
            int[] goalPos = new int[k];
            for (int i = 0; i < k; i++) {
                goalPos[i] = tiles[i] - 1;
            }
            int g = encode(goalPos, cells);
            d[g] = 0;
            ArrayDeque<Integer> q = new ArrayDeque<>();
            q.add(g);
            int[] pos = new int[k];
            while (!q.isEmpty()) {
                int s = q.poll();
                decode(s, cells, pos);
                for (int i = 0; i < k; i++) {
                    int c = pos[i];
                    int x = c % side;
                    int y = c / side;
                    int[][] dir = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
                    for (int[] m : dir) {
                        int nx = x + m[0];
                        int ny = y + m[1];
                        if (nx < 0 || ny < 0 || nx >= side || ny >= side) {
                            continue;
                        }
                        int nc = ny * side + nx;
                        boolean occupied = false;
                        for (int j = 0; j < k; j++) {
                            if (pos[j] == nc) {
                                occupied = true;
                                break;
                            }
                        }
                        if (occupied) {
                            continue;
                        }
                        pos[i] = nc;
                        int t = encode(pos, cells);
                        pos[i] = c;
                        if (d[t] < 0) {
                            d[t] = (byte) (d[s] + 1);
                            q.add(t);
                        }
                    }
                }
            }
            tables.add(d);
        }
        int[][] part = partition.clone();
        return board -> {
            int[] where = new int[cells];
            for (int c = 0; c < cells; c++) {
                where[board[c]] = c;
            }
            int sum = 0;
            int[] pos = new int[0];
            for (int p = 0; p < part.length; p++) {
                if (pos.length != part[p].length) {
                    pos = new int[part[p].length];
                }
                for (int i = 0; i < part[p].length; i++) {
                    pos[i] = where[part[p][i]];
                }
                sum += tables.get(p)[encode(pos, cells)];
            }
            return sum;
        };
    }

    private static int encode(int[] pos, int cells) {
        int s = 0;
        for (int i = pos.length - 1; i >= 0; i--) {
            s = s * cells + pos[i];
        }
        return s;
    }

    private static void decode(int s, int cells, int[] pos) {
        for (int i = 0; i < pos.length; i++) {
            pos[i] = s % cells;
            s /= cells;
        }
    }
}
