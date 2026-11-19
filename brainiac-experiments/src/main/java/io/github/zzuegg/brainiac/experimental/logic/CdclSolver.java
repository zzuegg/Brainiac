package io.github.zzuegg.brainiac.experimental.logic;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * A compact MiniSat-style conflict-driven clause-learning SAT solver (Marques-Silva &amp; Sakallah,
 * "GRASP", IEEE TC 1999; Moskewicz et al., "Chaff", DAC 2001; Eén &amp; Sörensson, "An extensible
 * SAT-solver", SAT 2003; Audemard &amp; Simon, "Glucose", IJCAI 2009).
 *
 * <ul>
 *   <li>Unit propagation with two watched literals per clause.</li>
 *   <li>Conflict analysis to the first unique implication point (1-UIP), non-chronological
 *       backjumping to the second-highest level of the learnt clause.</li>
 *   <li>VSIDS branching: variables in recent conflicts are bumped, all activities decay
 *       geometrically; a binary heap picks the most active unassigned variable.</li>
 *   <li>Luby or glucose-style (LBD moving average) restarts, phase saving, and learnt-clause
 *       deletion that keeps low-LBD ("glue") clauses.</li>
 *   <li>Hooks for external guidance (used by {@link RlafGuidance}): a positive weight per
 *       variable multiplies its VSIDS score (initial scores are 1, so the weights alone order the
 *       first decisions) and an initial polarity per variable sets the first value tried.</li>
 * </ul>
 *
 * Literals use the DIMACS convention ({@code +v} / {@code -v}, variables from 1). The solver is
 * incremental: clauses can be added between {@link #solve} calls (learnt clauses are kept),
 * which {@link AnswerSetSolver} uses for lazily added loop formulas and model enumeration.
 * Every run's costs (decisions, propagations, conflicts) are counted for the learned-guidance
 * and planning experiments.
 */
public final class CdclSolver {

    /** The result of {@link #solve}. */
    public enum Status {
        /** A model was found. */
        SAT,
        /** The clauses are unsatisfiable. */
        UNSAT,
        /** The conflict budget ran out. */
        UNKNOWN
    }

    /** The restart policy. */
    public enum Restarts {
        /** Never restart. */
        NONE,
        /** Luby sequence times a unit number of conflicts. */
        LUBY,
        /** Glucose: restart when the recent LBD average exceeds the global one. */
        GLUCOSE
    }

    /**
     * Solver options; every feature can be switched off for ablations.
     *
     * @param vsids whether to bump and decay activities (off: a static order by weight, then
     *     index)
     * @param restarts the restart policy
     * @param phaseSaving whether to re-use a variable's last value
     * @param deletion whether to delete high-LBD learnt clauses
     * @param varDecay the VSIDS decay factor
     * @param lubyUnit conflicts per Luby unit
     * @param conflictLimit conflicts before giving up (UNKNOWN)
     */
    public record Options(boolean vsids, Restarts restarts, boolean phaseSaving, boolean deletion, double varDecay,
            int lubyUnit, long conflictLimit) {

        /** @return the default configuration (all features on, Luby restarts) */
        public static Options defaults() {
            return new Options(true, Restarts.LUBY, true, true, 0.95, 100, Long.MAX_VALUE);
        }

        /**
         * @param limit the conflict budget
         * @return a copy with that budget
         */
        public Options withConflictLimit(long limit) {
            return new Options(vsids, restarts, phaseSaving, deletion, varDecay, lubyUnit, limit);
        }

        /**
         * @param on VSIDS on or off
         * @return a copy
         */
        public Options withVsids(boolean on) {
            return new Options(on, restarts, phaseSaving, deletion, varDecay, lubyUnit, conflictLimit);
        }

        /**
         * @param policy the restart policy
         * @return a copy
         */
        public Options withRestarts(Restarts policy) {
            return new Options(vsids, policy, phaseSaving, deletion, varDecay, lubyUnit, conflictLimit);
        }

        /**
         * @param on phase saving on or off
         * @return a copy
         */
        public Options withPhaseSaving(boolean on) {
            return new Options(vsids, restarts, on, deletion, varDecay, lubyUnit, conflictLimit);
        }

        /**
         * @param on clause deletion on or off
         * @return a copy
         */
        public Options withDeletion(boolean on) {
            return new Options(vsids, restarts, phaseSaving, on, varDecay, lubyUnit, conflictLimit);
        }
    }

    /**
     * Run costs, cumulative over all {@link #solve} calls.
     *
     * @param decisions branching decisions
     * @param propagations literals propagated
     * @param conflicts conflicts
     * @param restarts restarts
     * @param learnt learnt clauses added
     * @param deleted learnt clauses deleted
     */
    public record Stats(long decisions, long propagations, long conflicts, long restarts, long learnt, long deleted) {
    }

    private static final class Clause {
        final int[] lits;
        final boolean learnt;
        int lbd;
        double activity;
        boolean deleted;

        Clause(int[] lits, boolean learnt) {
            this.lits = lits;
            this.learnt = learnt;
        }
    }

    private final Options options;
    private int vars;
    private byte[] assign = new byte[16];
    private int[] level = new int[16];
    private Clause[] reason = new Clause[16];
    private double[] activity = new double[16];
    private double[] weight = new double[16];
    private boolean[] phase = new boolean[16];
    private boolean[] seen = new boolean[16];
    private int[] heap = new int[16];
    private int heapSize;
    private int[] heapIndex = new int[16];
    private List<List<Clause>> watches = new ArrayList<>();
    private int[] trail = new int[16];
    private int trailSize;
    private int qhead;
    private final List<Integer> trailLim = new ArrayList<>();
    private final List<Clause> clauses = new ArrayList<>();
    private final List<Clause> learnts = new ArrayList<>();
    private double varInc = 1;
    private double clauseInc = 1;
    private boolean ok = true;
    private boolean[] model;
    private long decisions;
    private long propagations;
    private long conflicts;
    private long restartCount;
    private long learntCount;
    private long deletedCount;
    private double maxLearnts;
    private final double[] lbdQueue = new double[50];
    private int lbdQueueSize;
    private int lbdQueueHead;
    private double lbdSum;
    private long lbdTotal;

    /**
     * Creates a solver.
     *
     * @param variables the initial number of variables
     * @param options the options
     */
    public CdclSolver(int variables, Options options) {
        this.options = options;
        for (int i = 0; i < variables; i++) {
            addVariable();
        }
    }

    /**
     * Creates a solver with default options and loads clauses.
     *
     * @param variables the number of variables
     * @param clauses DIMACS clauses
     * @param options the options
     * @return the solver
     */
    public static CdclSolver of(int variables, List<int[]> clauses, Options options) {
        CdclSolver s = new CdclSolver(variables, options);
        for (int[] c : clauses) {
            s.addClause(c);
        }
        return s;
    }

    /** @return the number of variables */
    public int variables() {
        return vars;
    }

    /** @return a new variable (DIMACS index) */
    public int newVariable() {
        addVariable();
        return vars;
    }

    private void addVariable() {
        int v = vars++;
        if (v >= assign.length) {
            int n = assign.length * 2;
            assign = Arrays.copyOf(assign, n);
            level = Arrays.copyOf(level, n);
            reason = Arrays.copyOf(reason, n);
            activity = Arrays.copyOf(activity, n);
            weight = Arrays.copyOf(weight, n);
            phase = Arrays.copyOf(phase, n);
            seen = Arrays.copyOf(seen, n);
            heap = Arrays.copyOf(heap, n);
            heapIndex = Arrays.copyOf(heapIndex, n);
            trail = Arrays.copyOf(trail, n);
        }
        activity[v] = 1;
        weight[v] = 1;
        heapIndex[v] = -1;
        watches.add(new ArrayList<>());
        watches.add(new ArrayList<>());
        heapInsert(v);
    }

    /**
     * Sets the per-variable branching weights (the score is weight times activity).
     *
     * @param weights one positive weight per variable, index 0 for variable 1
     */
    public void setWeights(double[] weights) {
        for (int v = 0; v < Math.min(vars, weights.length); v++) {
            weight[v] = Math.max(1e-12, weights[v]);
        }
        rebuildHeap();
    }

    /**
     * Sets the initial polarities (the value tried first, before phase saving takes over).
     *
     * @param positive one flag per variable, index 0 for variable 1
     */
    public void setPolarities(boolean[] positive) {
        for (int v = 0; v < Math.min(vars, positive.length); v++) {
            phase[v] = positive[v];
        }
    }

    /** @return the cumulative run costs */
    public Stats stats() {
        return new Stats(decisions, propagations, conflicts, restartCount, learntCount, deletedCount);
    }

    /** @return the model of the last SAT answer, index 0 for variable 1 */
    public boolean[] model() {
        return model == null ? null : model.clone();
    }

    /**
     * @param variable a DIMACS variable
     * @return its value in the last model
     */
    public boolean value(int variable) {
        return model[variable - 1];
    }

    // ---- literals ---------------------------------------------------------------------------

    private static int lit(int dimacs) {
        return dimacs > 0 ? 2 * (dimacs - 1) : 2 * (-dimacs - 1) + 1;
    }

    private int litValue(int l) {
        byte a = assign[l >> 1];
        if (a == 0) {
            return 0;
        }
        return ((a == 1) == ((l & 1) == 0)) ? 1 : -1;
    }

    private int decisionLevel() {
        return trailLim.size();
    }

    // ---- clauses ----------------------------------------------------------------------------

    /**
     * Adds a clause (backtracking to level 0 first).
     *
     * @param dimacs the literals
     * @return false if the clause set became trivially unsatisfiable
     */
    public boolean addClause(int... dimacs) {
        if (!ok) {
            return false;
        }
        cancelUntil(0);
        int[] ls = new int[dimacs.length];
        int n = 0;
        outer:
        for (int d : dimacs) {
            while (Math.abs(d) > vars) {
                addVariable();
            }
            int l = lit(d);
            for (int i = 0; i < n; i++) {
                if (ls[i] == l) {
                    continue outer;
                }
                if (ls[i] == (l ^ 1)) {
                    return true; // tautology
                }
            }
            int val = litValue(l);
            if (val == 1) {
                return true;
            }
            if (val == -1) {
                continue;
            }
            ls[n++] = l;
        }
        if (n == 0) {
            ok = false;
            return false;
        }
        if (n == 1) {
            enqueue(ls[0], null);
            if (propagate() != null) {
                ok = false;
            }
            return ok;
        }
        Clause c = new Clause(Arrays.copyOf(ls, n), false);
        clauses.add(c);
        attach(c);
        return true;
    }

    private void attach(Clause c) {
        watches.get(c.lits[0]).add(c);
        watches.get(c.lits[1]).add(c);
    }

    private void enqueue(int l, Clause from) {
        int v = l >> 1;
        assign[v] = (byte) ((l & 1) == 0 ? 1 : -1);
        level[v] = decisionLevel();
        reason[v] = from;
        trail[trailSize++] = l;
    }

    private Clause propagate() {
        while (qhead < trailSize) {
            int p = trail[qhead++];
            int fl = p ^ 1;
            propagations++;
            List<Clause> ws = watches.get(fl);
            int i = 0;
            int j = 0;
            int size = ws.size();
            while (i < size) {
                Clause c = ws.get(i++);
                if (c.deleted) {
                    continue;
                }
                int[] ls = c.lits;
                if (ls[0] == fl) {
                    ls[0] = ls[1];
                    ls[1] = fl;
                }
                if (litValue(ls[0]) == 1) {
                    ws.set(j++, c);
                    continue;
                }
                boolean moved = false;
                for (int k = 2; k < ls.length; k++) {
                    if (litValue(ls[k]) != -1) {
                        ls[1] = ls[k];
                        ls[k] = fl;
                        watches.get(ls[1]).add(c);
                        moved = true;
                        break;
                    }
                }
                if (moved) {
                    continue;
                }
                ws.set(j++, c);
                if (litValue(ls[0]) == -1) {
                    while (i < size) {
                        ws.set(j++, ws.get(i++));
                    }
                    ws.subList(j, size).clear();
                    qhead = trailSize;
                    return c;
                }
                enqueue(ls[0], c);
            }
            ws.subList(j, size).clear();
        }
        return null;
    }

    private void cancelUntil(int lvl) {
        if (decisionLevel() <= lvl) {
            return;
        }
        int stop = trailLim.get(lvl);
        for (int i = trailSize - 1; i >= stop; i--) {
            int v = trail[i] >> 1;
            if (options.phaseSaving()) {
                phase[v] = assign[v] == 1;
            }
            assign[v] = 0;
            reason[v] = null;
            if (heapIndex[v] < 0) {
                heapInsert(v);
            }
        }
        trailSize = stop;
        qhead = stop;
        trailLim.subList(lvl, trailLim.size()).clear();
    }

    // ---- conflict analysis ------------------------------------------------------------------

    private int[] learntBuf = new int[16];
    private int[] levelStamp = new int[16];
    private int stamp;

    private int[] analyze(Clause confl, int[] backLevel) {
        int size = 1;
        if (learntBuf.length < vars + 1) {
            learntBuf = new int[2 * vars + 1];
        }
        int pathC = 0;
        int p = -1;
        int idx = trailSize - 1;
        do {
            if (confl.learnt) {
                bumpClause(confl);
            }
            int[] ls = confl.lits;
            for (int k = (p == -1 ? 0 : 1); k < ls.length; k++) {
                int q = ls[k];
                int v = q >> 1;
                if (!seen[v] && level[v] > 0) {
                    bumpVar(v);
                    seen[v] = true;
                    if (level[v] >= decisionLevel()) {
                        pathC++;
                    } else {
                        learntBuf[size++] = q;
                    }
                }
            }
            while (!seen[trail[idx] >> 1]) {
                idx--;
            }
            p = trail[idx];
            idx--;
            confl = reason[p >> 1];
            seen[p >> 1] = false;
            pathC--;
        } while (pathC > 0);
        learntBuf[0] = p ^ 1;
        // recursive-free minimisation: drop literals whose reason is subsumed by the clause
        int[] all = Arrays.copyOf(learntBuf, size);
        int kept = 1;
        for (int k = 1; k < size; k++) {
            int v = learntBuf[k] >> 1;
            Clause r = reason[v];
            boolean redundant = r != null;
            if (r != null) {
                for (int j = 1; j < r.lits.length; j++) {
                    int u = r.lits[j] >> 1;
                    if (!seen[u] && level[u] > 0) {
                        redundant = false;
                        break;
                    }
                }
            }
            if (!redundant) {
                learntBuf[kept++] = learntBuf[k];
            }
        }
        for (int k = 1; k < size; k++) {
            seen[all[k] >> 1] = false;
        }
        int[] out = Arrays.copyOf(learntBuf, kept);
        int bt = 0;
        int at = 1;
        for (int k = 1; k < out.length; k++) {
            if (level[out[k] >> 1] > bt) {
                bt = level[out[k] >> 1];
                at = k;
            }
        }
        if (out.length > 1) {
            int t = out[1];
            out[1] = out[at];
            out[at] = t;
        }
        backLevel[0] = bt;
        return out;
    }

    private int lbd(int[] ls) {
        if (levelStamp.length < vars + 2) {
            levelStamp = new int[2 * vars + 2];
        }
        stamp++;
        int n = 0;
        for (int l : ls) {
            int lv = level[l >> 1];
            if (levelStamp[lv] != stamp) {
                levelStamp[lv] = stamp;
                n++;
            }
        }
        return n;
    }

    private void bumpVar(int v) {
        if (!options.vsids()) {
            return;
        }
        activity[v] += varInc;
        if (activity[v] > 1e100) {
            for (int i = 0; i < vars; i++) {
                activity[i] *= 1e-100;
            }
            varInc *= 1e-100;
        }
        if (heapIndex[v] >= 0) {
            siftUp(heapIndex[v]);
        }
    }

    private void bumpClause(Clause c) {
        c.activity += clauseInc;
        if (c.activity > 1e20) {
            for (Clause l : learnts) {
                l.activity *= 1e-20;
            }
            clauseInc *= 1e-20;
        }
    }

    // ---- heap --------------------------------------------------------------------------------

    private boolean better(int a, int b) {
        double sa = activity[a] * weight[a];
        double sb = activity[b] * weight[b];
        return sa > sb || (sa == sb && a < b);
    }

    private void heapInsert(int v) {
        heapIndex[v] = heapSize;
        heap[heapSize++] = v;
        siftUp(heapSize - 1);
    }

    private void siftUp(int i) {
        int v = heap[i];
        while (i > 0) {
            int parent = (i - 1) / 2;
            if (!better(v, heap[parent])) {
                break;
            }
            heap[i] = heap[parent];
            heapIndex[heap[i]] = i;
            i = parent;
        }
        heap[i] = v;
        heapIndex[v] = i;
    }

    private void siftDown(int i) {
        int v = heap[i];
        while (true) {
            int c = 2 * i + 1;
            if (c >= heapSize) {
                break;
            }
            if (c + 1 < heapSize && better(heap[c + 1], heap[c])) {
                c++;
            }
            if (!better(heap[c], v)) {
                break;
            }
            heap[i] = heap[c];
            heapIndex[heap[i]] = i;
            i = c;
        }
        heap[i] = v;
        heapIndex[v] = i;
    }

    private int heapPop() {
        int v = heap[0];
        heapIndex[v] = -1;
        heapSize--;
        if (heapSize > 0) {
            heap[0] = heap[heapSize];
            heapIndex[heap[0]] = 0;
            siftDown(0);
        }
        return v;
    }

    private void rebuildHeap() {
        heapSize = 0;
        for (int v = 0; v < vars; v++) {
            heapIndex[v] = -1;
        }
        for (int v = 0; v < vars; v++) {
            if (assign[v] == 0) {
                heapInsert(v);
            }
        }
    }

    // ---- search ------------------------------------------------------------------------------

    private static double luby(double y, int x) {
        int size = 1;
        int seq = 0;
        while (size < x + 1) {
            seq++;
            size = 2 * size + 1;
        }
        while (size - 1 != x) {
            size = (size - 1) >> 1;
            seq--;
            x = x % size;
        }
        return Math.pow(y, seq);
    }

    private void reduceDb() {
        List<Clause> sorted = new ArrayList<>(learnts);
        sorted.sort((a, b) -> a.lbd != b.lbd ? Integer.compare(b.lbd, a.lbd) : Double.compare(a.activity, b.activity));
        int target = sorted.size() / 2;
        int removed = 0;
        for (Clause c : sorted) {
            if (removed >= target) {
                break;
            }
            if (c.lbd <= 2 || locked(c)) {
                continue;
            }
            c.deleted = true;
            removed++;
        }
        learnts.removeIf(c -> c.deleted);
        deletedCount += removed;
    }

    private boolean locked(Clause c) {
        int v = c.lits[0] >> 1;
        return reason[v] == c && assign[v] != 0;
    }

    /**
     * Solves the current clause set.
     *
     * @return SAT, UNSAT or UNKNOWN (conflict budget exhausted; counted per call)
     */
    public Status solve() {
        model = null;
        if (!ok) {
            return Status.UNSAT;
        }
        cancelUntil(0);
        if (propagate() != null) {
            ok = false;
            return Status.UNSAT;
        }
        if (maxLearnts == 0) {
            maxLearnts = Math.max(200, clauses.size() / 3.0);
        }
        long start = conflicts;
        int restartIndex = 0;
        long nextRestart = (long) (luby(2, restartIndex) * options.lubyUnit());
        long sinceRestart = 0;
        int[] bt = new int[1];
        while (true) {
            Clause confl = propagate();
            if (confl != null) {
                conflicts++;
                sinceRestart++;
                if (decisionLevel() == 0) {
                    ok = false;
                    return Status.UNSAT;
                }
                int[] learnt = analyze(confl, bt);
                cancelUntil(bt[0]);
                int l = lbd(learnt);
                if (learnt.length == 1) {
                    enqueue(learnt[0], null);
                } else {
                    Clause c = new Clause(learnt, true);
                    c.lbd = l;
                    bumpClause(c);
                    learnts.add(c);
                    attach(c);
                    enqueue(learnt[0], c);
                }
                learntCount++;
                varInc /= options.varDecay();
                clauseInc /= 0.999;
                lbdSum += l;
                lbdTotal++;
                lbdQueue[lbdQueueHead] = l;
                lbdQueueHead = (lbdQueueHead + 1) % lbdQueue.length;
                lbdQueueSize = Math.min(lbdQueueSize + 1, lbdQueue.length);
                if (conflicts - start >= options.conflictLimit()) {
                    cancelUntil(0);
                    return Status.UNKNOWN;
                }
                continue;
            }
            boolean restart = switch (options.restarts()) {
                case NONE -> false;
                case LUBY -> sinceRestart >= nextRestart;
                case GLUCOSE -> lbdQueueSize == lbdQueue.length
                    && recentLbd() * 0.8 > lbdSum / Math.max(1, lbdTotal);
            };
            if (restart) {
                cancelUntil(0);
                restartCount++;
                sinceRestart = 0;
                restartIndex++;
                nextRestart = (long) (luby(2, restartIndex) * options.lubyUnit());
                lbdQueueSize = 0;
                continue;
            }
            if (options.deletion() && learnts.size() - trailSize >= maxLearnts) {
                reduceDb();
                maxLearnts *= 1.1;
            }
            int next = -1;
            while (heapSize > 0) {
                int v = heapPop();
                if (assign[v] == 0) {
                    next = v;
                    break;
                }
            }
            if (next < 0) {
                model = new boolean[vars];
                for (int v = 0; v < vars; v++) {
                    model[v] = assign[v] == 1;
                }
                cancelUntil(0);
                return Status.SAT;
            }
            decisions++;
            trailLim.add(trailSize);
            enqueue(2 * next + (phase[next] ? 0 : 1), null);
        }
    }

    private double recentLbd() {
        double s = 0;
        for (int i = 0; i < lbdQueueSize; i++) {
            s += lbdQueue[i];
        }
        return s / Math.max(1, lbdQueueSize);
    }

    /**
     * Convenience: solves a DIMACS formula.
     *
     * @param variables the number of variables
     * @param clauses the clauses
     * @param options the options
     * @return the model (index 0 for variable 1), or null if unsatisfiable or out of budget
     */
    public static boolean[] solve(int variables, List<int[]> clauses, Options options) {
        CdclSolver s = of(variables, clauses, options);
        return s.solve() == Status.SAT ? s.model() : null;
    }
}
