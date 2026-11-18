package io.github.zzuegg.brainiac.experimental.logic;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;

/**
 * Complete search for binary {@link Csp}s with the classical refinements (Mackworth, "Consistency
 * in networks of relations", AIJ 1977; Haralick &amp; Elliott, "Increasing tree search efficiency
 * for constraint satisfaction problems", AIJ 1980; Prosser, "Hybrid algorithms for the constraint
 * satisfaction problem", Comput. Intell. 1993; Sabin &amp; Freuder, "Contradicting conventional
 * wisdom in constraint satisfaction", CP 1994).
 *
 * <ul>
 *   <li>{@link Algorithm#BT}: chronological backtracking, checking each new value against the
 *       assigned neighbours.</li>
 *   <li>{@link Algorithm#FC}: forward checking, removing the values of unassigned neighbours that
 *       conflict with each new assignment and backtracking on a wiped-out domain.</li>
 *   <li>{@link Algorithm#MAC}: maintaining arc consistency, running AC-3 after every
 *       assignment.</li>
 *   <li>{@link Algorithm#CBJ}: conflict-directed backjumping: every variable keeps the levels of
 *       the assignments that ruled out its values and a dead end jumps back to the deepest of
 *       them, passing on the rest.</li>
 * </ul>
 * Variable ordering: static, minimum remaining values (MRV), or MRV with the degree heuristic as
 * tie-break; value ordering: natural or least constraining value (LCV). AC-3 can also run once
 * before search. Counted costs are search nodes (value assignments tried) and constraint checks.
 */
public final class ConstraintSolver {

    /** The search algorithm. */
    public enum Algorithm {
        /** Chronological backtracking. */
        BT,
        /** Forward checking. */
        FC,
        /** Maintaining arc consistency. */
        MAC,
        /** Conflict-directed backjumping (no look-ahead). */
        CBJ
    }

    /** The variable ordering. */
    public enum VariableOrder {
        /** Index order. */
        STATIC,
        /** Minimum remaining values. */
        MRV,
        /** MRV with the most constraints on unassigned variables as tie-break. */
        MRV_DEG
    }

    /**
     * A search result.
     *
     * @param solved whether a solution was found
     * @param complete whether the search finished (false: node budget exhausted)
     * @param assignment the solution (null if none)
     * @param nodes values tried
     * @param checks constraint checks
     * @param millis wall time
     */
    public record Result(boolean solved, boolean complete, int[] assignment, long nodes, long checks, double millis) {
    }

    private final Csp csp;
    private final Algorithm algorithm;
    private final VariableOrder order;
    private final boolean lcv;
    private final boolean ac3First;
    private final long maxNodes;
    private boolean[][] live;
    private int[] size;
    private int[] value;
    private int[] levelOf;
    private int[] trailVar;
    private int[] trailVal;
    private int trailSize;
    private long nodes;
    private long checks;
    private boolean outOfBudget;
    private BitSet[] conflict;
    private int[] varAt;

    /**
     * Creates a solver.
     *
     * @param csp the problem
     * @param algorithm the search
     * @param order the variable ordering
     * @param lcv whether to order values by least constraining value
     * @param ac3First whether to run AC-3 before search
     * @param maxNodes the node budget
     */
    public ConstraintSolver(Csp csp, Algorithm algorithm, VariableOrder order, boolean lcv, boolean ac3First,
            long maxNodes) {
        this.csp = csp;
        this.algorithm = algorithm;
        this.order = order;
        this.lcv = lcv;
        this.ac3First = ac3First;
        this.maxNodes = maxNodes;
    }

    /**
     * Searches for a solution.
     *
     * @return the result
     */
    public Result solve() {
        long t0 = System.nanoTime();
        int n = csp.variables();
        live = new boolean[n][];
        size = new int[n];
        value = new int[n];
        levelOf = new int[n];
        int total = 0;
        for (int x = 0; x < n; x++) {
            live[x] = new boolean[csp.domain(x)];
            Arrays.fill(live[x], true);
            size[x] = csp.domain(x);
            total += size[x];
        }
        Arrays.fill(value, -1);
        trailVar = new int[total];
        trailVal = new int[total];
        trailSize = 0;
        nodes = 0;
        checks = 0;
        outOfBudget = false;
        boolean ok = true;
        if (ac3First || algorithm == Algorithm.MAC) {
            ok = ac3(allArcs());
        }
        boolean solved;
        if (!ok) {
            solved = false;
        } else if (algorithm == Algorithm.CBJ) {
            conflict = new BitSet[n];
            varAt = new int[n + 1];
            solved = cbj(0) == -1;
        } else {
            solved = search(0);
        }
        return new Result(solved, !outOfBudget, solved ? value.clone() : null, nodes, checks,
            (System.nanoTime() - t0) / 1e6);
    }

    private void remove(int x, int a) {
        live[x][a] = false;
        size[x]--;
        trailVar[trailSize] = x;
        trailVal[trailSize] = a;
        trailSize++;
    }

    private void undo(int mark) {
        while (trailSize > mark) {
            trailSize--;
            live[trailVar[trailSize]][trailVal[trailSize]] = true;
            size[trailVar[trailSize]]++;
        }
    }

    private boolean allows(int c, int x, int a, int b) {
        checks++;
        return csp.allows(c, x, a, b);
    }

    private int select() {
        int best = -1;
        for (int x = 0; x < csp.variables(); x++) {
            if (value[x] >= 0) {
                continue;
            }
            if (order == VariableOrder.STATIC) {
                return x;
            }
            if (best < 0 || size[x] < size[best] || (size[x] == size[best] && order == VariableOrder.MRV_DEG
                && degree(x) > degree(best))) {
                best = x;
            }
        }
        return best;
    }

    private int degree(int x) {
        int d = 0;
        for (int c : csp.constraintsOf(x)) {
            if (value[csp.other(c, x)] < 0) {
                d++;
            }
        }
        return d;
    }

    private int[] values(int x) {
        List<Integer> vs = new ArrayList<>();
        for (int a = 0; a < live[x].length; a++) {
            if (live[x][a]) {
                vs.add(a);
            }
        }
        if (lcv) {
            int[] cost = new int[live[x].length];
            for (int a : vs) {
                for (int c : csp.constraintsOf(x)) {
                    int y = csp.other(c, x);
                    if (value[y] >= 0) {
                        continue;
                    }
                    for (int b = 0; b < live[y].length; b++) {
                        if (live[y][b] && !allows(c, x, a, b)) {
                            cost[a]++;
                        }
                    }
                }
            }
            vs.sort((p, q) -> Integer.compare(cost[p], cost[q]));
        }
        return vs.stream().mapToInt(Integer::intValue).toArray();
    }

    private boolean consistentWithAssigned(int x, int a) {
        for (int c : csp.constraintsOf(x)) {
            int y = csp.other(c, x);
            if (value[y] >= 0 && !allows(c, x, a, value[y])) {
                return false;
            }
        }
        return true;
    }

    private boolean search(int level) {
        int x = select();
        if (x < 0) {
            return true;
        }
        for (int a : values(x)) {
            if (nodes >= maxNodes) {
                outOfBudget = true;
                return false;
            }
            nodes++;
            if (algorithm == Algorithm.BT && !consistentWithAssigned(x, a)) {
                continue;
            }
            int mark = trailSize;
            value[x] = a;
            boolean ok = true;
            if (algorithm == Algorithm.FC) {
                ok = forwardCheck(x, a);
            } else if (algorithm == Algorithm.MAC) {
                for (int b = 0; b < live[x].length; b++) {
                    if (b != a && live[x][b]) {
                        remove(x, b);
                    }
                }
                ArrayDeque<int[]> q = new ArrayDeque<>();
                for (int c : csp.constraintsOf(x)) {
                    q.add(new int[] {csp.other(c, x), c});
                }
                ok = ac3(q);
            }
            if (ok && search(level + 1)) {
                return true;
            }
            undo(mark);
            value[x] = -1;
            if (outOfBudget) {
                return false;
            }
        }
        return false;
    }

    private boolean forwardCheck(int x, int a) {
        for (int c : csp.constraintsOf(x)) {
            int y = csp.other(c, x);
            if (value[y] >= 0) {
                continue;
            }
            for (int b = 0; b < live[y].length; b++) {
                if (live[y][b] && !allows(c, x, a, b)) {
                    remove(y, b);
                }
            }
            if (size[y] == 0) {
                return false;
            }
        }
        return true;
    }

    private ArrayDeque<int[]> allArcs() {
        ArrayDeque<int[]> q = new ArrayDeque<>();
        for (int c = 0; c < csp.constraints().size(); c++) {
            Csp.Constraint k = csp.constraints().get(c);
            q.add(new int[] {k.x(), c});
            q.add(new int[] {k.y(), c});
        }
        return q;
    }

    /** AC-3: arcs are (variable to revise, constraint). */
    private boolean ac3(ArrayDeque<int[]> q) {
        while (!q.isEmpty()) {
            int[] arc = q.poll();
            int x = arc[0];
            int c = arc[1];
            int y = csp.other(c, x);
            boolean revised = false;
            for (int a = 0; a < live[x].length; a++) {
                if (!live[x][a]) {
                    continue;
                }
                boolean support = false;
                for (int b = 0; b < live[y].length && !support; b++) {
                    support = live[y][b] && allows(c, x, a, b);
                }
                if (!support) {
                    remove(x, a);
                    revised = true;
                }
            }
            if (size[x] == 0) {
                return false;
            }
            if (revised) {
                for (int c2 : csp.constraintsOf(x)) {
                    if (c2 != c) {
                        q.add(new int[] {csp.other(c2, x), c2});
                    }
                }
            }
        }
        return true;
    }

    /** Prosser's CBJ; returns -1 when solved, otherwise the level to jump back to. */
    private int cbj(int level) {
        int x = select();
        if (x < 0) {
            return -1;
        }
        varAt[level] = x;
        levelOf[x] = level;
        conflict[x] = new BitSet();
        for (int a : values(x)) {
            if (nodes >= maxNodes) {
                outOfBudget = true;
                return -2;
            }
            nodes++;
            int culprit = -1;
            for (int c : csp.constraintsOf(x)) {
                int y = csp.other(c, x);
                if (value[y] >= 0 && !allows(c, x, a, value[y])) {
                    if (culprit < 0 || levelOf[y] < levelOf[culprit]) {
                        culprit = y;
                    }
                }
            }
            if (culprit >= 0) {
                conflict[x].set(levelOf[culprit]);
                continue;
            }
            value[x] = a;
            int r = cbj(level + 1);
            value[x] = -1;
            if (r == -1) {
                value[x] = a;
                return -1;
            }
            if (r == -2) {
                return -2;
            }
            if (r < level) {
                return r;
            }
        }
        conflict[x].clear(level);
        int h = conflict[x].length() - 1;
        if (h < 0) {
            return -3; // no culprit: unsatisfiable
        }
        BitSet pass = (BitSet) conflict[x].clone();
        pass.clear(h);
        conflict[varAt[h]].or(pass);
        return h;
    }
}
