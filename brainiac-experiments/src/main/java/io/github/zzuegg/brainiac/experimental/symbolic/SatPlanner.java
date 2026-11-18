package io.github.zzuegg.brainiac.experimental.symbolic;

import io.github.zzuegg.brainiac.experimental.logic.CdclSolver;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.function.BiFunction;

/**
 * Planning as satisfiability (Kautz &amp; Selman, ECAI 1992; "Pushing the envelope", AAAI 1996;
 * Rintanen, Heljanko &amp; Niemelä, AIJ 2006; Rintanen, "Planning as satisfiability: heuristics",
 * AIJ 2012). "Is there a plan with {@code T} steps?" becomes a CNF with one variable per fact and
 * time point and one per action and step: the initial state and the goal as unit clauses,
 * preconditions and effects as implications, explanatory frame axioms (a fact changes only if an
 * action at that step changes it), and a parallelism constraint per encoding:
 *
 * <ul>
 *   <li>{@link Encoding#SEQUENTIAL}: at most one action per step (a sequential counter), so the
 *       first satisfiable {@code T} is the optimal sequential plan length.</li>
 *   <li>{@link Encoding#FORALL_STEP}: actions may share a step when they do not interfere (no
 *       action deletes another's precondition or add effect), so every order of a step is valid
 *       and the first satisfiable {@code T} is the optimal parallel makespan.</li>
 *   <li>{@link Encoding#EXISTS_STEP}: under a fixed total order of the actions, two actions may
 *       share a step unless the earlier one deletes the later one's precondition or their effects
 *       conflict, so the step is executable in that order; this allows more parallelism and fewer
 *       steps.</li>
 * </ul>
 * Horizons grow linearly ({@code T = 0, 1, 2, ...}) or by doubling (fewer, larger calls; the
 * makespan is then no longer minimal). The CNFs are solved by {@link CdclSolver}, or by any solver
 * given as a function (for the DPLL baseline).
 */
public final class SatPlanner {

    private SatPlanner() {
    }

    /** The step semantics. */
    public enum Encoding {
        /** One action per step. */
        SEQUENTIAL,
        /** Non-interfering parallel actions (any order). */
        FORALL_STEP,
        /** Parallel actions executable in a fixed order. */
        EXISTS_STEP
    }

    /**
     * A CNF for one horizon.
     *
     * @param variables the number of variables
     * @param clauses DIMACS clauses
     * @param horizon the number of steps
     */
    public record Cnf(int variables, List<int[]> clauses, int horizon) {
    }

    /**
     * A planning result.
     *
     * @param plan the sequentialised plan (empty if none)
     * @param found whether a plan was found
     * @param makespan the number of steps of the satisfiable horizon
     * @param horizonsTried horizons encoded and solved
     * @param decisions total solver decisions (CDCL only)
     * @param largestCnf clauses of the largest formula
     * @param millis wall time
     */
    public record Result(List<StripsTask.Action> plan, boolean found, int makespan, int horizonsTried,
            long decisions, int largestCnf, double millis) {
    }

    private static int factVar(StripsTask task, int f, int t) {
        return t * task.facts() + f + 1;
    }

    private static int actionVar(StripsTask task, int horizon, int a, int t) {
        return (horizon + 1) * task.facts() + t * task.actions().size() + a + 1;
    }

    private static boolean contains(int[] xs, int x) {
        for (int y : xs) {
            if (y == x) {
                return true;
            }
        }
        return false;
    }

    /**
     * Encodes a horizon.
     *
     * @param task the task
     * @param horizon the number of steps
     * @param encoding the step semantics
     * @return the CNF
     */
    public static Cnf encode(StripsTask task, int horizon, Encoding encoding) {
        int nf = task.facts();
        int na = task.actions().size();
        List<int[]> cs = new ArrayList<>();
        int next = (horizon + 1) * nf + horizon * na + 1;
        BitSet init = task.initialState();
        for (int f = 0; f < nf; f++) {
            cs.add(new int[] {init.get(f) ? factVar(task, f, 0) : -factVar(task, f, 0)});
        }
        for (int g : task.goal()) {
            cs.add(new int[] {factVar(task, g, horizon)});
        }
        List<List<Integer>> adders = new ArrayList<>();
        List<List<Integer>> deleters = new ArrayList<>();
        for (int f = 0; f < nf; f++) {
            adders.add(new ArrayList<>());
            deleters.add(new ArrayList<>());
        }
        for (StripsTask.Action a : task.actions()) {
            for (int f : a.add()) {
                adders.get(f).add(a.id());
            }
            for (int f : a.del()) {
                if (!contains(a.add(), f)) {
                    deleters.get(f).add(a.id());
                }
            }
        }
        // interference pairs, computed once
        List<int[]> conflicts = new ArrayList<>();
        if (encoding != Encoding.SEQUENTIAL) {
            for (int i = 0; i < na; i++) {
                StripsTask.Action a = task.actions().get(i);
                for (int j = i + 1; j < na; j++) {
                    StripsTask.Action b = task.actions().get(j);
                    boolean effectClash = clash(a, b) || clash(b, a);
                    boolean aHurtsB = deletesPre(a, b);
                    boolean bHurtsA = deletesPre(b, a);
                    boolean bad = encoding == Encoding.FORALL_STEP ? effectClash || aHurtsB || bHurtsA
                        : effectClash || aHurtsB; // EXISTS: order i < j, so only the earlier may not hurt the later
                    if (bad) {
                        conflicts.add(new int[] {i, j});
                    }
                }
            }
        }
        for (int t = 0; t < horizon; t++) {
            for (StripsTask.Action a : task.actions()) {
                int av = actionVar(task, horizon, a.id(), t);
                for (int p : a.pre()) {
                    cs.add(new int[] {-av, factVar(task, p, t)});
                }
                for (int f : a.add()) {
                    cs.add(new int[] {-av, factVar(task, f, t + 1)});
                }
                for (int f : a.del()) {
                    if (!contains(a.add(), f)) {
                        cs.add(new int[] {-av, -factVar(task, f, t + 1)});
                    }
                }
            }
            for (int f = 0; f < nf; f++) {
                int[] becomesTrue = new int[adders.get(f).size() + 2];
                becomesTrue[0] = factVar(task, f, t);
                becomesTrue[1] = -factVar(task, f, t + 1);
                for (int k = 0; k < adders.get(f).size(); k++) {
                    becomesTrue[k + 2] = actionVar(task, horizon, adders.get(f).get(k), t);
                }
                cs.add(becomesTrue);
                int[] becomesFalse = new int[deleters.get(f).size() + 2];
                becomesFalse[0] = -factVar(task, f, t);
                becomesFalse[1] = factVar(task, f, t + 1);
                for (int k = 0; k < deleters.get(f).size(); k++) {
                    becomesFalse[k + 2] = actionVar(task, horizon, deleters.get(f).get(k), t);
                }
                cs.add(becomesFalse);
            }
            if (encoding == Encoding.SEQUENTIAL) {
                // sequential-counter at-most-one: s_i means "some action <= i is chosen"
                int prev = 0;
                for (int i = 0; i < na; i++) {
                    int x = actionVar(task, horizon, i, t);
                    int s = next++;
                    cs.add(new int[] {-x, s});
                    if (prev != 0) {
                        cs.add(new int[] {-prev, s});
                        cs.add(new int[] {-prev, -x});
                    }
                    prev = s;
                }
            } else {
                for (int[] c : conflicts) {
                    cs.add(new int[] {-actionVar(task, horizon, c[0], t), -actionVar(task, horizon, c[1], t)});
                }
            }
        }
        return new Cnf(next - 1, cs, horizon);
    }

    private static boolean clash(StripsTask.Action a, StripsTask.Action b) {
        for (int f : a.add()) {
            if (contains(b.del(), f) && !contains(b.add(), f)) {
                return true;
            }
        }
        return false;
    }

    private static boolean deletesPre(StripsTask.Action a, StripsTask.Action b) {
        for (int f : a.del()) {
            if (!contains(a.add(), f) && contains(b.pre(), f)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Decodes a model into a plan (actions of a step in index order, which is valid for every
     * encoding here).
     *
     * @param task the task
     * @param horizon the horizon
     * @param model the model (index 0 for variable 1)
     * @return the plan
     */
    public static List<StripsTask.Action> decode(StripsTask task, int horizon, boolean[] model) {
        List<StripsTask.Action> plan = new ArrayList<>();
        for (int t = 0; t < horizon; t++) {
            for (StripsTask.Action a : task.actions()) {
                if (model[actionVar(task, horizon, a.id(), t) - 1]) {
                    plan.add(a);
                }
            }
        }
        return plan;
    }

    /**
     * Plans with CDCL.
     *
     * @param task the task
     * @param encoding the step semantics
     * @param doubling whether horizons double instead of growing by one
     * @param maxHorizon the largest horizon tried
     * @param conflictLimit the per-call conflict budget
     * @return the result
     */
    public static Result plan(StripsTask task, Encoding encoding, boolean doubling, int maxHorizon,
            long conflictLimit) {
        long[] decisions = new long[1];
        return plan(task, encoding, doubling, maxHorizon, (n, cs) -> {
            CdclSolver s = CdclSolver.of(n, cs, CdclSolver.Options.defaults().withConflictLimit(conflictLimit));
            CdclSolver.Status st = s.solve();
            decisions[0] += s.stats().decisions();
            return st == CdclSolver.Status.SAT ? s.model() : null;
        }, decisions);
    }

    /**
     * Plans with any SAT solver.
     *
     * @param task the task
     * @param encoding the step semantics
     * @param doubling whether horizons double
     * @param maxHorizon the largest horizon
     * @param solver maps (variables, clauses) to a model (index 0 for variable 1) or null
     * @param decisions a counter the solver may fill (reported in the result)
     * @return the result
     */
    public static Result plan(StripsTask task, Encoding encoding, boolean doubling, int maxHorizon,
            BiFunction<Integer, List<int[]>, boolean[]> solver, long[] decisions) {
        long t0 = System.nanoTime();
        int tried = 0;
        int largest = 0;
        int h = 0;
        while (h <= maxHorizon) {
            Cnf cnf = encode(task, h, encoding);
            largest = Math.max(largest, cnf.clauses().size());
            tried++;
            boolean[] model = solver.apply(cnf.variables(), cnf.clauses());
            if (model != null) {
                List<StripsTask.Action> plan = decode(task, h, model);
                if (Double.isInfinite(task.validate(plan))) {
                    throw new IllegalStateException("decoded plan is invalid at horizon " + h);
                }
                return new Result(plan, true, h, tried, decisions[0], largest, (System.nanoTime() - t0) / 1e6);
            }
            h = doubling ? Math.max(1, 2 * h) : h + 1;
        }
        return new Result(List.of(), false, -1, tried, decisions[0], largest, (System.nanoTime() - t0) / 1e6);
    }
}
