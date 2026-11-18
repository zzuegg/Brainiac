package io.github.zzuegg.brainiac.experiments.symbolic;

import io.github.zzuegg.brainiac.experimental.data.ConstraintInstances;
import io.github.zzuegg.brainiac.experimental.data.RandomSatFormulas;
import io.github.zzuegg.brainiac.experimental.logic.AnyCsp;
import io.github.zzuegg.brainiac.experimental.logic.CdclSolver;
import io.github.zzuegg.brainiac.experimental.logic.ConstraintSolver;
import io.github.zzuegg.brainiac.experimental.logic.Csp;
import io.github.zzuegg.brainiac.experimental.logic.MinConflicts;
import io.github.zzuegg.brainiac.experiments.graphs.GraphReasoning;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Wave 13 items 13.2.7-13.2.8: constraint satisfaction by search with propagation, by
 * min-conflicts, and by ANYCSP's learned global search.
 *
 * <ul>
 *   <li>{@link #sudoku}, {@link #queens}, {@link #phaseTransition} (13.2.7): backtracking, forward
 *       checking, MAC and conflict-directed backjumping with MRV/degree/LCV, min-conflicts, and
 *       the direct CNF encoding solved by {@link RandomSatFormulas}' DPLL and by CDCL.</li>
 *   <li>{@link #learned} (13.2.8): ANYCSP trained on small 3-colouring instances near the
 *       threshold and tested on twice-larger ones, against min-conflicts with tabu, WalkSAT on the
 *       CNF encoding and MAC.</li>
 * </ul>
 */
public final class ConstraintSolving {

    private ConstraintSolving() {
    }

    /**
     * A row.
     *
     * @param task the instance family
     * @param method the solver
     * @param solved instances solved
     * @param instances instances
     * @param work mean nodes (complete search) or steps (local search)
     * @param millis mean wall time
     */
    public record Row(String task, String method, int solved, int instances, double work, double millis) {

        @Override
        public String toString() {
            return String.format("%s/%s %d/%d work=%.0f %.1fms", task, method, solved, instances, work, millis);
        }
    }

    private static Row complete(String task, String name, List<Csp> csps, ConstraintSolver.Algorithm alg,
            ConstraintSolver.VariableOrder order, boolean lcv, long budget) {
        int solved = 0;
        double nodes = 0;
        double ms = 0;
        for (Csp c : csps) {
            ConstraintSolver.Result r = new ConstraintSolver(c, alg, order, lcv, false, budget).solve();
            if (r.solved()) {
                if (c.violations(r.assignment()) != 0) {
                    throw new IllegalStateException("invalid CSP solution");
                }
                solved++;
            }
            nodes += r.nodes();
            ms += r.millis();
        }
        return new Row(task, name, solved, csps.size(), nodes / csps.size(), ms / csps.size());
    }

    private static Row cnf(String task, String name, List<Csp> csps, boolean cdcl) {
        int solved = 0;
        double ms = 0;
        double work = 0;
        for (Csp c : csps) {
            long t0 = System.nanoTime();
            boolean[] model;
            if (cdcl) {
                CdclSolver s = CdclSolver.of(c.cnfVariables(), c.toCnf(), CdclSolver.Options.defaults());
                model = s.solve() == CdclSolver.Status.SAT ? s.model() : null;
                work += s.stats().decisions();
            } else {
                boolean[] m = RandomSatFormulas.solve(c.cnfVariables(), c.toCnf());
                model = m == null ? null : java.util.Arrays.copyOfRange(m, 1, m.length);
            }
            ms += (System.nanoTime() - t0) / 1e6;
            if (model != null && c.violations(c.decode(model)) == 0) {
                solved++;
            }
        }
        return new Row(task, name, solved, csps.size(), work / csps.size(), ms / csps.size());
    }

    /**
     * 9x9 Sudoku.
     *
     * @param puzzles puzzles
     * @param blanks blank cells
     * @param seed the seed
     * @return rows
     */
    public static List<Row> sudoku(int puzzles, int blanks, long seed) {
        List<Csp> csps = new ArrayList<>();
        for (int i = 0; i < puzzles; i++) {
            csps.add(ConstraintInstances.sudoku(3, blanks, seed + i).csp());
        }
        String t = "sudoku-9x9-" + blanks;
        List<Row> rows = new ArrayList<>();
        rows.add(complete(t, "BT static", csps, ConstraintSolver.Algorithm.BT, ConstraintSolver.VariableOrder.STATIC,
            false, 200_000));
        rows.add(complete(t, "FC MRV", csps, ConstraintSolver.Algorithm.FC, ConstraintSolver.VariableOrder.MRV, false,
            200_000));
        rows.add(complete(t, "MAC MRV", csps, ConstraintSolver.Algorithm.MAC, ConstraintSolver.VariableOrder.MRV,
            false, 200_000));
        rows.add(complete(t, "MAC MRV+deg+LCV", csps, ConstraintSolver.Algorithm.MAC,
            ConstraintSolver.VariableOrder.MRV_DEG, true, 200_000));
        rows.add(cnf(t, "DPLL on CNF", csps, false));
        rows.add(cnf(t, "CDCL on CNF (work=decisions)", csps, true));
        return rows;
    }

    /**
     * N-queens: complete search at small n and min-conflicts at large n.
     *
     * @param small the board for complete search
     * @param large the board for min-conflicts
     * @param seed the seed
     * @return rows
     */
    public static List<Row> queens(int small, int large, long seed) {
        List<Row> rows = new ArrayList<>();
        List<Csp> s = List.of(ConstraintInstances.nQueens(small));
        String t = "queens-" + small;
        rows.add(complete(t, "BT static", s, ConstraintSolver.Algorithm.BT, ConstraintSolver.VariableOrder.STATIC,
            false, 2_000_000));
        rows.add(complete(t, "FC MRV", s, ConstraintSolver.Algorithm.FC, ConstraintSolver.VariableOrder.MRV, false,
            2_000_000));
        rows.add(complete(t, "MAC MRV", s, ConstraintSolver.Algorithm.MAC, ConstraintSolver.VariableOrder.MRV, false,
            2_000_000));
        Csp big = ConstraintInstances.nQueens(large);
        rows.add(complete("queens-" + large, "FC MRV", List.of(big), ConstraintSolver.Algorithm.FC,
            ConstraintSolver.VariableOrder.MRV, false, 20_000));
        rows.add(complete("queens-" + large, "BT static", List.of(big), ConstraintSolver.Algorithm.BT,
            ConstraintSolver.VariableOrder.STATIC, false, 20_000));
        MinConflicts.Result mc = MinConflicts.solve(big, 10_000, true, 0, 0, seed);
        rows.add(new Row("queens-" + large, "min-conflicts (greedy init)", mc.solved() ? 1 : 0, 1, mc.steps(),
            mc.millis()));
        return rows;
    }

    /**
     * Model RB swept across the phase transition.
     *
     * @param n variables
     * @param tightness the tightness values
     * @param perPoint instances per value
     * @param seed the seed
     * @return per tightness, rows for BT, FC, MAC and CBJ (MRV ordering) with the share of
     *     satisfiable instances in the task name
     */
    public static Map<Double, List<Row>> phaseTransition(int n, double[] tightness, int perPoint, long seed) {
        Map<Double, List<Row>> out = new LinkedHashMap<>();
        for (double p : tightness) {
            List<Csp> csps = new ArrayList<>();
            for (int i = 0; i < perPoint; i++) {
                csps.add(ConstraintInstances.modelRb(n, 0.8, 0.7, p, seed + 1000L * i + (long) (p * 100)));
            }
            String t = String.format("RB n=%d p=%.2f", n, p);
            List<Row> rows = new ArrayList<>();
            rows.add(complete(t, "BT", csps, ConstraintSolver.Algorithm.BT, ConstraintSolver.VariableOrder.MRV_DEG,
                false, 1_000_000));
            rows.add(complete(t, "CBJ", csps, ConstraintSolver.Algorithm.CBJ, ConstraintSolver.VariableOrder.MRV_DEG,
                false, 1_000_000));
            rows.add(complete(t, "FC", csps, ConstraintSolver.Algorithm.FC, ConstraintSolver.VariableOrder.MRV_DEG,
                false, 1_000_000));
            rows.add(complete(t, "MAC", csps, ConstraintSolver.Algorithm.MAC, ConstraintSolver.VariableOrder.MRV_DEG,
                false, 1_000_000));
            out.put(p, rows);
        }
        return out;
    }

    /**
     * 13.2.8: ANYCSP on 3-colouring.
     *
     * @param trainN training graph size
     * @param testN test graph size
     * @param episodes training episodes
     * @param testInstances test instances
     * @param steps search step budget for ANYCSP, min-conflicts and WalkSAT flips (x10)
     * @param seed the seed
     * @return rows; the first is "ANYCSP untrained" as an ablation
     */
    public static List<Row> learned(int trainN, int testN, int episodes, int testInstances, int steps, long seed) {
        double degree = 4.4;
        List<Csp> train = new ArrayList<>();
        for (int i = 0; i < 16; i++) {
            train.add(ConstraintInstances.colouring(trainN, ConstraintInstances.plantedGraph(trainN, 3, degree,
                seed + i), 3));
        }
        List<Csp> test = new ArrayList<>();
        for (int i = 0; i < testInstances; i++) {
            test.add(ConstraintInstances.colouring(testN, ConstraintInstances.plantedGraph(testN, 3, degree,
                seed + 500 + i), 3));
        }
        String t = String.format("3-COL n=%d (train n=%d)", testN, trainN);
        List<Row> rows = new ArrayList<>();
        AnyCsp untrained = new AnyCsp(16, seed);
        rows.add(anyRow(t, "ANYCSP untrained", untrained, test, steps, seed));
        AnyCsp net = new AnyCsp(16, seed);
        long t0 = System.nanoTime();
        net.train(train, episodes, 24, 0.75, 3e-3f, seed + 1);
        double trainMs = (System.nanoTime() - t0) / 1e6;
        Row trained = anyRow(t, String.format("ANYCSP (train %.0fms)", trainMs), net, test, steps, seed);
        rows.add(trained);
        int mcSolved = 0;
        double mcSteps = 0;
        double mcMs = 0;
        for (int i = 0; i < test.size(); i++) {
            MinConflicts.Result r = MinConflicts.solve(test.get(i), steps, false, 0.0, 5, seed + i);
            mcSolved += r.solved() ? 1 : 0;
            mcSteps += r.steps();
            mcMs += r.millis();
        }
        rows.add(new Row(t, "min-conflicts tabu", mcSolved, test.size(), mcSteps / test.size(), mcMs / test.size()));
        int wsSolved = 0;
        double wsFlips = 0;
        double wsMs = 0;
        SplittableRandom r = new SplittableRandom(seed);
        for (Csp c : test) {
            long t1 = System.nanoTime();
            int f = GraphReasoning.walkSatFlips(new RandomSatFormulas.Formula(c.cnfVariables(), c.toCnf(), true),
                steps * 10, r);
            wsMs += (System.nanoTime() - t1) / 1e6;
            wsSolved += f >= 0 ? 1 : 0;
            wsFlips += f >= 0 ? f : steps * 10;
        }
        rows.add(new Row(t, "WalkSAT on CNF (flips)", wsSolved, test.size(), wsFlips / test.size(),
            wsMs / test.size()));
        rows.add(complete(t, "MAC MRV", test, ConstraintSolver.Algorithm.MAC, ConstraintSolver.VariableOrder.MRV_DEG,
            false, 200_000));
        return rows;
    }

    private static Row anyRow(String t, String name, AnyCsp net, List<Csp> test, int steps, long seed) {
        int solved = 0;
        double st = 0;
        double ms = 0;
        for (int i = 0; i < test.size(); i++) {
            long t0 = System.nanoTime();
            AnyCsp.Run r = net.solve(test.get(i), steps, seed + 77 + i);
            ms += (System.nanoTime() - t0) / 1e6;
            solved += r.solved() ? 1 : 0;
            st += r.steps();
        }
        return new Row(t, name, solved, test.size(), st / test.size(), ms / test.size());
    }

    /**
     * Prints the tables.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        sudoku(5, 55, 1).forEach(System.out::println);
        queens(12, 200, 1).forEach(System.out::println);
        phaseTransition(20, new double[] {0.4, 0.5, 0.6, 0.68, 0.75, 0.85}, 4, 1)
            .forEach((k, v) -> v.forEach(System.out::println));
        learned(30, 60, 300, 6, 300, 1).forEach(System.out::println);
    }
}
