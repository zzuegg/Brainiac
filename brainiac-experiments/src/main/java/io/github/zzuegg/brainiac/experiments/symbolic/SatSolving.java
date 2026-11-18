package io.github.zzuegg.brainiac.experiments.symbolic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ConstraintInstances;
import io.github.zzuegg.brainiac.experimental.data.PlanningDomains;
import io.github.zzuegg.brainiac.experimental.data.RandomSatFormulas;
import io.github.zzuegg.brainiac.experimental.graph.NeuroSat;
import io.github.zzuegg.brainiac.experimental.logic.CdclSolver;
import io.github.zzuegg.brainiac.experimental.logic.RlafGuidance;
import io.github.zzuegg.brainiac.experimental.symbolic.Landmarks;
import io.github.zzuegg.brainiac.experimental.symbolic.RelaxationHeuristics;
import io.github.zzuegg.brainiac.experimental.symbolic.SatPlanner;
import io.github.zzuegg.brainiac.experimental.symbolic.StateSpaceSearch;
import io.github.zzuegg.brainiac.experimental.symbolic.StripsTask;
import io.github.zzuegg.brainiac.experiments.graphs.GraphReasoning;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Wave 13 items 13.2.9, 13.2.10 and 13.2.16: conflict-driven clause learning, learned one-shot
 * solver guidance (RLAF) and planning as satisfiability.
 *
 * <ul>
 *   <li>{@link #cdcl} (13.2.9): CDCL against {@link RandomSatFormulas}' DPLL and WalkSAT on random
 *       3-SAT at the threshold, pigeonhole formulas, and structured colouring and planning CNFs;
 *       {@link #ablation} switches each CDCL feature off in turn; {@link #neuroSat} trains the
 *       NeuroSAT classifier as the decision-only baseline.</li>
 *   <li>{@link #rlaf} (13.2.10): plain VSIDS against RLAF's one-shot weights and polarities, the
 *       supervised backbone baseline and random weights, on training-size and twice-larger
 *       colouring CNFs.</li>
 *   <li>{@link #planning} (13.2.16): SATPlan's three encodings with linear and doubling horizons
 *       against GBFS with h_FF and optimal A* with LM-cut, and DPLL on the same CNFs.</li>
 * </ul>
 */
public final class SatSolving {

    private SatSolving() {
    }

    /**
     * A row.
     *
     * @param task the formula family
     * @param method the solver
     * @param solved formulas decided (or plans found)
     * @param instances formulas
     * @param conflicts mean conflicts (or another work count named in the method)
     * @param millis mean wall time
     */
    public record Row(String task, String method, int solved, int instances, double conflicts, double millis) {

        @Override
        public String toString() {
            return String.format("%s/%s %d/%d work=%.0f %.1fms", task, method, solved, instances, conflicts, millis);
        }
    }

    private static Row cdclRow(String task, String name, List<ConstraintInstances.Cnf> fs, CdclSolver.Options o) {
        int solved = 0;
        double conf = 0;
        double ms = 0;
        for (ConstraintInstances.Cnf f : fs) {
            long t0 = System.nanoTime();
            CdclSolver s = CdclSolver.of(f.variables(), f.clauses(), o);
            CdclSolver.Status st = s.solve();
            ms += (System.nanoTime() - t0) / 1e6;
            if (st == CdclSolver.Status.SAT && !RandomSatFormulas.satisfies(f.clauses(), shift(s.model()))) {
                throw new IllegalStateException("CDCL model does not satisfy the formula");
            }
            solved += st == CdclSolver.Status.UNKNOWN ? 0 : 1;
            conf += s.stats().conflicts();
        }
        return new Row(task, name, solved, fs.size(), conf / fs.size(), ms / fs.size());
    }

    private static boolean[] shift(boolean[] m) {
        boolean[] out = new boolean[m.length + 1];
        System.arraycopy(m, 0, out, 1, m.length);
        return out;
    }

    private static Row dpllRow(String task, List<ConstraintInstances.Cnf> fs) {
        double ms = 0;
        for (ConstraintInstances.Cnf f : fs) {
            long t0 = System.nanoTime();
            RandomSatFormulas.solve(f.variables(), f.clauses());
            ms += (System.nanoTime() - t0) / 1e6;
        }
        return new Row(task, "DPLL (RandomSatFormulas)", fs.size(), fs.size(), 0, ms / fs.size());
    }

    private static Row walkRow(String task, List<ConstraintInstances.Cnf> fs, int flips, long seed) {
        int solved = 0;
        double work = 0;
        double ms = 0;
        SplittableRandom r = new SplittableRandom(seed);
        for (ConstraintInstances.Cnf f : fs) {
            long t0 = System.nanoTime();
            int k = GraphReasoning.walkSatFlips(f.formula(), flips, r);
            ms += (System.nanoTime() - t0) / 1e6;
            solved += k >= 0 ? 1 : 0;
            work += k >= 0 ? k : flips;
        }
        return new Row(task, "WalkSAT (flips; SAT only)", solved, fs.size(), work / fs.size(), ms / fs.size());
    }

    /**
     * 13.2.9: CDCL against DPLL and WalkSAT.
     *
     * @param sizes random 3-SAT sizes for CDCL
     * @param dpllMax the largest size given to DPLL (it has no budget)
     * @param perSize formulas per size
     * @param holes pigeonhole sizes
     * @param seed the seed
     * @return rows
     */
    public static List<Row> cdcl(int[] sizes, int dpllMax, int perSize, int[] holes, long seed) {
        List<Row> rows = new ArrayList<>();
        CdclSolver.Options o = CdclSolver.Options.defaults().withConflictLimit(200_000);
        for (int n : sizes) {
            List<ConstraintInstances.Cnf> fs = new ArrayList<>();
            for (int i = 0; i < perSize; i++) {
                fs.add(ConstraintInstances.randomKSat(n, 3, 4.26, seed + 100L * n + i));
            }
            String t = "3-SAT n=" + n;
            rows.add(cdclRow(t, "CDCL", fs, o));
            if (n <= dpllMax) {
                rows.add(dpllRow(t, fs));
            }
            rows.add(walkRow(t, fs, 20 * n * 10, seed));
        }
        for (int h : holes) {
            List<ConstraintInstances.Cnf> fs = List.of(ConstraintInstances.pigeonhole(h));
            rows.add(cdclRow("PHP-" + h, "CDCL", fs, o));
            if (h <= 6) {
                rows.add(dpllRow("PHP-" + h, fs));
            }
        }
        return rows;
    }

    /**
     * 13.2.9 ablation: every feature switched off in turn on structured instances.
     *
     * @param graphN vertices of the 3-colouring CNFs
     * @param perFamily instances per family
     * @param packages packages of the Logistics planning CNFs (3 cities)
     * @param withDpll whether to add the (unbudgeted, slow) DPLL rows
     * @param seed the seed
     * @return rows (work = conflicts)
     */
    public static List<Row> ablation(int graphN, int perFamily, int packages, boolean withDpll, long seed) {
        Map<String, List<ConstraintInstances.Cnf>> fam = new LinkedHashMap<>();
        List<ConstraintInstances.Cnf> col = new ArrayList<>();
        for (int i = 0; i < perFamily; i++) {
            col.add(ConstraintInstances.colouringCnf(graphN, ConstraintInstances.randomGraph(graphN, 4.6, seed + i),
                3));
        }
        fam.put("3-COL CNF n=" + graphN, col);
        List<ConstraintInstances.Cnf> plan = new ArrayList<>();
        for (int i = 0; i < perFamily; i++) {
            StripsTask t = PlanningDomains.logistics(3, 1, packages, false, seed + i);
            SatPlanner.Result r = SatPlanner.plan(t, SatPlanner.Encoding.FORALL_STEP, false, 30, 100_000);
            SatPlanner.Cnf c = SatPlanner.encode(t, Math.max(0, r.makespan() - 1), SatPlanner.Encoding.FORALL_STEP);
            plan.add(new ConstraintInstances.Cnf(c.variables(), c.clauses(), "plan-unsat"));
        }
        fam.put("logistics SATPlan h*-1 (UNSAT)", plan);
        CdclSolver.Options base = CdclSolver.Options.defaults().withConflictLimit(100_000);
        List<Row> rows = new ArrayList<>();
        for (Map.Entry<String, List<ConstraintInstances.Cnf>> e : fam.entrySet()) {
            rows.add(cdclRow(e.getKey(), "CDCL full", e.getValue(), base));
            rows.add(cdclRow(e.getKey(), "no VSIDS", e.getValue(), base.withVsids(false)));
            rows.add(cdclRow(e.getKey(), "no restarts", e.getValue(), base.withRestarts(CdclSolver.Restarts.NONE)));
            rows.add(cdclRow(e.getKey(), "glucose restarts", e.getValue(),
                base.withRestarts(CdclSolver.Restarts.GLUCOSE)));
            rows.add(cdclRow(e.getKey(), "no phase saving", e.getValue(), base.withPhaseSaving(false)));
            rows.add(cdclRow(e.getKey(), "no clause deletion", e.getValue(), base.withDeletion(false)));
            if (withDpll) {
                rows.add(dpllRow(e.getKey(), e.getValue()));
            }
        }
        return rows;
    }

    /**
     * NeuroSAT as a SAT/UNSAT classifier (the decision-only baseline).
     *
     * @param trainN variables of the training pairs
     * @param testN variables of the test pairs
     * @param steps training steps (one SR pair each)
     * @param seed the seed
     * @return test accuracy
     */
    public static double neuroSat(int trainN, int testN, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        NeuroSat net = new NeuroSat(16, r);
        Optimizer opt = Optimizer.adam(2e-3f);
        for (int s = 0; s < steps; s++) {
            RandomSatFormulas.Formula[] pair = RandomSatFormulas.srPair(trainN, r);
            NeuroSat.Batch b = NeuroSat.batch(List.of(pair[0], pair[1]));
            Var logit = net.logits(b, 8);
            Var y = Var.of(Tensor.of(Shape.of(2), new float[] {1, 0}));
            Var loss = logit.softplus().sub(y.mul(logit)).mean();
            loss.backward();
            opt.step(net.parameters());
        }
        int correct = 0;
        int total = 0;
        for (int i = 0; i < 20; i++) {
            RandomSatFormulas.Formula[] pair = RandomSatFormulas.srPair(testN, r);
            float[] lg = Var.noGrad(() -> net.logits(NeuroSat.batch(List.of(pair[0], pair[1])), 16)).value()
                .toArray();
            correct += (lg[0] > 0 ? 1 : 0) + (lg[1] <= 0 ? 1 : 0);
            total += 2;
        }
        return correct / (double) total;
    }

    /**
     * 13.2.10: RLAF guidance.
     *
     * @param trainN graph vertices of the training CNFs
     * @param testN graph vertices of the larger test CNFs
     * @param train training formulas
     * @param test test formulas per size
     * @param iterations RL passes
     * @param group samples per formula
     * @param seed the seed
     * @return per test size, method to {@code {mean decisions / plain, median ratio}}
     */
    public static Map<String, double[]> rlaf(int trainN, int testN, int train, int test, int iterations, int group,
            long seed) {
        CdclSolver.Options o = CdclSolver.Options.defaults().withConflictLimit(50_000);
        List<RandomSatFormulas.Formula> tr = new ArrayList<>();
        for (int i = 0; i < train; i++) {
            tr.add(ConstraintInstances.colouringCnf(trainN, ConstraintInstances.randomGraph(trainN, 4.4, seed + i), 3)
                .formula());
        }
        RlafGuidance rl = new RlafGuidance(16, 6, seed);
        rl.trainRl(tr, iterations, group, 3e-3f, o, seed);
        RlafGuidance sup = new RlafGuidance(16, 6, seed + 1);
        sup.trainSupervised(tr, 15, 3e-3f, o);
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int n : new int[] {trainN, testN}) {
            List<RandomSatFormulas.Formula> te = new ArrayList<>();
            for (int i = 0; i < test; i++) {
                te.add(ConstraintInstances.colouringCnf(n, ConstraintInstances.randomGraph(n, 4.4, seed + 900 + i), 3)
                    .formula());
            }
            double[][] ratio = new double[3][test];
            double plainSum = 0;
            double[] sums = new double[3];
            for (int i = 0; i < test; i++) {
                RandomSatFormulas.Formula f = te.get(i);
                double plain = Math.max(1, RlafGuidance.cost(f, null, o)[0]);
                plainSum += plain;
                RlafGuidance.Guidance[] gs = {rl.guidance(f), sup.supervisedGuidance(f),
                    RlafGuidance.random(f.variables(), seed + i)};
                for (int k = 0; k < 3; k++) {
                    double c = RlafGuidance.cost(f, gs[k], o)[0];
                    sums[k] += c;
                    ratio[k][i] = c / plain;
                }
            }
            String[] names = {"RLAF", "supervised backbone", "random weights"};
            for (int k = 0; k < 3; k++) {
                Arrays.sort(ratio[k]);
                out.put(String.format("n=%d %s", n, names[k]), new double[] {sums[k] / plainSum,
                    ratio[k][test / 2]});
            }
        }
        return out;
    }

    /**
     * 13.2.16: planning as SAT.
     *
     * @param tasks the tasks
     * @param dpllTasks how many of the first tasks also get the DPLL baseline
     * @return rows; work = makespan for SATPlan and plan length for the search planners
     */
    public static List<Row> planning(Map<String, StripsTask> tasks, int dpllTasks) {
        List<Row> rows = new ArrayList<>();
        int k = 0;
        for (Map.Entry<String, StripsTask> e : tasks.entrySet()) {
            StripsTask t = e.getValue();
            String n = e.getKey();
            for (SatPlanner.Encoding enc : SatPlanner.Encoding.values()) {
                SatPlanner.Result r = SatPlanner.plan(t, enc, false, 60, 200_000);
                rows.add(new Row(n, "SATPlan " + enc + " linear (len " + r.plan().size() + ")", r.found() ? 1 : 0, 1,
                    r.makespan(), r.millis()));
            }
            SatPlanner.Result d = SatPlanner.plan(t, SatPlanner.Encoding.FORALL_STEP, true, 64, 200_000);
            rows.add(new Row(n, "SATPlan FORALL_STEP doubling (len " + d.plan().size() + ")", d.found() ? 1 : 0, 1,
                d.makespan(), d.millis()));
            StateSpaceSearch.Result ff = StateSpaceSearch.search(t, StateSpaceSearch.Method.GBFS,
                List.of(new RelaxationHeuristics(t, RelaxationHeuristics.Kind.H_FF)), false, 1, 100_000);
            rows.add(new Row(n, "GBFS h_FF", ff.found() ? 1 : 0, 1, ff.plan().size(), ff.millis()));
            StateSpaceSearch.Result lm = StateSpaceSearch.astar(t, Landmarks.lmCut(t), 100_000);
            rows.add(new Row(n, "A* LM-cut (optimal length)", lm.found() ? 1 : 0, 1, lm.plan().size(), lm.millis()));
            if (k++ < dpllTasks) {
                SatPlanner.Result dp = SatPlanner.plan(t, SatPlanner.Encoding.FORALL_STEP, false, 60, (nv, cs) -> {
                    boolean[] m = RandomSatFormulas.solve(nv, cs);
                    return m == null ? null : Arrays.copyOfRange(m, 1, m.length);
                }, new long[1]);
                rows.add(new Row(n, "SATPlan FORALL_STEP with DPLL", dp.found() ? 1 : 0, 1, dp.makespan(),
                    dp.millis()));
            }
        }
        return rows;
    }

    /**
     * Prints the tables.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        cdcl(new int[] {50, 100, 150, 200, 250}, 75, 5, new int[] {5, 6, 7, 8}, 1).forEach(System.out::println);
        ablation(120, 4, 5, true, 1).forEach(System.out::println);
        System.out.println("NeuroSAT accuracy " + neuroSat(10, 20, 400, 1));
        rlaf(20, 40, 16, 16, 6, 8, 1).forEach((k, v) -> System.out.println(k + " " + Arrays.toString(v)));
        Map<String, StripsTask> tasks = new LinkedHashMap<>();
        tasks.put("gripper-4", PlanningDomains.gripper(4));
        tasks.put("logistics-2c-4", PlanningDomains.logistics(2, 1, 4, false, 1));
        tasks.put("blocksworld-6", PlanningDomains.blocksworld(6, 1));
        planning(tasks, 1).forEach(System.out::println);
    }
}
