package io.github.zzuegg.brainiac.experiments.combinatorial;

import io.github.zzuegg.brainiac.experimental.combopt.ClassicHeuristics;
import io.github.zzuegg.brainiac.experimental.combopt.GFlowNetCo;
import io.github.zzuegg.brainiac.experimental.combopt.QuboGnn;
import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances;
import io.github.zzuegg.brainiac.experimental.ml.SimulatedBifurcation;
import io.github.zzuegg.brainiac.experimental.train.GFlowNet;
import io.github.zzuegg.brainiac.experiments.exotic.ProbabilisticBits;
import io.github.zzuegg.brainiac.experiments.exotic.ProbabilisticBits.IsingSolver;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Unsupervised GNNs and GFlowNets against Ising solvers and greedy rules on MaxCut and
 * maximum independent set (items 11.2.8 and 11.2.9).
 *
 * <ul>
 *   <li>{@link #maxCut}: ballistic, adiabatic and discrete simulated bifurcation, p-bit
 *       annealing, simulated annealing and the spectral cut (all through
 *       {@link ProbabilisticBits#ising}), greedy with one-flip local search, PI-GNN
 *       ({@link QuboGnn#trainInstance}) and Erdős goes neural ({@link QuboGnn#trainAcross})
 *       with threshold and conditional-expectation decoding;</li>
 *   <li>{@link #independentSet}: greedy MIS, simulated annealing and ballistic SB on the MIS
 *       QUBO (via {@link QuboGnn#toIsing}, repaired to independence), PI-GNN and Erdős;</li>
 *   <li>{@link #gflownets}: FL-DB ({@link GFlowNetCo}) against the wave 6 trajectory-balance
 *       {@link GFlowNet} at equal training trajectories.</li>
 * </ul>
 * Every value is reported as a ratio to the best known (the exact optimum for small graphs,
 * else the best value any method or the reference search found).
 */
public final class GraphCoShootout {

    /**
     * One method's score.
     *
     * @param method the method
     * @param ratio the mean value over the best known (1 is optimal)
     * @param millis mean wall time per graph
     */
    public record Row(String method, double ratio, double millis) {
        @Override
        public String toString() {
            return String.format("%s %.3f (%.1f ms)", method, ratio, millis);
        }
    }

    private GraphCoShootout() {
    }

    private interface Solver {
        double value(double[][] graph, int index);
    }

    private static List<Row> table(List<double[][]> graphs, Map<String, Solver> solvers, double[] reference) {
        int g = graphs.size();
        Map<String, double[]> values = new LinkedHashMap<>();
        Map<String, Double> times = new LinkedHashMap<>();
        double[] best = reference.clone();
        for (Map.Entry<String, Solver> e : solvers.entrySet()) {
            double[] v = new double[g];
            long t0 = System.nanoTime();
            for (int i = 0; i < g; i++) {
                v[i] = e.getValue().value(graphs.get(i), i);
                best[i] = Math.max(best[i], v[i]);
            }
            times.put(e.getKey(), (System.nanoTime() - t0) / 1e6 / g);
            values.put(e.getKey(), v);
        }
        List<Row> rows = new ArrayList<>();
        for (String k : values.keySet()) {
            double r = 0;
            for (int i = 0; i < g; i++) {
                r += values.get(k)[i] / best[i];
            }
            rows.add(new Row(k, r / g, times.get(k)));
        }
        return rows;
    }

    /**
     * MaxCut on test graphs; the across-instance GNN trains on {@code train}.
     *
     * @param graphs test graphs
     * @param train training graphs for Erdős goes neural
     * @param gnnSteps PI-GNN steps per instance
     * @param solverEffort integration steps or sweeps of the Ising solvers
     * @param seed the seed
     * @return one row per method
     */
    public static List<Row> maxCut(List<double[][]> graphs, List<double[][]> train, int gnnSteps, int solverEffort,
            long seed) {
        double[] ref = new double[graphs.size()];
        for (int i = 0; i < ref.length; i++) {
            ref[i] = CombinatorialInstances.maxCut(graphs.get(i), seed).value();
        }
        QuboGnn erdos = QuboGnn.of(QuboGnn.Problem.MAX_CUT, 0, 32, seed);
        erdos.trainAcross(train, 10, 3e-3f);
        Map<String, Solver> s = new LinkedHashMap<>();
        for (IsingSolver solver : IsingSolver.values()) {
            s.put(solver.name(), (w, i) -> ProbabilisticBits.ising(w, true, solver, solver == IsingSolver.SPECTRAL ? 1 : 4,
                    solverEffort, seed + i).best());
        }
        s.put("greedy+local search", (w, i) -> ClassicHeuristics.cut(w, ClassicHeuristics.greedyMaxCut(w)));
        double[][] cache = new double[graphs.size()][];
        s.put("PI-GNN threshold", (w, i) -> {
            cache[i] = QuboGnn.of(QuboGnn.Problem.MAX_CUT, 0, 32, seed + i).trainInstance(w, 16, gnnSteps, 1e-2f);
            return ClassicHeuristics.cut(w, QuboGnn.threshold(QuboGnn.Problem.MAX_CUT, w, cache[i]));
        });
        s.put("PI-GNN cond. expectation", (w, i) -> ClassicHeuristics.cut(w, QuboGnn.conditionalExpectation(
                QuboGnn.qubo(QuboGnn.Problem.MAX_CUT, w, 0), cache[i])));
        s.put("Erdos threshold", (w, i) -> ClassicHeuristics.cut(w,
                QuboGnn.threshold(QuboGnn.Problem.MAX_CUT, w, erdos.predict(w, 1000 + i))));
        s.put("Erdos cond. expectation", (w, i) -> ClassicHeuristics.cut(w, QuboGnn.conditionalExpectation(
                QuboGnn.qubo(QuboGnn.Problem.MAX_CUT, w, 0), erdos.predict(w, 1000 + i))));
        return table(graphs, s, ref);
    }

    /**
     * Maximum independent set on test graphs.
     *
     * @param graphs test graphs
     * @param train training graphs for Erdős goes neural
     * @param gnnSteps PI-GNN steps per instance
     * @param solverEffort sweeps or steps of the Ising solvers
     * @param seed the seed
     * @return one row per method
     */
    public static List<Row> independentSet(List<double[][]> graphs, List<double[][]> train, int gnnSteps,
            int solverEffort, long seed) {
        double penalty = 2;
        double[] ref = new double[graphs.size()];
        for (int i = 0; i < ref.length; i++) {
            ref[i] = CombinatorialInstances.maxIndependentSet(graphs.get(i)).value();
        }
        QuboGnn erdos = QuboGnn.of(QuboGnn.Problem.MIS, penalty, 32, seed);
        erdos.trainAcross(train, 10, 3e-3f);
        Map<String, Solver> s = new LinkedHashMap<>();
        s.put("greedy MIS", (a, i) -> size(ClassicHeuristics.greedyMis(a)));
        s.put("SIMULATED_ANNEALING", (a, i) -> {
            SimulatedBifurcation sb = ising(QuboGnn.qubo(QuboGnn.Problem.MIS, a, penalty));
            return size(repair(a, sb.anneal(solverEffort, 0.1, 5, seed + i).spins()));
        });
        s.put("SB_BALLISTIC", (a, i) -> {
            SimulatedBifurcation sb = ising(QuboGnn.qubo(QuboGnn.Problem.MIS, a, penalty));
            int best = 0;
            for (int run = 0; run < 4; run++) {
                best = Math.max(best, size(repair(a, sb.solve(SimulatedBifurcation.Variant.BALLISTIC, solverEffort, 1.0,
                        seed + i + 1000L * run).spins())));
            }
            return best;
        });
        double[][] cache = new double[graphs.size()][];
        s.put("PI-GNN threshold", (a, i) -> {
            cache[i] = QuboGnn.of(QuboGnn.Problem.MIS, penalty, 32, seed + i).trainInstance(a, 16, gnnSteps, 1e-2f);
            return size(QuboGnn.threshold(QuboGnn.Problem.MIS, a, cache[i]));
        });
        s.put("PI-GNN cond. expectation", (a, i) -> size(QuboGnn.conditionalExpectation(
                QuboGnn.qubo(QuboGnn.Problem.MIS, a, penalty), cache[i])));
        s.put("Erdos threshold", (a, i) -> size(QuboGnn.threshold(QuboGnn.Problem.MIS, a, erdos.predict(a, 1000 + i))));
        s.put("Erdos cond. expectation", (a, i) -> size(QuboGnn.conditionalExpectation(
                QuboGnn.qubo(QuboGnn.Problem.MIS, a, penalty), erdos.predict(a, 1000 + i))));
        return table(graphs, s, ref);
    }

    private static SimulatedBifurcation ising(double[][] q) {
        double[][] jh = QuboGnn.toIsing(q);
        int n = q.length;
        return SimulatedBifurcation.of(Arrays.copyOf(jh, n), jh[n]);
    }

    private static int[] repair(double[][] a, int[] spins) {
        double[] p = new double[spins.length];
        for (int i = 0; i < p.length; i++) {
            p[i] = spins[i] > 0 ? 1 : 0;
        }
        return QuboGnn.threshold(QuboGnn.Problem.MIS, a, p);
    }

    private static int size(int[] x) {
        return Arrays.stream(x).sum();
    }

    /**
     * Training graphs or test graphs: Erdős–Rényi or 3-regular.
     *
     * @param count how many
     * @param n nodes
     * @param p the edge probability (0 for 3-regular)
     * @param seed the seed
     * @return the graphs
     */
    public static List<double[][]> graphs(int count, int n, double p, long seed) {
        List<double[][]> out = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            out.add(p > 0 ? CombinatorialInstances.erdosRenyi(n, p, seed + i)
                    : CombinatorialInstances.randomRegular(n, 3, seed + i));
        }
        return out;
    }

    /**
     * A GFlowNet comparison on one graph.
     *
     * @param tb trajectory balance on the problem
     * @param flDb the forward-looking sampler
     * @param bestEnergy the best known energy
     * @param distinct distinct samples within 95% of the best among 32 (TB, FL-DB)
     * @param toReach training trajectories until a sample reached 95% of the best (TB, FL-DB;
     *     -1 if never)
     * @param bestOf32 the best of 32 samples after training, as a ratio to the best (TB, FL-DB)
     */
    public record GflowResult(double tb, double flDb, double bestEnergy, int[] distinct, long[] toReach,
            double[] bestOf32) {
        @Override
        public String toString() {
            return String.format("best of 32: TB %.3f, FL-DB %.3f; distinct >=95%%: TB %d, FL-DB %d; trajectories to 95%%: "
                    + "TB %d, FL-DB %d", bestOf32[0], bestOf32[1], distinct[0], distinct[1], toReach[0], toReach[1]);
        }
    }

    /**
     * Trains trajectory balance and FL-DB on the same problem with equal trajectories.
     *
     * @param problem the problem
     * @param bestEnergy the best known energy (negative)
     * @param steps training steps
     * @param batch trajectories per step
     * @param seed the seed
     * @return the comparison
     */
    public static GflowResult gflownets(GFlowNetCo.EnergyProblem problem, double bestEnergy, int steps, int batch,
            long seed) {
        GFlowNet tb = GFlowNet.of(problem, 64, 1e-3f, seed).exploration(0.05);
        GFlowNetCo fl = GFlowNetCo.of(problem, 64, 1e-3f, seed);
        SplittableRandom r1 = new SplittableRandom(seed);
        SplittableRandom r2 = new SplittableRandom(seed);
        long[] reach = {-1, -1};
        for (int s = 0; s < steps; s++) {
            tb.trainStep(batch, r1);
            fl.trainStep(batch, 0.05, r2);
            if (reach[0] < 0 && best(GFlowNetCo.energies(tb.sample(4, 0, r1), problem)) <= 0.95 * bestEnergy) {
                reach[0] = (long) (s + 1) * batch;
            }
            if (reach[1] < 0 && best(GFlowNetCo.energies(fl.sample(4, 0, r2), problem)) <= 0.95 * bestEnergy) {
                reach[1] = (long) (s + 1) * batch;
            }
        }
        List<GFlowNet.Trajectory> a = tb.sample(32, 0, r1);
        List<GFlowNet.Trajectory> b = fl.sample(32, 0, r2);
        double ea = best(GFlowNetCo.energies(a, problem));
        double eb = best(GFlowNetCo.energies(b, problem));
        return new GflowResult(ea, eb, bestEnergy,
                new int[] {GFlowNetCo.distinctNearOptimal(a, problem, bestEnergy, 0.95),
                    GFlowNetCo.distinctNearOptimal(b, problem, bestEnergy, 0.95)},
                reach, new double[] {ea / bestEnergy, eb / bestEnergy});
    }

    private static double best(double[] energies) {
        return Arrays.stream(energies).min().orElse(0);
    }
}
