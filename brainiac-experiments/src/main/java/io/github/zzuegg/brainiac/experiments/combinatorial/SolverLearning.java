package io.github.zzuegg.brainiac.experiments.combinatorial;

import io.github.zzuegg.brainiac.experimental.combopt.BipartiteBranchingGnn;
import io.github.zzuegg.brainiac.experimental.combopt.BranchAndBound;
import io.github.zzuegg.brainiac.experimental.combopt.LearnedWalkSat;
import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances;
import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances.Ilp;
import io.github.zzuegg.brainiac.experimental.data.RandomSatFormulas;
import io.github.zzuegg.brainiac.experiments.graphs.GraphReasoning;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.LongFunction;

/**
 * Learning inside exact and local solvers (items 11.2.10 and 11.2.11): a bipartite GNN that
 * imitates strong branching in {@link BranchAndBound}, and a GNN that picks WalkSAT's flips.
 */
public final class SolverLearning {

    private SolverLearning() {
    }

    /**
     * A branching rule's mean effort on a test set.
     *
     * @param rule the rule's name
     * @param nodes mean branch-and-bound nodes
     * @param lpSolves mean LP solves (strong-branching probes included)
     * @param millis mean wall time
     * @param solved share finished within the node limit at the optimum
     */
    public record BranchRow(String rule, double nodes, double lpSolves, double millis, double solved) {
        @Override
        public String toString() {
            return String.format("%s nodes %.1f, LPs %.1f, %.1f ms, solved %.2f", rule, nodes, lpSolves, millis, solved);
        }
    }

    /**
     * The branching comparison.
     *
     * @param rows one row per rule
     * @param samples strong-branching samples used for imitation
     * @param agreement the trained GNN's top-1 agreement with strong branching on its samples
     */
    public record Branching(List<BranchRow> rows, int samples, double agreement) {
        /**
         * @param rule a rule's name
         * @return its row
         */
        public BranchRow row(String rule) {
            return rows.stream().filter(r -> r.rule().equals(rule)).findFirst().orElseThrow();
        }
    }

    /**
     * Trains {@link BipartiteBranchingGnn} on strong-branching samples recorded on small
     * instances, then compares the rules (best-bound search) on larger ones.
     *
     * @param trainGenerator the (smaller) training instances by seed
     * @param generator test instances by seed
     * @param trainInstances instances to record samples on
     * @param testInstances test instances (seeds offset by 1000)
     * @param epochs imitation passes
     * @param seed the seed
     * @return the comparison
     */
    public static Branching branching(LongFunction<Ilp> trainGenerator, LongFunction<Ilp> generator,
            int trainInstances, int testInstances, int epochs, long seed) {
        List<BranchAndBound.Sample> samples = new ArrayList<>();
        for (int i = 0; i < trainInstances; i++) {
            Ilp ilp = trainGenerator.apply(seed * 7919 + i);
            samples.addAll(BranchAndBound.of(BranchAndBound.Rule.STRONG).bestFirst(true).record(true).solve(ilp).samples());
        }
        BipartiteBranchingGnn gnn = new BipartiteBranchingGnn(16, seed);
        double agreement = gnn.train(samples, epochs, 3e-3f, new SplittableRandom(seed));
        List<Ilp> test = new ArrayList<>();
        for (int i = 0; i < testInstances; i++) {
            test.add(generator.apply(seed * 7919 + 1000 + i));
        }
        List<BranchRow> rows = new ArrayList<>();
        for (BranchAndBound.Rule rule : BranchAndBound.Rule.values()) {
            double nodes = 0;
            double lps = 0;
            double ms = 0;
            double solved = 0;
            for (Ilp ilp : test) {
                BranchAndBound bb = BranchAndBound.of(rule).bestFirst(true).nodeLimit(5000);
                if (rule == BranchAndBound.Rule.LEARNED) {
                    bb.scorer(gnn);
                }
                BranchAndBound.Result r = bb.solve(ilp);
                nodes += r.nodes();
                lps += r.lpSolves();
                ms += r.nanos() / 1e6;
                solved += r.optimal() && Math.abs(r.objective() - ilp.optimum()) < 1e-6 ? 1 : 0;
            }
            int k = test.size();
            rows.add(new BranchRow(rule.name(), nodes / k, lps / k, ms / k, solved / k));
        }
        return new Branching(rows, samples.size(), agreement);
    }

    /**
     * Flip statistics of one local-search method on a test set.
     *
     * @param method the method
     * @param n the variable count
     * @param medianFlips median flips to a model (unsolved count as the budget)
     * @param solved share solved within the budget
     * @param millis mean wall time per formula
     */
    public record SatRow(String method, int n, double medianFlips, double solved, double millis) {
        @Override
        public String toString() {
            return String.format("%s n=%d median flips %.0f, solved %.2f, %.2f ms", method, n, medianFlips, solved, millis);
        }
    }

    /**
     * Trains {@link LearnedWalkSat} by REINFORCE with a size curriculum, then compares it with
     * WalkSAT and DPLL on held-out random 3-SAT formulas at the given sizes.
     *
     * @param curriculum training sizes, one stage each
     * @param stepsPerStage REINFORCE steps per stage
     * @param batch formulas per step
     * @param trainBudget flips per training episode
     * @param testN test sizes
     * @param tests test formulas per size
     * @param budget the test flip budget
     * @param ratio clauses per variable
     * @param seed the seed
     * @return one row per method and size
     */
    public static List<SatRow> walkSat(int[] curriculum, int stepsPerStage, int batch, int trainBudget, int[] testN,
            int tests, int budget, double ratio, long seed) {
        LearnedWalkSat policy = LearnedWalkSat.of(16, seed);
        Optimizer opt = Optimizer.adam(1e-2f);
        SplittableRandom r = new SplittableRandom(seed);
        long draw = seed * 104_729;
        for (int n : curriculum) {
            for (int s = 0; s < stepsPerStage; s++) {
                List<RandomSatFormulas.Formula> b = new ArrayList<>();
                for (int i = 0; i < batch; i++) {
                    b.add(CombinatorialInstances.random3Sat(n, ratio, draw++));
                }
                policy.trainStep(b, trainBudget, opt, r);
            }
        }
        List<SatRow> rows = new ArrayList<>();
        for (int n : testN) {
            List<RandomSatFormulas.Formula> fs = new ArrayList<>();
            for (int i = 0; i < tests; i++) {
                fs.add(CombinatorialInstances.random3Sat(n, ratio, seed * 31 + 50_000 + 1000L * n + i));
            }
            double[] learned = new double[tests];
            double[] walk = new double[tests];
            double[] random = new double[tests];
            int ls = 0;
            int ws = 0;
            int rs = 0;
            long tl = 0;
            long tw = 0;
            long tr = 0;
            long td = 0;
            for (int i = 0; i < tests; i++) {
                RandomSatFormulas.Formula f = fs.get(i);
                long t0 = System.nanoTime();
                LearnedWalkSat.Episode e = policy.solve(f, budget, new SplittableRandom(seed + i), false);
                long t1 = System.nanoTime();
                int w = GraphReasoning.walkSatFlips(f, budget, new SplittableRandom(seed + i));
                long t2 = System.nanoTime();
                int rw = randomWalk(f, budget, new SplittableRandom(seed + i));
                long t3 = System.nanoTime();
                RandomSatFormulas.solve(f.variables(), f.clauses());
                long t4 = System.nanoTime();
                tl += t1 - t0;
                tw += t2 - t1;
                tr += t3 - t2;
                td += t4 - t3;
                learned[i] = e.solved() ? e.flips() : budget;
                ls += e.solved() ? 1 : 0;
                walk[i] = w >= 0 ? w : budget;
                ws += w >= 0 ? 1 : 0;
                random[i] = rw >= 0 ? rw : budget;
                rs += rw >= 0 ? 1 : 0;
            }
            rows.add(new SatRow("learned WalkSAT", n, median(learned), ls / (double) tests, tl / 1e6 / tests));
            rows.add(new SatRow("WalkSAT p=0.5", n, median(walk), ws / (double) tests, tw / 1e6 / tests));
            rows.add(new SatRow("random walk", n, median(random), rs / (double) tests, tr / 1e6 / tests));
            rows.add(new SatRow("DPLL", n, 0, 1, td / 1e6 / tests));
        }
        return rows;
    }

    /**
     * Papadimitriou's random walk: flip a random variable of a random unsatisfied clause.
     *
     * @param f the formula
     * @param budget the flip budget
     * @param r the source
     * @return flips to a model, or -1
     */
    public static int randomWalk(RandomSatFormulas.Formula f, int budget, SplittableRandom r) {
        int n = f.variables();
        boolean[] a = new boolean[n + 1];
        for (int v = 1; v <= n; v++) {
            a[v] = r.nextBoolean();
        }
        for (int s = 0; s < budget; s++) {
            List<int[]> unsat = new ArrayList<>();
            for (int[] c : f.clauses()) {
                boolean ok = false;
                for (int lit : c) {
                    ok |= a[Math.abs(lit)] == lit > 0;
                }
                if (!ok) {
                    unsat.add(c);
                }
            }
            if (unsat.isEmpty()) {
                return s;
            }
            int[] c = unsat.get(r.nextInt(unsat.size()));
            int v = Math.abs(c[r.nextInt(c.length)]);
            a[v] = !a[v];
        }
        return RandomSatFormulas.satisfies(f.clauses(), a) ? budget : -1;
    }

    private static double median(double[] x) {
        double[] s = x.clone();
        Arrays.sort(s);
        int k = s.length;
        return k % 2 == 1 ? s[k / 2] : (s[k / 2 - 1] + s[k / 2]) / 2;
    }
}
