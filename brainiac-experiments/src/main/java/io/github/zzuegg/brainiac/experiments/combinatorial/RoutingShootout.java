package io.github.zzuegg.brainiac.experiments.combinatorial;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.combopt.ActiveSearch;
import io.github.zzuegg.brainiac.experimental.combopt.AttentionModel;
import io.github.zzuegg.brainiac.experimental.combopt.ClassicHeuristics;
import io.github.zzuegg.brainiac.experimental.combopt.EdgeHeatmapGcn;
import io.github.zzuegg.brainiac.experimental.combopt.LearnedTwoOpt;
import io.github.zzuegg.brainiac.experimental.combopt.PointerRouting;
import io.github.zzuegg.brainiac.experimental.combopt.Pomo;
import io.github.zzuegg.brainiac.experimental.combopt.PreferenceOptimization;
import io.github.zzuegg.brainiac.experimental.combopt.RoutingEnv;
import io.github.zzuegg.brainiac.experimental.combopt.SoftPermutationTsp;
import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances;
import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances.Tsp;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Neural routing solvers against the classical heuristics and exact optima (items 11.2.1 to
 * 11.2.7), on {@link CombinatorialInstances} Euclidean TSP (and CVRP) instances. Every row
 * reports the mean optimality gap in percent against the Held–Karp optimum (or the best-known
 * tour above 13 nodes) and the inference time per instance.
 *
 * <ul>
 *   <li>{@link #heuristics}: random tours, nearest neighbour, farthest insertion, 2-opt from
 *       random starts and from nearest neighbour, Or-opt, and their combination;</li>
 *   <li>{@link #pointerNetworks} (11.2.1): {@link PointerRouting.Supervised} on optimal tours
 *       against the actor-critic {@link PointerRouting} at equal instances;</li>
 *   <li>{@link #attentionModel} (11.2.2): {@link AttentionModel} with the greedy-rollout
 *       baseline, greedy and best-of-k sampled decoding;</li>
 *   <li>{@link #pomo} (11.2.3): POMO's shared baseline against the rollout baseline at equal
 *       training rollouts, with and without the x8 augmentation, and Sym-NCO on rotated
 *       instances;</li>
 *   <li>{@link #curve} (11.2.4): preference optimisation against POMO's REINFORCE at
 *       equal rollouts;</li>
 *   <li>{@link #activeSearch} (11.2.5): sampling, full active search and the three EAS
 *       variants at equal per-instance rollouts;</li>
 *   <li>{@link #improvement} (11.2.6): the learned 2-opt move selector against first- and
 *       best-improvement 2-opt and simulated annealing at equal move counts;</li>
 *   <li>{@link #heatmaps} (11.2.7): the supervised edge GCN with beam search and the
 *       unsupervised soft-permutation model against nearest neighbour.</li>
 * </ul>
 */
public final class RoutingShootout {

    /**
     * One method's scores.
     *
     * @param method the method
     * @param label the variant or test set
     * @param gap the mean optimality gap in percent
     * @param millis inference time per instance
     */
    public record Row(String method, String label, double gap, double millis) {
        @Override
        public String toString() {
            return String.format("%s%s %.1f%% (%.2f ms)", method, label.isEmpty() ? "" : "[" + label + "]", gap, millis);
        }
    }

    /** The entropy temperature of preference optimisation in {@link #curve}. */
    public static final float PREFERENCE_ALPHA = 0.1f;

    private RoutingShootout() {
    }

    /**
     * Wraps TSP instances as environments.
     *
     * @param set the instances
     * @return the environments
     */
    public static List<RoutingEnv> envs(List<Tsp> set) {
        List<RoutingEnv> out = new ArrayList<>();
        for (Tsp t : set) {
            out.add(RoutingEnv.tsp(t.coords()));
        }
        return out;
    }

    /**
     * Random training instances.
     *
     * @param count how many
     * @param n nodes
     * @param random the source
     * @return the environments
     */
    public static List<RoutingEnv> randomBatch(int count, int n, SplittableRandom random) {
        List<RoutingEnv> out = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            out.add(RoutingEnv.tsp(CombinatorialInstances.points(n, false, random)));
        }
        return out;
    }

    /**
     * The mean gap of lengths against the instances' optima.
     *
     * @param lengths one length per instance
     * @param set the instances
     * @return the gap in percent
     */
    public static double gap(double[] lengths, List<Tsp> set) {
        double g = 0;
        for (int i = 0; i < lengths.length; i++) {
            g += lengths[i] / set.get(i).optimum() - 1;
        }
        return 100 * g / lengths.length;
    }

    private static Row timed(String method, String label, List<Tsp> set, ToDoubleFunction<Tsp> solver) {
        long t0 = System.nanoTime();
        double[] len = new double[set.size()];
        for (int i = 0; i < len.length; i++) {
            len[i] = solver.applyAsDouble(set.get(i));
        }
        return new Row(method, label, gap(len, set), (System.nanoTime() - t0) / 1e6 / set.size());
    }

    /**
     * The classical heuristics.
     *
     * @param set test instances
     * @param seed the seed of random starts
     * @return random, nearest neighbour, farthest insertion, 2-opt (random start), 2-opt
     *     (nearest-neighbour start), Or-opt (nearest-neighbour start), 2-opt + Or-opt
     */
    public static List<Row> heuristics(List<Tsp> set, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Row> rows = new ArrayList<>();
        rows.add(timed("random", "", set, t -> ClassicHeuristics.tourLength(t.distances(),
                ClassicHeuristics.randomTour(t.size(), r))));
        rows.add(timed("nearest neighbour", "", set, t -> ClassicHeuristics.tourLength(t.distances(),
                ClassicHeuristics.nearestNeighbour(t.distances(), 0))));
        rows.add(timed("farthest insertion", "", set, t -> ClassicHeuristics.tourLength(t.distances(),
                ClassicHeuristics.farthestInsertion(t.distances()))));
        rows.add(timed("2-opt", "random start", set, t -> ClassicHeuristics.twoOpt(t.distances(),
                ClassicHeuristics.randomTour(t.size(), r), false, 10_000).length()));
        rows.add(timed("2-opt", "NN start", set, t -> ClassicHeuristics.twoOpt(t.distances(),
                ClassicHeuristics.nearestNeighbour(t.distances(), 0), false, 10_000).length()));
        rows.add(timed("Or-opt", "NN start", set, t -> ClassicHeuristics.orOpt(t.distances(),
                ClassicHeuristics.nearestNeighbour(t.distances(), 0), 10_000).length()));
        rows.add(timed("2-opt+Or-opt", "NN start", set, t -> ClassicHeuristics.twoOptOrOpt(t.distances(),
                ClassicHeuristics.nearestNeighbour(t.distances(), 0)).length()));
        return rows;
    }

    // ---------------------------------------------------------------------------------------
    // 11.2.1 pointer networks
    // ---------------------------------------------------------------------------------------

    /**
     * Trains the supervised pointer network on Held–Karp tours and the actor-critic pointer
     * network on rewards, with the same instances, and tests both on each test set.
     *
     * @param n the training size (at most 13, for exact labels)
     * @param steps training steps
     * @param batch instances per step
     * @param tests the test sets (the first should have the training size)
     * @param seed the seed
     * @return supervised and actor-critic rows per test set (labelled by size)
     */
    public static List<Row> pointerNetworks(int n, int steps, int batch, List<List<Tsp>> tests, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        PointerRouting.Supervised sup = new PointerRouting.Supervised(32, seed);
        PointerRouting ac = PointerRouting.of(32, seed);
        Optimizer supOpt = Optimizer.adam(2e-3f);
        Optimizer acOpt = Optimizer.adam(2e-3f);
        for (int s = 0; s < steps; s++) {
            List<RoutingEnv> envs = randomBatch(batch, n, r);
            int[][] tours = new int[batch][];
            for (int i = 0; i < batch; i++) {
                tours[i] = CombinatorialInstances.heldKarp(envs.get(i).distances());
            }
            sup.step(envs, tours, supOpt);
            ac.actorCriticStep(envs, acOpt, r);
        }
        List<Row> rows = new ArrayList<>();
        for (List<Tsp> test : tests) {
            String label = "TSP-" + test.get(0).size();
            List<RoutingEnv> envs = envs(test);
            long t0 = System.nanoTime();
            double[] a = sup.greedyLengths(envs);
            long t1 = System.nanoTime();
            double[] b = ac.greedyLengths(envs);
            long t2 = System.nanoTime();
            rows.add(new Row("pointer supervised", label, gap(a, test), (t1 - t0) / 1e6 / test.size()));
            rows.add(new Row("pointer actor-critic", label, gap(b, test), (t2 - t1) / 1e6 / test.size()));
        }
        return rows;
    }

    // ---------------------------------------------------------------------------------------
    // 11.2.2 attention model
    // ---------------------------------------------------------------------------------------

    /**
     * Trains an attention model with the greedy-rollout baseline.
     *
     * @param n training size
     * @param epochs epochs (the baseline test runs after each)
     * @param stepsPerEpoch steps per epoch
     * @param batch instances per step
     * @param seed the seed
     * @return the trained model
     */
    public static AttentionModel trainAttentionModel(int n, int epochs, int stepsPerEpoch, int batch, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        AttentionModel model = AttentionModel.of(RoutingEnv.Kind.TSP, 32, 4, 1, seed);
        AttentionModel.RolloutBaseline baseline = new AttentionModel.RolloutBaseline(model, randomBatch(64, n, r));
        Optimizer opt = Optimizer.adam(1e-3f);
        for (int e = 0; e < epochs; e++) {
            for (int s = 0; s < stepsPerEpoch; s++) {
                List<RoutingEnv> envs = randomBatch(batch, n, r);
                // during the first epoch the untrained copy is a poor baseline; Kool et al. use
                // an exponential moving average there, here the batch mean
                double[] b;
                AttentionModel.Rollout ro = model.rollout(envs, AttentionModel.Decode.SAMPLE, null, r);
                if (e == 0) {
                    b = new double[batch];
                    java.util.Arrays.fill(b, ro.meanLength());
                } else {
                    b = baseline.lengths(envs);
                }
                AttentionModel.step(AttentionModel.reinforceLoss(ro, b), model.parameters(), opt, 1.0);
            }
            baseline.update(model);
        }
        return model;
    }

    /**
     * Greedy and sampled decoding of an attention model.
     *
     * @param model the model
     * @param test the test set
     * @param samples samples per instance for the sampled row
     * @param seed the seed
     * @return greedy and best-of-{@code samples} rows
     */
    public static List<Row> attentionModel(AttentionModel model, List<Tsp> test, int samples, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        String label = "TSP-" + test.get(0).size();
        List<RoutingEnv> envs = envs(test);
        long t0 = System.nanoTime();
        double[] g = model.greedyLengths(envs);
        long t1 = System.nanoTime();
        double[] s = new double[test.size()];
        for (int i = 0; i < s.length; i++) {
            s[i] = model.bestOfSamples(envs.get(i), samples, r);
        }
        long t2 = System.nanoTime();
        return List.of(new Row("attention model greedy", label, gap(g, test), (t1 - t0) / 1e6 / test.size()),
                new Row("attention model sampled", label + " x" + samples, gap(s, test), (t2 - t1) / 1e6 / test.size()));
    }

    // ---------------------------------------------------------------------------------------
    // 11.2.3 POMO and Sym-NCO
    // ---------------------------------------------------------------------------------------

    /**
     * Trains a POMO policy.
     *
     * @param n training size
     * @param steps steps
     * @param instances instances per step (each rolled out from every node)
     * @param symWeight the Sym-NCO invariance weight (0 with {@code rotations = 0} is plain POMO)
     * @param rotations Sym-NCO rotations per instance
     * @param seed the seed
     * @return the trained POMO
     */
    public static Pomo trainPomo(int n, int steps, int instances, double symWeight, int rotations, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Pomo pomo = Pomo.of(AttentionModel.of(RoutingEnv.Kind.TSP, 32, 4, 1, seed)).symNco(symWeight, rotations);
        Optimizer opt = Optimizer.adam(1e-3f);
        for (int s = 0; s < steps; s++) {
            pomo.trainStep(randomBatch(instances, n, r), opt, r);
        }
        return pomo;
    }

    /**
     * Tests a POMO policy with and without augmentation.
     *
     * @param pomo the policy
     * @param name the row name
     * @param test the test set
     * @return no-augmentation and x8 rows
     */
    public static List<Row> pomo(Pomo pomo, String name, List<Tsp> test) {
        String label = "TSP-" + test.get(0).size();
        return List.of(
                timed(name, label, test, t -> RoutingEnv.tsp(t.coords()).length(pomo.solve(RoutingEnv.tsp(t.coords()), false))),
                timed(name, label + " x8", test, t -> RoutingEnv.tsp(t.coords())
                        .length(pomo.solve(RoutingEnv.tsp(t.coords()), true))));
    }

    /**
     * Rotated copies of a test set (random angles; optima rescaled).
     *
     * @param test the test set
     * @param seed the seed
     * @return the rotated set
     */
    public static List<Tsp> rotated(List<Tsp> test, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Tsp> out = new ArrayList<>();
        for (Tsp t : test) {
            out.add(CombinatorialInstances.rotated(t, 2 * Math.PI * r.nextDouble(), false));
        }
        return out;
    }

    // ---------------------------------------------------------------------------------------
    // 11.2.4 preference optimisation
    // ---------------------------------------------------------------------------------------

    /**
     * The gap curve of a trainer: the greedy POMO gap (no augmentation) on a test set every
     * {@code every} steps.
     *
     * @param n training size
     * @param steps steps
     * @param every evaluation interval
     * @param instances instances per step
     * @param preference {@code null} for POMO REINFORCE, else the preference model
     * @param localSearch 2-opt moves on the preference pairs (0 for none)
     * @param test the test set
     * @param seed the seed
     * @return the gaps and the trained policy
     */
    public static Curve curve(int n, int steps, int every, int instances,
            PreferenceOptimization.PreferenceModel preference, int localSearch, List<Tsp> test, long seed) {
        return curve(n, steps, every, instances, preference, localSearch, 0, 0, test, seed);
    }

    /**
     * A training curve and the trained policy.
     *
     * @param gaps the greedy gap at each evaluation
     * @param pomo the policy
     */
    public record Curve(double[] gaps, Pomo pomo) {
        /** @return the last gap */
        public double last() {
            return gaps[gaps.length - 1];
        }
    }

    /**
     * As {@link #curve(int, int, int, int, PreferenceOptimization.PreferenceModel, int, List, long)},
     * with Sym-NCO settings for REINFORCE.
     *
     * @param n training size
     * @param steps steps
     * @param every evaluation interval
     * @param instances instances per step
     * @param preference {@code null} for POMO REINFORCE, else the preference model
     * @param localSearch 2-opt moves on the preference pairs (0 for none)
     * @param symWeight the Sym-NCO invariance weight
     * @param rotations Sym-NCO rotations per instance
     * @param test the test set
     * @param seed the seed
     * @return the gaps and the trained policy
     */
    public static Curve curve(int n, int steps, int every, int instances,
            PreferenceOptimization.PreferenceModel preference, int localSearch, double symWeight, int rotations,
            List<Tsp> test, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Pomo pomo = Pomo.of(AttentionModel.of(RoutingEnv.Kind.TSP, 32, 4, 1, seed)).symNco(symWeight, rotations);
        PreferenceOptimization po = preference == null ? null
                : PreferenceOptimization.of(pomo, preference, PREFERENCE_ALPHA).localSearch(localSearch);
        Optimizer opt = Optimizer.adam(1e-3f);
        double[] out = new double[steps / every];
        for (int s = 0; s < steps; s++) {
            List<RoutingEnv> b = randomBatch(instances, n, r);
            if (po == null) {
                pomo.trainStep(b, opt, r);
            } else {
                po.trainStep(b, opt, r);
            }
            if ((s + 1) % every == 0) {
                out[(s + 1) / every - 1] = pomo(pomo, "", test).get(0).gap();
            }
        }
        return new Curve(out, pomo);
    }

    // ---------------------------------------------------------------------------------------
    // 11.2.5 active search
    // ---------------------------------------------------------------------------------------

    /**
     * Test-time search with a trained model at equal rollouts per instance.
     *
     * @param model the trained policy
     * @param test the test set
     * @param iterations iterations per instance
     * @param maxStarts POMO starts per iteration
     * @param seed the seed
     * @return one row per {@link ActiveSearch.Mode}
     */
    public static List<Row> activeSearch(AttentionModel model, List<Tsp> test, int iterations, int maxStarts,
            long seed) {
        List<Row> rows = new ArrayList<>();
        String label = "TSP-" + test.get(0).size();
        for (ActiveSearch.Mode mode : ActiveSearch.Mode.values()) {
            SplittableRandom r = new SplittableRandom(seed);
            ActiveSearch as = ActiveSearch.of(model, mode).maxStarts(maxStarts)
                    .learningRate(mode == ActiveSearch.Mode.FULL ? 5e-4f : 3e-3f);
            rows.add(timed(mode.name(), label + " " + iterations * Math.min(maxStarts, test.get(0).size()) + " rollouts",
                    test, t -> as.search(RoutingEnv.tsp(t.coords()), iterations, r).length()));
        }
        return rows;
    }

    // ---------------------------------------------------------------------------------------
    // 11.2.6 learned improvement
    // ---------------------------------------------------------------------------------------

    /**
     * The learned 2-opt selector against classical improvement at equal moves, all from
     * random starts.
     *
     * @param policy a trained selector
     * @param test the test set
     * @param moves the move budgets
     * @param seed the seed
     * @return rows for first- and best-improvement 2-opt, annealing and the learned policy per
     *     budget
     */
    public static List<Row> improvement(LearnedTwoOpt policy, List<Tsp> test, int[] moves, long seed) {
        List<Row> rows = new ArrayList<>();
        for (int m : moves) {
            String label = m + " moves";
            SplittableRandom r1 = new SplittableRandom(seed);
            rows.add(timed("2-opt first", label, test, t -> ClassicHeuristics.twoOpt(t.distances(),
                    ClassicHeuristics.randomTour(t.size(), r1), true, m).length()));
            SplittableRandom r2 = new SplittableRandom(seed);
            rows.add(timed("2-opt best", label, test, t -> ClassicHeuristics.twoOpt(t.distances(),
                    ClassicHeuristics.randomTour(t.size(), r2), false, m).length()));
            SplittableRandom r3 = new SplittableRandom(seed);
            rows.add(timed("annealing", label, test, t -> ClassicHeuristics.annealTwoOpt(t.distances(),
                    ClassicHeuristics.randomTour(t.size(), r3), m, 0.1, 0.002, seed).length()));
            SplittableRandom r4 = new SplittableRandom(seed);
            int[][] starts = new int[test.size()][];
            for (int i = 0; i < starts.length; i++) {
                starts[i] = ClassicHeuristics.randomTour(test.get(i).size(), r4);
            }
            long t0 = System.nanoTime();
            List<ClassicHeuristics.Improvement> imp = policy.improveAll(envs(test), starts, m,
                    new SplittableRandom(seed + 1));
            double[] len = new double[test.size()];
            for (int i = 0; i < len.length; i++) {
                len[i] = imp.get(i).length();
            }
            rows.add(new Row("learned 2-opt", label, gap(len, test), (System.nanoTime() - t0) / 1e6 / test.size()));
        }
        return rows;
    }

    // ---------------------------------------------------------------------------------------
    // 11.2.7 heatmaps
    // ---------------------------------------------------------------------------------------

    /**
     * Non-autoregressive heatmap solvers.
     *
     * @param gcn a trained supervised edge GCN
     * @param beam its beam width
     * @param soft a trained soft-permutation model
     * @param candidates its sampled candidates per instance
     * @param test the test set
     * @param seed the seed
     * @return edge GCN greedy and beam, soft-permutation single and best-of-candidates, and
     *     the same with 2-opt
     */
    public static List<Row> heatmaps(EdgeHeatmapGcn gcn, int beam, SoftPermutationTsp soft, int candidates,
            List<Tsp> test, long seed) {
        String label = "TSP-" + test.get(0).size();
        List<Row> rows = new ArrayList<>();
        rows.add(timed("edge GCN", label + " greedy", test, t -> len(t, gcn.decode(RoutingEnv.tsp(t.coords()), 1))));
        rows.add(timed("edge GCN", label + " beam " + beam, test, t -> len(t, gcn.decode(RoutingEnv.tsp(t.coords()), beam))));
        SplittableRandom r = new SplittableRandom(seed);
        rows.add(timed("soft permutation", label + " single", test, t -> len(t, soft.decode(RoutingEnv.tsp(t.coords()), 1, r))));
        rows.add(timed("soft permutation", label + " best of " + candidates, test,
                t -> len(t, soft.decode(RoutingEnv.tsp(t.coords()), candidates, r))));
        rows.add(timed("soft permutation + 2-opt", label, test, t -> ClassicHeuristics.twoOpt(t.distances(),
                soft.decode(RoutingEnv.tsp(t.coords()), candidates, r), false, 10_000).length()));
        return rows;
    }

    private static double len(Tsp t, int[] tour) {
        return ClassicHeuristics.tourLength(t.distances(), tour);
    }

    /**
     * Greedy lengths of a model on a set, without gradients.
     *
     * @param model the model
     * @param name the row name
     * @param test the set
     * @return the gap row
     */
    public static Row greedy(AttentionModel model, String name, List<Tsp> test) {
        long t0 = System.nanoTime();
        double[] g = Var.noGrad(() -> model.greedyLengths(envs(test)));
        return new Row(name, "TSP-" + test.get(0).size(), gap(g, test), (System.nanoTime() - t0) / 1e6 / test.size());
    }
}
