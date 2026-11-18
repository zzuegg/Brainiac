package io.github.zzuegg.brainiac.experiments.symbolic;

import io.github.zzuegg.brainiac.experimental.data.PlanningDomains;
import io.github.zzuegg.brainiac.experimental.symbolic.GeneralizedPlanner;
import io.github.zzuegg.brainiac.experimental.symbolic.LearnedPlanningHeuristic;
import io.github.zzuegg.brainiac.experimental.symbolic.PartialSpaceSearch;
import io.github.zzuegg.brainiac.experimental.symbolic.RankingHeuristic;
import io.github.zzuegg.brainiac.experimental.symbolic.RelaxationHeuristics;
import io.github.zzuegg.brainiac.experimental.symbolic.StateSpaceSearch;
import io.github.zzuegg.brainiac.experimental.symbolic.StripsTask;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Wave 13 items 13.2.4-13.2.6: learned planning that generalises across instance sizes. Every
 * learned component trains on optimal plans (A* with LM-cut) of small instances and is tested on
 * strictly larger ones.
 *
 * <ul>
 *   <li>{@link #heuristics} (13.2.4): GBFS with WL-GOOSE heuristics (ridge and GP on WL
 *       features), a GIN heuristic on the same graphs, h_FF and blind search: coverage,
 *       expansions, training time; and the GP's predictive variance on training-size and
 *       test-size states.</li>
 *   <li>{@link #ranking} (13.2.5): the ranking-trained linear heuristic against the regression
 *       one at equal features, and partial-space search with an action-set ranker against
 *       grounded GBFS (generated nodes) on high-branching Logistics and Ferry and on
 *       Blocksworld.</li>
 *   <li>{@link #generalisation} (13.2.6): the WL transition model (ridge and boosted trees) and
 *       the GNN policy run without search, against random actions, the learned heuristic with
 *       GBFS and h_FF GBFS: success rate and plan length relative to h_FF GBFS.</li>
 * </ul>
 */
public final class LearnedPlanning {

    private LearnedPlanning() {
    }

    /**
     * Training and test instances of one domain.
     *
     * @param train small instances
     * @param test larger instances
     */
    public record Split(List<StripsTask> train, List<StripsTask> test) {
    }

    /**
     * The domains.
     *
     * @param seed the seed
     * @return domain name to split
     */
    public static Map<String, Split> domains(long seed) {
        Map<String, Split> d = new LinkedHashMap<>();
        List<StripsTask> bw = new ArrayList<>();
        for (int n = 3; n <= 5; n++) {
            for (int i = 0; i < 4; i++) {
                bw.add(PlanningDomains.blocksworld(n, seed + 10 * n + i));
            }
        }
        d.put("blocksworld", new Split(bw, List.of(PlanningDomains.blocksworld(9, seed + 100),
            PlanningDomains.blocksworld(10, seed + 101))));
        d.put("gripper", new Split(List.of(PlanningDomains.gripper(2), PlanningDomains.gripper(3),
            PlanningDomains.gripper(4)), List.of(PlanningDomains.gripper(10), PlanningDomains.gripper(14))));
        List<StripsTask> ferry = new ArrayList<>();
        for (int c = 2; c <= 3; c++) {
            for (int i = 0; i < 3; i++) {
                ferry.add(PlanningDomains.ferry(3, c, seed + 20 * c + i));
            }
        }
        d.put("ferry", new Split(ferry, List.of(PlanningDomains.ferry(5, 8, seed + 200),
            PlanningDomains.ferry(6, 10, seed + 201))));
        d.put("visitall", new Split(List.of(PlanningDomains.visitAll(2), PlanningDomains.visitAll(3)),
            List.of(PlanningDomains.visitAll(6), PlanningDomains.visitAll(8))));
        return d;
    }

    /**
     * Logistics splits (one truck per city for training, several trucks and many packages for
     * testing: a high branching factor).
     *
     * @param seed the seed
     * @param trucksPerCity trucks per city in the test instances
     * @param packages packages in the test instances
     * @return the split
     */
    public static Split logistics(long seed, int trucksPerCity, int packages) {
        List<StripsTask> train = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            train.add(PlanningDomains.logistics(2, 1, 2, false, seed + i));
            train.add(PlanningDomains.logistics(2, 1, 3, false, seed + 10 + i));
        }
        return new Split(train, List.of(PlanningDomains.logistics(3, trucksPerCity, packages, false, seed + 50),
            PlanningDomains.logistics(3, trucksPerCity, packages + 1, false, seed + 51)));
    }

    /**
     * A row.
     *
     * @param domain the domain
     * @param method the planner
     * @param solved test instances solved
     * @param instances test instances
     * @param expanded mean expansions (solved)
     * @param generated mean generated nodes (solved)
     * @param cost mean plan cost (solved)
     * @param trainMillis training time
     */
    public record Row(String domain, String method, int solved, int instances, double expanded, double generated,
            double cost, double trainMillis) {

        @Override
        public String toString() {
            return String.format("%s/%s %d/%d exp=%.0f gen=%.0f cost=%.1f train=%.0fms", domain, method, solved,
                instances, expanded, generated, cost, trainMillis);
        }
    }

    private static Row searchRow(String domain, String method, List<StripsTask> test,
            java.util.function.Function<StripsTask, StateSpaceSearch.Evaluator> h, int budget, double trainMillis) {
        int solved = 0;
        double exp = 0;
        double gen = 0;
        double cost = 0;
        for (StripsTask t : test) {
            StateSpaceSearch.Result r = StateSpaceSearch.search(t, StateSpaceSearch.Method.GBFS, List.of(h.apply(t)),
                false, 1, budget);
            if (r.found()) {
                solved++;
                exp += r.expanded();
                gen += r.generated();
                cost += r.cost();
            }
        }
        int s = Math.max(1, solved);
        return new Row(domain, method, solved, test.size(), exp / s, gen / s, cost / s, trainMillis);
    }

    /**
     * 13.2.4: WL-GOOSE heuristics.
     *
     * @param domains the splits
     * @param budget the expansion budget
     * @param withGnn whether to include the GIN baseline
     * @param seed the seed
     * @return rows, and per domain a row "GP variance" whose {@code expanded} holds the mean
     *     variance on training states and {@code generated} on test initial states
     */
    public static List<Row> heuristics(Map<String, Split> domains, int budget, boolean withGnn, long seed) {
        List<Row> rows = new ArrayList<>();
        for (Map.Entry<String, Split> e : domains.entrySet()) {
            String dn = e.getKey();
            List<StripsTask> test = e.getValue().test();
            List<LearnedPlanningHeuristic.Sample> samples = LearnedPlanningHeuristic.optimalSamples(
                e.getValue().train(), 50_000);
            rows.add(searchRow(dn, "blind", test, t -> new RelaxationHeuristics(t, RelaxationHeuristics.Kind.BLIND),
                budget, 0));
            rows.add(searchRow(dn, "h_FF", test, t -> new RelaxationHeuristics(t, RelaxationHeuristics.Kind.H_FF),
                budget, 0));
            LearnedPlanningHeuristic ridge = LearnedPlanningHeuristic.train(LearnedPlanningHeuristic.Mode.LINEAR_RIDGE,
                samples, 2, seed);
            rows.add(searchRow(dn, "WL ridge", test, ridge::evaluator, budget, ridge.trainMillis()));
            LearnedPlanningHeuristic gp = LearnedPlanningHeuristic.train(LearnedPlanningHeuristic.Mode.GP, samples, 2,
                seed);
            rows.add(searchRow(dn, "WL GP", test, gp::evaluator, budget, gp.trainMillis()));
            double vt = 0;
            for (LearnedPlanningHeuristic.Sample s : samples) {
                vt += gp.variance(s.task(), s.state());
            }
            double vo = 0;
            for (StripsTask t : test) {
                vo += gp.variance(t, t.initialState());
            }
            rows.add(new Row(dn, "GP variance train/test", 0, 0, vt / samples.size(), vo / test.size(), 0, 0));
            if (withGnn) {
                LearnedPlanningHeuristic gnn = LearnedPlanningHeuristic.train(LearnedPlanningHeuristic.Mode.GNN,
                    samples, 2, seed);
                rows.add(searchRow(dn, "GIN", test, gnn::evaluator, budget / 4, gnn.trainMillis()));
            }
        }
        return rows;
    }

    /**
     * 13.2.5: ranking against regression, and partial-space search.
     *
     * @param domains splits for the ranking-vs-regression rows
     * @param partial splits for the partial-space rows
     * @param budget the expansion budget
     * @param seed the seed
     * @return rows
     */
    public static List<Row> ranking(Map<String, Split> domains, Map<String, Split> partial, int budget, long seed) {
        List<Row> rows = new ArrayList<>();
        for (Map.Entry<String, Split> e : domains.entrySet()) {
            List<LearnedPlanningHeuristic.Sample> samples = LearnedPlanningHeuristic.optimalSamples(
                e.getValue().train(), 50_000);
            LearnedPlanningHeuristic reg = LearnedPlanningHeuristic.train(LearnedPlanningHeuristic.Mode.LINEAR_RIDGE,
                samples, 2, seed);
            RankingHeuristic rank = RankingHeuristic.trainStates(samples, 2, 300);
            rows.add(searchRow(e.getKey(), "WL regression", e.getValue().test(), reg::evaluator, budget,
                reg.trainMillis()));
            rows.add(searchRow(e.getKey(), "WL ranking", e.getValue().test(), rank::evaluator, budget,
                rank.trainMillis()));
        }
        for (Map.Entry<String, Split> e : partial.entrySet()) {
            String dn = e.getKey();
            List<LearnedPlanningHeuristic.Sample> samples = LearnedPlanningHeuristic.optimalSamples(
                e.getValue().train(), 50_000);
            RankingHeuristic rank = RankingHeuristic.trainStates(samples, 2, 300);
            rows.add(searchRow(dn, "grounded GBFS ranking", e.getValue().test(), rank::evaluator, budget,
                rank.trainMillis()));
            rows.add(searchRow(dn, "grounded GBFS h_FF", e.getValue().test(),
                t -> new RelaxationHeuristics(t, RelaxationHeuristics.Kind.H_FF), budget, 0));
            RankingHeuristic sets = RankingHeuristic.trainActionSets(samples, 2, 300);
            int solved = 0;
            double exp = 0;
            double gen = 0;
            double cost = 0;
            for (StripsTask t : e.getValue().test()) {
                PartialSpaceSearch.Result r = PartialSpaceSearch.search(t, sets, budget * 4);
                if (r.found()) {
                    solved++;
                    exp += r.expanded();
                    gen += r.generated();
                    cost += r.cost();
                }
            }
            int s = Math.max(1, solved);
            rows.add(new Row(dn, "partial-space action-set ranking", solved, e.getValue().test().size(), exp / s,
                gen / s, cost / s, sets.trainMillis()));
        }
        return rows;
    }

    /**
     * 13.2.6: generalised planning without search.
     *
     * @param domains the splits
     * @param budget the expansion budget of the search baselines
     * @param seed the seed
     * @return rows; {@code cost} holds the mean plan length of solved instances relative to h_FF
     *     GBFS's plan on the same instance (rollouts are capped at three times that length)
     */
    public static List<Row> generalisation(Map<String, Split> domains, int budget, long seed) {
        List<Row> rows = new ArrayList<>();
        for (Map.Entry<String, Split> e : domains.entrySet()) {
            String dn = e.getKey();
            List<StripsTask> test = e.getValue().test();
            List<LearnedPlanningHeuristic.Sample> samples = LearnedPlanningHeuristic.optimalSamples(
                e.getValue().train(), 50_000);
            double[] ffLen = new double[test.size()];
            int[] cap = new int[test.size()];
            int ffSolved = 0;
            for (int i = 0; i < test.size(); i++) {
                StateSpaceSearch.Result r = StateSpaceSearch.search(test.get(i), StateSpaceSearch.Method.GBFS,
                    List.of(new RelaxationHeuristics(test.get(i), RelaxationHeuristics.Kind.H_FF)), false, 1, budget);
                ffLen[i] = r.found() ? r.plan().size() : Double.NaN;
                cap[i] = r.found() ? 3 * r.plan().size() : 200;
                ffSolved += r.found() ? 1 : 0;
            }
            rows.add(new Row(dn, "h_FF GBFS", ffSolved, test.size(), 0, 0, 1, 0));
            LearnedPlanningHeuristic h = LearnedPlanningHeuristic.train(LearnedPlanningHeuristic.Mode.LINEAR_RIDGE,
                samples, 2, seed);
            int hs = 0;
            double hr = 0;
            int hn = 0;
            for (int i = 0; i < test.size(); i++) {
                StateSpaceSearch.Result r = StateSpaceSearch.search(test.get(i), StateSpaceSearch.Method.GBFS,
                    List.of(h.evaluator(test.get(i))), false, 1, budget);
                if (r.found()) {
                    hs++;
                    if (!Double.isNaN(ffLen[i])) {
                        hr += r.plan().size() / ffLen[i];
                        hn++;
                    }
                }
            }
            rows.add(new Row(dn, "WL ridge GBFS", hs, test.size(), 0, 0, hn == 0 ? Double.NaN : hr / hn,
                h.trainMillis()));
            List<GeneralizedPlanner> planners = List.of(
                GeneralizedPlanner.random(seed),
                GeneralizedPlanner.transitionModel(samples, 2, GeneralizedPlanner.Regressor.RIDGE, seed),
                GeneralizedPlanner.transitionModel(samples, 2, GeneralizedPlanner.Regressor.BOOSTED_TREES, seed),
                GeneralizedPlanner.gnnPolicy(samples, 4, seed));
            String[] names = {"random actions", "transition model ridge", "transition model GBT", "GNN policy"};
            for (int p = 0; p < planners.size(); p++) {
                int solved = 0;
                double rel = 0;
                int n = 0;
                for (int i = 0; i < test.size(); i++) {
                    StripsTask t = test.get(i);
                    GeneralizedPlanner.Rollout ro = planners.get(p).run(t, cap[i]);
                    if (ro.solved()) {
                        if (Double.isInfinite(t.validate(ro.plan()))) {
                            throw new IllegalStateException("invalid rollout");
                        }
                        solved++;
                        if (!Double.isNaN(ffLen[i])) {
                            rel += ro.plan().size() / ffLen[i];
                            n++;
                        }
                    }
                }
                rows.add(new Row(dn, names[p], solved, test.size(), 0, 0, n == 0 ? Double.NaN : rel / n,
                    planners.get(p).trainMillis()));
            }
        }
        return rows;
    }

    /**
     * Mean WL-GP variance over states (kept for callers that want it separately).
     *
     * @param gp a GP heuristic
     * @param task a task
     * @param states states
     * @return the mean variance
     */
    public static double meanVariance(LearnedPlanningHeuristic gp, StripsTask task, List<BitSet> states) {
        double v = 0;
        for (BitSet s : states) {
            v += gp.variance(task, s);
        }
        return v / Math.max(1, states.size());
    }

    /**
     * Prints the tables.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        heuristics(domains(1), 20_000, true, 1).forEach(System.out::println);
        Map<String, Split> partial = new LinkedHashMap<>();
        partial.put("logistics", logistics(1, 2, 8));
        ranking(domains(1), partial, 20_000, 1).forEach(System.out::println);
        Map<String, Split> gen = new LinkedHashMap<>(domains(1));
        gen.put("logistics", logistics(1, 1, 6));
        generalisation(gen, 20_000, 1).forEach(System.out::println);
    }
}
