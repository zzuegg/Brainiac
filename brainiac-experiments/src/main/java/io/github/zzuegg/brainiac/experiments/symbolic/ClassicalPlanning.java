package io.github.zzuegg.brainiac.experiments.symbolic;

import io.github.zzuegg.brainiac.experimental.data.PlanningDomains;
import io.github.zzuegg.brainiac.experimental.data.PlanningMaps;
import io.github.zzuegg.brainiac.experimental.planning.GraphSearch;
import io.github.zzuegg.brainiac.experimental.planning.LearnedHeuristic;
import io.github.zzuegg.brainiac.experimental.symbolic.Landmarks;
import io.github.zzuegg.brainiac.experimental.symbolic.PatternDatabases;
import io.github.zzuegg.brainiac.experimental.symbolic.RelaxationHeuristics;
import io.github.zzuegg.brainiac.experimental.symbolic.StateSpaceSearch;
import io.github.zzuegg.brainiac.experimental.symbolic.StripsTask;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Supplier;
import java.util.function.ToDoubleFunction;
import java.util.function.ToIntFunction;

/**
 * Wave 13 items 13.2.1-13.2.3: classical planning with delete-relaxation, landmark and
 * abstraction heuristics on the {@link PlanningDomains} tasks.
 *
 * <ul>
 *   <li>{@link #relaxation} (13.2.1): blind breadth-first search (wave 12's {@link GraphSearch}),
 *       A* with goal count and with h_max, and greedy best-first search with goal count, h_add
 *       and h_FF, eager and lazy with preferred operators: coverage within an expansion budget,
 *       expansions, evaluations and plan cost.</li>
 *   <li>{@link #landmarks} (13.2.2): optimal A* with blind, h_max and LM-cut (expansions and
 *       h(s0)/h*), and satisficing GBFS with h_FF against LAMA-style alternation of h_FF and
 *       landmark count.</li>
 *   <li>{@link #abstraction} (13.2.3): additive 4+4 tile PDBs against Manhattan + linear
 *       conflict and wave 12's DAVI heuristic on the 8-puzzle, and the canonical iPDB heuristic
 *       against a single PDB, h_max and LM-cut on Gripper and Blocksworld.</li>
 * </ul>
 */
public final class ClassicalPlanning {

    private ClassicalPlanning() {
    }

    /**
     * One table row.
     *
     * @param domain the domain
     * @param method the planner
     * @param solved instances solved within the budget
     * @param instances instances tried
     * @param expanded mean expansions over solved instances
     * @param evaluated mean heuristic evaluations over solved instances
     * @param cost mean plan cost over solved instances
     * @param millis mean wall time
     */
    public record Row(String domain, String method, int solved, int instances, double expanded, double evaluated,
            double cost, double millis) {

        @Override
        public String toString() {
            return String.format("%s/%s: %d/%d exp=%.0f eval=%.0f cost=%.1f %.0fms", domain, method, solved,
                instances, expanded, evaluated, cost, millis);
        }
    }

    /** A named planner. */
    private record Planner(String name, java.util.function.Function<StripsTask, StateSpaceSearch.Result> run) {
    }

    private static Row run(String domain, Planner p, List<StripsTask> tasks) {
        int solved = 0;
        double exp = 0;
        double eval = 0;
        double cost = 0;
        double ms = 0;
        for (StripsTask t : tasks) {
            StateSpaceSearch.Result r = p.run().apply(t);
            ms += r.millis();
            if (r.found()) {
                double c = t.validate(r.plan());
                if (Double.isInfinite(c)) {
                    throw new IllegalStateException("invalid plan from " + p.name());
                }
                solved++;
                exp += r.expanded();
                eval += r.evaluated();
                cost += c;
            }
        }
        int s = Math.max(1, solved);
        return new Row(domain, p.name(), solved, tasks.size(), exp / s, eval / s, cost / s, ms / tasks.size());
    }

    /**
     * Blind breadth-first search through {@link GraphSearch}'s generic search.
     *
     * @param task the task
     * @param budget the expansion budget
     * @return the result
     */
    public static StateSpaceSearch.Result blindBfs(StripsTask task, int budget) {
        long t0 = System.nanoTime();
        GraphSearch.GenericResult<BitSet> r = GraphSearch.search(task.initialState(), task::isGoal,
            s -> task.applicable(s).stream().map(a -> a.apply(s)).toList(), s -> s, s -> 0, GraphSearch.Method.BFS,
            1, budget);
        List<StripsTask.Action> plan = new ArrayList<>();
        for (int i = 0; i + 1 < r.path().size(); i++) {
            BitSet from = r.path().get(i);
            BitSet to = r.path().get(i + 1);
            for (StripsTask.Action a : task.applicable(from)) {
                if (a.apply(from).equals(to)) {
                    plan.add(a);
                    break;
                }
            }
        }
        return new StateSpaceSearch.Result(plan, r.found(), r.found() ? task.validate(plan) : Double.POSITIVE_INFINITY,
            r.expanded(), 0, r.generated(), (System.nanoTime() - t0) / 1e6);
    }

    private static StateSpaceSearch.Result gbfs(StripsTask t, RelaxationHeuristics.Kind k, boolean lazy,
            boolean preferred, int budget) {
        return StateSpaceSearch.search(t, lazy ? StateSpaceSearch.Method.LAZY_GBFS : StateSpaceSearch.Method.GBFS,
            List.of(new RelaxationHeuristics(t, k)), preferred, 1, budget);
    }

    /**
     * The 13.2.1 domains at one size each.
     *
     * @param scale 0 for the test sizes, larger for bigger instances
     * @param seed the seed
     * @return domain name to instances
     */
    public static Map<String, List<StripsTask>> domains(int scale, long seed) {
        Map<String, List<StripsTask>> d = new LinkedHashMap<>();
        d.put("blocksworld-" + (7 + scale), List.of(PlanningDomains.blocksworld(7 + scale, seed),
            PlanningDomains.blocksworld(7 + scale, seed + 1)));
        d.put("gripper-" + (8 + 4 * scale), List.of(PlanningDomains.gripper(8 + 4 * scale)));
        d.put("logistics-2c-" + (4 + scale), List.of(PlanningDomains.logistics(2, 1, 4 + scale, false, seed),
            PlanningDomains.logistics(2, 1, 4 + scale, false, seed + 1)));
        d.put("visitall-" + (5 + scale), List.of(PlanningDomains.visitAll(5 + scale)));
        d.put("ferry-4-" + (5 + scale), List.of(PlanningDomains.ferry(4, 5 + scale, seed)));
        return d;
    }

    /**
     * 13.2.1: delete-relaxation heuristics with greedy and optimal search.
     *
     * @param domains the instances
     * @param budget the expansion budget per run
     * @return rows
     */
    public static List<Row> relaxation(Map<String, List<StripsTask>> domains, int budget) {
        List<Planner> planners = List.of(
            new Planner("BFS blind (GraphSearch)", t -> blindBfs(t, budget)),
            new Planner("A* goal-count", t -> StateSpaceSearch.astar(t,
                new RelaxationHeuristics(t, RelaxationHeuristics.Kind.GOAL_COUNT), budget)),
            new Planner("A* h_max", t -> StateSpaceSearch.astar(t,
                new RelaxationHeuristics(t, RelaxationHeuristics.Kind.H_MAX), budget)),
            new Planner("GBFS goal-count", t -> gbfs(t, RelaxationHeuristics.Kind.GOAL_COUNT, false, false, budget)),
            new Planner("GBFS h_add", t -> gbfs(t, RelaxationHeuristics.Kind.H_ADD, false, false, budget)),
            new Planner("GBFS h_FF", t -> gbfs(t, RelaxationHeuristics.Kind.H_FF, false, false, budget)),
            new Planner("lazy GBFS h_FF", t -> gbfs(t, RelaxationHeuristics.Kind.H_FF, true, false, budget)),
            new Planner("lazy GBFS h_FF+pref", t -> gbfs(t, RelaxationHeuristics.Kind.H_FF, true, true, budget)));
        List<Row> rows = new ArrayList<>();
        for (Map.Entry<String, List<StripsTask>> e : domains.entrySet()) {
            for (Planner p : planners) {
                rows.add(run(e.getKey(), p, e.getValue()));
            }
        }
        return rows;
    }

    /**
     * 13.2.2: LM-cut in optimal search and landmark count in satisficing search.
     *
     * @param budget the expansion budget
     * @param seed the seed
     * @return {@code optimal} rows (A* blind, h_max, LM-cut; with mean h(s0)/h* in the
     *     {@code evaluated} column replaced by its ratio) and satisficing rows
     */
    public static Map<String, List<Row>> landmarks(int budget, long seed) {
        Map<String, List<StripsTask>> small = new LinkedHashMap<>();
        small.put("gripper-4", List.of(PlanningDomains.gripper(4)));
        small.put("logistics-2c-3 unit", List.of(PlanningDomains.logistics(2, 1, 3, false, seed),
            PlanningDomains.logistics(2, 1, 3, false, seed + 1)));
        small.put("logistics-2c-3 costs", List.of(PlanningDomains.logistics(2, 1, 3, true, seed),
            PlanningDomains.logistics(2, 1, 3, true, seed + 1)));
        small.put("blocksworld-5", List.of(PlanningDomains.blocksworld(5, seed), PlanningDomains.blocksworld(5,
            seed + 1)));
        List<Row> optimal = new ArrayList<>();
        for (Map.Entry<String, List<StripsTask>> e : small.entrySet()) {
            List<Supplier<Planner>> ps = List.of(
                () -> new Planner("A* blind", t -> StateSpaceSearch.astar(t,
                    new RelaxationHeuristics(t, RelaxationHeuristics.Kind.BLIND), budget)),
                () -> new Planner("A* h_max", t -> StateSpaceSearch.astar(t,
                    new RelaxationHeuristics(t, RelaxationHeuristics.Kind.H_MAX), budget)),
                () -> new Planner("A* LM-cut", t -> StateSpaceSearch.astar(t, Landmarks.lmCut(t), budget)));
            for (Supplier<Planner> sp : ps) {
                Row r = run(e.getKey(), sp.get(), e.getValue());
                optimal.add(r);
            }
            // h(s0)/h* for h_max and LM-cut, using the LM-cut optimum
            double hmaxRatio = 0;
            double lmRatio = 0;
            for (StripsTask t : e.getValue()) {
                double opt = StateSpaceSearch.astar(t, Landmarks.lmCut(t), budget).cost();
                hmaxRatio += new RelaxationHeuristics(t, RelaxationHeuristics.Kind.H_MAX).evaluate(t.initialState())
                    .h() / opt;
                lmRatio += Landmarks.lmCutValue(t, t.initialState(), null) / opt;
            }
            int n = e.getValue().size();
            optimal.add(new Row(e.getKey(), "h_max(s0)/h*", 0, 0, hmaxRatio / n, 0, 0, 0));
            optimal.add(new Row(e.getKey(), "LM-cut(s0)/h*", 0, 0, lmRatio / n, 0, 0, 0));
        }
        List<StripsTask> large = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            large.add(PlanningDomains.logistics(3, 1, 8, false, seed + 10 + i));
        }
        int satBudget = budget / 4;
        List<Row> satisficing = new ArrayList<>();
        satisficing.add(run("logistics-3c-8", new Planner("GBFS h_FF",
            t -> gbfs(t, RelaxationHeuristics.Kind.H_FF, false, false, satBudget)), large));
        satisficing.add(run("logistics-3c-8", new Planner("lazy GBFS h_FF+pref",
            t -> gbfs(t, RelaxationHeuristics.Kind.H_FF, true, true, satBudget)), large));
        satisficing.add(run("logistics-3c-8", new Planner("GBFS landmark count",
            t -> StateSpaceSearch.search(t, StateSpaceSearch.Method.GBFS,
                List.of(Landmarks.generate(t).landmarkCount()), false, 1, satBudget)), large));
        satisficing.add(run("logistics-3c-8", new Planner("LAMA-style h_FF + LM count, pref",
            t -> StateSpaceSearch.search(t, StateSpaceSearch.Method.LAZY_GBFS,
                List.of(new RelaxationHeuristics(t, RelaxationHeuristics.Kind.H_FF),
                    Landmarks.generate(t).landmarkCount()), true, 1, satBudget)), large));
        Map<String, List<Row>> out = new LinkedHashMap<>();
        out.put("optimal", optimal);
        out.put("satisficing", satisficing);
        return out;
    }

    /**
     * The 8-puzzle part of 13.2.3.
     *
     * @param instances test boards
     * @param daviSteps DAVI training steps
     * @param seed the seed
     * @return heuristic name to {@code {mean expansions, optimal share, share of states (on the
     *     found paths) where h > h*}}
     */
    public static Map<String, double[]> eightPuzzle(int instances, int daviSteps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        byte[] table = PlanningMaps.eightPuzzleTable();
        List<int[]> test = new ArrayList<>();
        for (int i = 0; i < instances; i++) {
            test.add(PlanningMaps.scrambleTiles(3, 40 + r.nextInt(40), r));
        }
        ToIntFunction<int[]> pdb = PatternDatabases.slidingTileAdditive(3, new int[][] {{1, 2, 3, 4}, {5, 6, 7, 8}});
        LearnedHeuristic.Domain eight = LearnedHeuristic.eightPuzzle();
        LearnedHeuristic davi = new LearnedHeuristic(eight, LearnedHeuristic.Mode.DAVI, 64, seed);
        davi.train(daviSteps, 64, 30, 2e-3f, 10, 1.0, seed + 1);
        Map<String, ToDoubleFunction<int[]>> hs = new LinkedHashMap<>();
        hs.put("Manhattan", s -> PlanningMaps.manhattan(s, 3));
        hs.put("Manhattan+LC", s -> PlanningMaps.manhattanLinearConflict(s, 3));
        hs.put("additive PDB 4+4", pdb::applyAsInt);
        hs.put("DAVI (wave 12)", davi::estimate);
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Map.Entry<String, ToDoubleFunction<int[]>> e : hs.entrySet()) {
            double exp = 0;
            int optimal = 0;
            int over = 0;
            int states = 0;
            for (int[] s : test) {
                GraphSearch.GenericResult<int[]> res = GraphSearch.search(s, eight::goal, eight::successors,
                    b -> PlanningMaps.rank9(b), e.getValue(), GraphSearch.Method.ASTAR, 1,
                    200_000);
                exp += res.expanded();
                if (res.found() && Math.round(res.cost()) == table[PlanningMaps.rank9(s)]) {
                    optimal++;
                }
                for (int[] p : res.path()) {
                    states++;
                    if (e.getValue().applyAsDouble(p) > table[PlanningMaps.rank9(p)] + 1e-9) {
                        over++;
                    }
                }
            }
            out.put(e.getKey(), new double[] {exp / test.size(), optimal / (double) test.size(),
                over / (double) Math.max(1, states)});
        }
        return out;
    }

    /**
     * The STRIPS part of 13.2.3: optimal A* with abstraction and landmark heuristics.
     *
     * @param budget the expansion budget
     * @param seed the seed
     * @return rows; the {@code evaluated} column of the PDB rows holds the total PDB entries
     */
    public static List<Row> abstraction(int budget, long seed) {
        Map<String, List<StripsTask>> d = new LinkedHashMap<>();
        d.put("gripper-5", List.of(PlanningDomains.gripper(5)));
        d.put("blocksworld-5", List.of(PlanningDomains.blocksworld(5, seed + 3),
            PlanningDomains.blocksworld(5, seed + 4)));
        List<Row> rows = new ArrayList<>();
        for (Map.Entry<String, List<StripsTask>> e : d.entrySet()) {
            rows.add(run(e.getKey(), new Planner("A* h_max", t -> StateSpaceSearch.astar(t,
                new RelaxationHeuristics(t, RelaxationHeuristics.Kind.H_MAX), budget)), e.getValue()));
            rows.add(run(e.getKey(), new Planner("A* LM-cut", t -> StateSpaceSearch.astar(t, Landmarks.lmCut(t),
                budget)), e.getValue()));
            double[] size = new double[2];
            double[] build = new double[2];
            Row single = run(e.getKey(), new Planner("A* single PDB", t -> {
                PatternDatabases.SasView v = new PatternDatabases.SasView(t);
                long t0 = System.nanoTime();
                PatternDatabases.Canonical c = PatternDatabases.canonical(v, List.of(greedyPattern(v, 20_000)),
                    20_000);
                size[0] += c.totalSize();
                build[0] += (System.nanoTime() - t0) / 1e6;
                return StateSpaceSearch.astar(t, c, budget);
            }), e.getValue());
            Row canon = run(e.getKey(), new Planner("A* canonical iPDB", t -> {
                PatternDatabases.SasView v = new PatternDatabases.SasView(t);
                long t0 = System.nanoTime();
                PatternDatabases.Canonical c = PatternDatabases.ipdb(v, 2_000, 20_000, 40, 3, 6, seed);
                size[1] += c.totalSize();
                build[1] += (System.nanoTime() - t0) / 1e6;
                return StateSpaceSearch.astar(t, c, budget);
            }), e.getValue());
            int n = e.getValue().size();
            rows.add(new Row(single.domain(), single.method() + String.format(" (build %.0fms)", build[0] / n),
                single.solved(), single.instances(), single.expanded(), size[0] / n, single.cost(), single.millis()));
            rows.add(new Row(canon.domain(), canon.method() + String.format(" (build %.0fms)", build[1] / n),
                canon.solved(), canon.instances(), canon.expanded(), size[1] / n, canon.cost(), canon.millis()));
        }
        return rows;
    }

    /** The goal variables, added greedily while the PDB stays under the size limit. */
    private static int[] greedyPattern(PatternDatabases.SasView v, int maxSize) {
        List<Integer> p = new ArrayList<>();
        long size = 1;
        for (int g : v.goalVariables()) {
            if (size * v.domain(g) <= maxSize) {
                p.add(g);
                size *= v.domain(g);
            }
        }
        return p.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Prints the tables.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        relaxation(domains(1, 1), 100_000).forEach(System.out::println);
        landmarks(100_000, 1).forEach((k, v) -> v.forEach(r -> System.out.println(k + " " + r)));
        eightPuzzle(30, 600, 1).forEach((k, v) -> System.out.println(k + " " + Arrays.toString(v)));
        abstraction(100_000, 1).forEach(System.out::println);
    }
}
