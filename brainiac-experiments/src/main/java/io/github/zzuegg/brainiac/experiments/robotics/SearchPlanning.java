package io.github.zzuegg.brainiac.experiments.robotics;

import io.github.zzuegg.brainiac.experimental.data.PlanningMaps;
import io.github.zzuegg.brainiac.experimental.planning.GraphSearch;
import io.github.zzuegg.brainiac.experimental.planning.LearnedHeuristic;
import io.github.zzuegg.brainiac.experimental.planning.NeuralAStar;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.planning.ValueIterationNetwork;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Wave 12 items 12.3.11 to 12.3.13: search-based planning.
 *
 * <ul>
 *   <li>{@link #grids}: BFS, Dijkstra, A*, weighted A* (w = 2), greedy best-first and jump
 *       point search on 8-connected random-obstacle, maze and room grids, by nodes expanded
 *       and path cost relative to Dijkstra;</li>
 *   <li>{@link #puzzles}: A* on the 8-puzzle and 3x3 Lights Out with hand-made heuristics
 *       against DeepCubeA-style learned heuristics (MSE value iteration, and the asymmetric,
 *       offset-calibrated variant), by optimality rate, expansions and inadmissibility;</li>
 *   <li>{@link #learnedPlanners}: a value iteration network against a reactive MLP that maps
 *       the whole map to the next action (imitation of optimal actions, success on unseen
 *       maps, and VIN on larger maps), and {@link #neuralAStar} against plain A*
 *       (expansions and path-cost ratio).</li>
 * </ul>
 */
public final class SearchPlanning {

    private SearchPlanning() {
    }

    /**
     * A grid-search row.
     *
     * @param map the map family
     * @param method the method
     * @param expanded mean nodes expanded
     * @param costRatio mean path cost over Dijkstra's
     * @param millis mean time per query
     */
    public record GridRow(String map, String method, double expanded, double costRatio, double millis) {
    }

    /**
     * Grid search on three map families.
     *
     * @param side the grid side
     * @param queries queries per family
     * @param seed the seed
     * @return rows
     */
    public static List<GridRow> grids(int side, int queries, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<GridRow> rows = new ArrayList<>();
        String[] families = {"random 20%", "maze", "rooms"};
        GraphSearch.Method[] methods = GraphSearch.Method.values();
        for (String fam : families) {
            double[] exp = new double[methods.length];
            double[] ratio = new double[methods.length];
            double[] time = new double[methods.length];
            int done = 0;
            while (done < queries) {
                PlanningMaps.Grid g = switch (fam) {
                    case "maze" -> PlanningMaps.maze(side - 1 + side % 2, side - 1 + side % 2, 0.1, r);
                    case "rooms" -> PlanningMaps.rooms(side, side, 4, 4, r);
                    default -> PlanningMaps.randomObstacles(side, side, 0.2, r);
                };
                int[] pair = g.randomPair(side / 2, r);
                if (pair == null) {
                    continue;
                }
                double reference = GraphSearch.grid(g.blocked(), g.width(), g.height(), pair[0], pair[1],
                    GraphSearch.Method.DIJKSTRA, GraphSearch.Heuristic.ZERO, 1, true).cost();
                if (!Double.isFinite(reference)) {
                    continue;
                }
                for (int m = 0; m < methods.length; m++) {
                    long t0 = System.nanoTime();
                    GraphSearch.Result res = GraphSearch.grid(g.blocked(), g.width(), g.height(), pair[0], pair[1],
                        methods[m], GraphSearch.Heuristic.OCTILE, 2, true);
                    time[m] += (System.nanoTime() - t0) / 1e6;
                    exp[m] += res.expanded();
                    ratio[m] += res.cost() / reference;
                }
                done++;
            }
            for (int m = 0; m < methods.length; m++) {
                String name = methods[m] == GraphSearch.Method.WEIGHTED_ASTAR ? "WEIGHTED_ASTAR(w=2)"
                    : methods[m].name();
                rows.add(new GridRow(fam, name, exp[m] / queries, ratio[m] / queries, time[m] / queries));
            }
        }
        return rows;
    }

    /**
     * A puzzle row.
     *
     * @param puzzle the puzzle
     * @param method the heuristic and search
     * @param optimalRate the share of instances solved at optimal cost
     * @param expanded mean nodes expanded
     * @param inadmissible the share of test states whose heuristic exceeds the true cost
     */
    public record PuzzleRow(String puzzle, String method, double optimalRate, double expanded,
            double inadmissible) {
    }

    /**
     * Learned against hand-made heuristics on the 8-puzzle and 3x3 Lights Out.
     *
     * @param instances test instances per puzzle
     * @param trainSteps value-iteration steps per learned heuristic
     * @param seed the seed
     * @return rows
     */
    public static List<PuzzleRow> puzzles(int instances, int trainSteps, long seed) {
        List<PuzzleRow> rows = new ArrayList<>();
        SplittableRandom r = new SplittableRandom(seed);
        byte[] table = PlanningMaps.eightPuzzleTable();
        LearnedHeuristic.Domain eight = LearnedHeuristic.eightPuzzle();
        List<int[]> test = new ArrayList<>();
        int[] cost = new int[instances];
        for (int i = 0; i < instances; i++) {
            int[] s = PlanningMaps.scrambleTiles(3, 18 + r.nextInt(30), r);
            test.add(s);
            cost[i] = table[PlanningMaps.rank9(s)];
        }
        List<int[]> val = new ArrayList<>();
        int[] valCost = new int[300];
        for (int i = 0; i < valCost.length; i++) {
            int[] s = PlanningMaps.scrambleTiles(3, 1 + r.nextInt(48), r);
            val.add(s);
            valCost[i] = table[PlanningMaps.rank9(s)];
        }
        rows.add(puzzleRow("8-puzzle", "A* Manhattan", eight, test, cost,
            s -> PlanningMaps.manhattan(s, 3), GraphSearch.Method.ASTAR, 1, s -> table[PlanningMaps.rank9(s)]));
        rows.add(puzzleRow("8-puzzle", "A* Manhattan+LC", eight, test, cost,
            s -> PlanningMaps.manhattanLinearConflict(s, 3), GraphSearch.Method.ASTAR, 1,
            s -> table[PlanningMaps.rank9(s)]));
        for (LearnedHeuristic.Mode mode : LearnedHeuristic.Mode.values()) {
            LearnedHeuristic h = new LearnedHeuristic(eight, mode, 96, seed + mode.ordinal());
            h.train(trainSteps, 128, 30, 2e-3f, 10, 4.0, seed + 7);
            double worst = h.calibrate(val, valCost);
            String name = mode == LearnedHeuristic.Mode.DAVI ? "A* DAVI (MSE)"
                : String.format("A* asymmetric (offset %.2f)", worst);
            rows.add(puzzleRow("8-puzzle", name, eight, test, cost, h::estimate, GraphSearch.Method.ASTAR, 1,
                s -> table[PlanningMaps.rank9(s)]));
            if (mode == LearnedHeuristic.Mode.DAVI) {
                rows.add(puzzleRow("8-puzzle", "weighted A* (w=1.5) DAVI", eight, test, cost, h::estimate,
                    GraphSearch.Method.WEIGHTED_ASTAR, 1.5, s -> table[PlanningMaps.rank9(s)]));
            }
        }
        byte[] lights = PlanningMaps.lightsOutTable(3);
        LearnedHeuristic.Domain lo = LearnedHeuristic.lightsOut(3);
        List<int[]> lt = new ArrayList<>();
        int[] lc = new int[instances];
        for (int i = 0; i < instances; i++) {
            int[] s = lo.scramble(5 + r.nextInt(8), r);
            lt.add(s);
            lc[i] = lights[LearnedHeuristic.mask(s)];
        }
        List<int[]> lv = new ArrayList<>();
        int[] lvc = new int[200];
        for (int i = 0; i < lvc.length; i++) {
            int[] s = lo.scramble(1 + r.nextInt(12), r);
            lv.add(s);
            lvc[i] = lights[LearnedHeuristic.mask(s)];
        }
        ToDoubleFunction<int[]> truth = s -> lights[LearnedHeuristic.mask(s)];
        rows.add(puzzleRow("lights out 3x3", "BFS", lo, lt, lc, s -> 0, GraphSearch.Method.BFS, 1, truth));
        rows.add(puzzleRow("lights out 3x3", "A* lit/5", lo, lt, lc, s -> Math.ceil(Arrays.stream(s).sum() / 5.0),
            GraphSearch.Method.ASTAR, 1, truth));
        for (LearnedHeuristic.Mode mode : LearnedHeuristic.Mode.values()) {
            LearnedHeuristic h = new LearnedHeuristic(lo, mode, 64, seed + 10 + mode.ordinal());
            h.train(trainSteps / 2, 128, 12, 2e-3f, 10, 4.0, seed + 11);
            double worst = h.calibrate(lv, lvc);
            String name = mode == LearnedHeuristic.Mode.DAVI ? "A* DAVI (MSE)"
                : String.format("A* asymmetric (offset %.2f)", worst);
            rows.add(puzzleRow("lights out 3x3", name, lo, lt, lc, h::estimate, GraphSearch.Method.ASTAR, 1, truth));
        }
        return rows;
    }

    private static PuzzleRow puzzleRow(String puzzle, String name, LearnedHeuristic.Domain d, List<int[]> test,
            int[] cost, ToDoubleFunction<int[]> h, GraphSearch.Method method, double weight,
            ToDoubleFunction<int[]> truth) {
        int optimal = 0;
        double expanded = 0;
        int over = 0;
        int states = 0;
        for (int i = 0; i < test.size(); i++) {
            GraphSearch.GenericResult<int[]> res = GraphSearch.search(test.get(i), d::goal, d::successors,
                SearchPlanning::pack, h, method, weight, 400_000);
            expanded += res.expanded();
            if (res.found() && Math.round(res.cost()) == cost[i]) {
                optimal++;
            }
            for (int[] s : res.path()) {
                states++;
                if (h.applyAsDouble(s) > truth.applyAsDouble(s) + 1e-9) {
                    over++;
                }
            }
        }
        return new PuzzleRow(puzzle, name, optimal / (double) test.size(), expanded / test.size(),
            states == 0 ? 0 : over / (double) states);
    }

    private static Object pack(int[] s) {
        long k = 0;
        for (int v : s) {
            k = (k << 4) | v;
        }
        return k;
    }

    /**
     * A learned-planner row.
     *
     * @param method the method
     * @param success the share of test queries that reach the goal
     * @param costRatio mean path cost over the optimal cost, on successes
     * @param expanded mean nodes expanded (search-based rows), NaN otherwise
     */
    public record LearnedRow(String method, double success, double costRatio, double expanded) {
    }

    private static final int[][] MOVES = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

    /** Optimal 8-connected cost-to-go to a goal, by Dijkstra from the goal (moves are symmetric). */
    private static double[] costToGo(PlanningMaps.Grid g, int goal) {
        int n = g.width() * g.height();
        double[] d = new double[n];
        Arrays.fill(d, Double.POSITIVE_INFINITY);
        d[goal] = 0;
        java.util.PriorityQueue<double[]> q = new java.util.PriorityQueue<>((a, b) -> Double.compare(a[0], b[0]));
        q.add(new double[] {0, goal});
        while (!q.isEmpty()) {
            double[] top = q.poll();
            int c = (int) top[1];
            if (top[0] > d[c]) {
                continue;
            }
            for (int[] m : MOVES) {
                int nc = step(g, c, m);
                if (nc < 0) {
                    continue;
                }
                double nd = d[c] + (m[0] != 0 && m[1] != 0 ? Math.sqrt(2) : 1);
                if (nd < d[nc]) {
                    d[nc] = nd;
                    q.add(new double[] {nd, nc});
                }
            }
        }
        return d;
    }

    private static int step(PlanningMaps.Grid g, int c, int[] m) {
        int x = c % g.width();
        int y = c / g.width();
        int nx = x + m[0];
        int ny = y + m[1];
        if (!g.free(nx, ny)) {
            return -1;
        }
        if (m[0] != 0 && m[1] != 0 && (!g.free(x + m[0], y) || !g.free(x, y + m[1]))) {
            return -1;
        }
        return ny * g.width() + nx;
    }

    private static int expertAction(PlanningMaps.Grid g, double[] ctg, int c) {
        int best = -1;
        double bv = Double.POSITIVE_INFINITY;
        for (int a = 0; a < MOVES.length; a++) {
            int nc = step(g, c, MOVES[a]);
            if (nc < 0) {
                continue;
            }
            double v = ctg[nc] + (a >= 4 ? Math.sqrt(2) : 1);
            if (v < bv - 1e-9) {
                bv = v;
                best = a;
            }
        }
        return best;
    }

    private static float[] planes(PlanningMaps.Grid g, int goal, int agent) {
        int n = g.width() * g.height();
        float[] x = new float[(agent >= 0 ? 3 : 2) * n];
        for (int i = 0; i < n; i++) {
            x[i] = g.blocked()[i] ? 1 : 0;
        }
        x[n + goal] = 1;
        if (agent >= 0) {
            x[2 * n + agent] = 1;
        }
        return x;
    }

    /** A query on a map. */
    private record Query(PlanningMaps.Grid grid, int start, int goal, double[] ctg) {
    }

    private static List<Query> queries(int side, int count, SplittableRandom r) {
        List<Query> out = new ArrayList<>();
        while (out.size() < count) {
            PlanningMaps.Grid g = PlanningMaps.blocks(side, side, side / 2, 3, r);
            int[] p = g.randomPair(side / 2, r);
            if (p == null) {
                continue;
            }
            double[] ctg = costToGo(g, p[1]);
            if (Double.isFinite(ctg[p[0]])) {
                out.add(new Query(g, p[0], p[1], ctg));
            }
        }
        return out;
    }

    /** A policy: next action from a query and the current cell. */
    @FunctionalInterface
    private interface Policy {
        int act(Query q, int cell);
    }

    private static double[] rollout(List<Query> test, Policy policy) {
        int success = 0;
        double ratio = 0;
        for (Query q : test) {
            int c = q.start();
            double len = 0;
            int limit = (int) Math.ceil(2 * q.ctg()[q.start()]) + 4;
            for (int t = 0; t < limit && c != q.goal(); t++) {
                int a = policy.act(q, c);
                int nc = step(q.grid(), c, MOVES[a]);
                if (nc < 0) {
                    break;
                }
                len += a >= 4 ? Math.sqrt(2) : 1;
                c = nc;
            }
            if (c == q.goal()) {
                success++;
                ratio += len / Math.max(1e-9, q.ctg()[q.start()]);
            }
        }
        return new double[] {success / (double) test.size(), success == 0 ? Double.NaN : ratio / success};
    }

    /**
     * Value iteration network, reactive CNN and Neural A*.
     *
     * @param side the training map side
     * @param largeSide the larger test map side
     * @param trainMaps training maps
     * @param testMaps test queries
     * @param steps gradient steps for each learned planner
     * @param seed the seed
     * @return rows
     */
    public static List<LearnedRow> learnedPlanners(int side, int largeSide, int trainMaps, int testMaps, int steps,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Query> train = queries(side, trainMaps, r);
        List<Query> test = queries(side, testMaps, r);
        List<Query> large = queries(largeSide, testMaps, r);
        List<float[]> vinMaps = new ArrayList<>();
        List<float[]> cnnMaps = new ArrayList<>();
        List<Integer> cells = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        for (Query q : train) {
            float[] base = planes(q.grid(), q.goal(), -1);
            for (int c = 0; c < side * side; c++) {
                if (c == q.goal() || !Double.isFinite(q.ctg()[c]) || r.nextDouble() > 0.35) {
                    continue;
                }
                int a = expertAction(q.grid(), q.ctg(), c);
                if (a < 0) {
                    continue;
                }
                vinMaps.add(base);
                cnnMaps.add(planes(q.grid(), q.goal(), c));
                cells.add(c);
                labels.add(a);
            }
        }
        int[] cellArr = cells.stream().mapToInt(Integer::intValue).toArray();
        int[] labelArr = labels.stream().mapToInt(Integer::intValue).toArray();
        List<LearnedRow> rows = new ArrayList<>();
        double[] oracle = rollout(test, (q, c) -> expertAction(q.grid(), q.ctg(), c));
        rows.add(new LearnedRow("value iteration oracle", oracle[0], oracle[1], Double.NaN));
        int k = (int) (1.5 * side);
        ValueIterationNetwork vin = new ValueIterationNetwork(16, 8, 8, k, side, seed + 1);
        vin.fit(vinMaps, cellArr, labelArr, side, steps, 32, 5e-3f, seed + 2);
        double[] v = rollout(test, (q, c) -> vin.act(planes(q.grid(), q.goal(), -1), side, c));
        rows.add(new LearnedRow("VIN K=" + k, v[0], v[1], Double.NaN));
        vin.iterations((int) (1.5 * largeSide));
        double[] vl = rollout(large, (q, c) -> vin.act(planes(q.grid(), q.goal(), -1), largeSide, c));
        rows.add(new LearnedRow("VIN on " + largeSide + "x" + largeSide + " K=" + vin.iterations(), vl[0], vl[1],
            Double.NaN));
        vin.iterations(k);
        double[] vls = rollout(large, (q, c) -> vin.act(planes(q.grid(), q.goal(), -1), largeSide, c));
        rows.add(new LearnedRow("VIN on " + largeSide + "x" + largeSide + " K=" + k, vls[0], vls[1], Double.NaN));
        SimpleMlp mlp = SimpleMlp.relu(seed + 3, 3 * side * side, 128, 64, 8);
        trainReactive(mlp, cnnMaps, labelArr, side, steps, seed + 4);
        double[] c = rollout(test, (q, cell) -> reactiveAct(mlp, planes(q.grid(), q.goal(), cell), side));
        rows.add(new LearnedRow("reactive MLP", c[0], c[1], Double.NaN));
        return rows;
    }

    /**
     * Neural A* against A* with the same tie-breaking, on 4-connected maps.
     *
     * @param side the map side
     * @param trainMaps training maps
     * @param testMaps test queries
     * @param steps gradient steps
     * @param seed the seed
     * @return rows: A*, untrained Neural A*, Neural A*
     */
    public static List<LearnedRow> neuralAStar(int side, int trainMaps, int testMaps, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Query> train = queries(side, trainMaps, r);
        List<Query> test = queries(side, testMaps, r);
        NeuralAStar na = new NeuralAStar(side, 16, 1f, seed + 5);
        List<boolean[]> maps = new ArrayList<>();
        List<List<Integer>> paths = new ArrayList<>();
        int[] starts = new int[train.size()];
        int[] goals = new int[train.size()];
        for (int i = 0; i < train.size(); i++) {
            Query q = train.get(i);
            maps.add(q.grid().blocked());
            starts[i] = q.start();
            goals[i] = q.goal();
            paths.add(GraphSearch.grid(q.grid().blocked(), side, side, q.start(), q.goal(), GraphSearch.Method.ASTAR,
                GraphSearch.Heuristic.MANHATTAN, 1, false).path());
        }
        double[] ones = new double[side * side];
        Arrays.fill(ones, 1);
        List<LearnedRow> rows = new ArrayList<>();
        double[] plain = neuralAStarStats(na, test, ones, side);
        rows.add(new LearnedRow("A* (4-connected)", plain[0], plain[1], plain[2]));
        double[] untrained = neuralAStarStats(na, test, null, side);
        rows.add(new LearnedRow("Neural A* untrained", untrained[0], untrained[1], untrained[2]));
        na.fit(maps, starts, goals, paths, steps, 8, 2e-3f, seed + 6);
        double[] learned = neuralAStarStats(na, test, null, side);
        rows.add(new LearnedRow("Neural A*", learned[0], learned[1], learned[2]));
        return rows;
    }

    private static double[] neuralAStarStats(NeuralAStar na, List<Query> test, double[] phi, int side) {
        int success = 0;
        double ratio = 0;
        double expanded = 0;
        for (Query q : test) {
            NeuralAStar.Search s = na.search(q.grid().blocked(), phi, q.start(), q.goal());
            expanded += s.expanded();
            if (!s.path().isEmpty()) {
                success++;
                double opt = GraphSearch.grid(q.grid().blocked(), side, side, q.start(), q.goal(),
                    GraphSearch.Method.ASTAR, GraphSearch.Heuristic.MANHATTAN, 1, false).cost();
                ratio += (s.path().size() - 1) / opt;
            }
        }
        return new double[] {success / (double) test.size(), success == 0 ? Double.NaN : ratio / success,
            expanded / test.size()};
    }

    private static void trainReactive(SimpleMlp mlp, List<float[]> maps, int[] labels, int side, int steps,
            long seed) {
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        int plane = 3 * side * side;
        int b = 32;
        for (int s = 0; s < steps; s++) {
            float[] x = new float[b * plane];
            float[] y = new float[b];
            for (int i = 0; i < b; i++) {
                int k = r.nextInt(labels.length);
                System.arraycopy(maps.get(k), 0, x, i * plane, plane);
                y[i] = labels[k];
            }
            Var logits = mlp.forward(Var.of(Tensor.of(Shape.of(b, plane), x)));
            Functional.crossEntropy(logits, Tensor.of(Shape.of(b), y)).backward();
            opt.step(mlp.parameters());
        }
    }

    private static int reactiveAct(SimpleMlp mlp, float[] map, int side) {
        float[] l = mlp.predict(Tensor.of(Shape.of(1, 3 * side * side), map)).toArray();
        int best = 0;
        for (int a = 1; a < l.length; a++) {
            if (l[a] > l[best]) {
                best = a;
            }
        }
        return best;
    }
}
