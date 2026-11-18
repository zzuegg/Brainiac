package io.github.zzuegg.brainiac.experimental.planning;

import io.github.zzuegg.brainiac.experimental.data.PlanningMaps;
import io.github.zzuegg.brainiac.experimental.geometry.GeometryLinalg;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.PriorityQueue;
import java.util.SplittableRandom;

/**
 * Sampling-based motion planning in continuous configuration spaces:
 *
 * <ul>
 *   <li>{@link Method#PRM}: a probabilistic roadmap (Kavraki et al., IEEE T-RA 1996) of free
 *       samples connected within a radius, queried by Dijkstra;</li>
 *   <li>{@link Method#RRT}: a rapidly-exploring random tree (LaValle 1998) with goal
 *       biasing;</li>
 *   <li>{@link Method#RRT_CONNECT}: two trees grown towards each other greedily (Kuffner
 *       &amp; LaValle 2000), usually the fastest to a first path;</li>
 *   <li>{@link Method#RRT_STAR}: RRT with choose-parent and rewiring within the shrinking radius
 *       {@code gamma (log n / n)^(1/d)}, asymptotically optimal (Karaman &amp; Frazzoli, IJRR
 *       2011, <a href="https://arxiv.org/abs/1105.1186">arXiv:1105.1186</a>);</li>
 *   <li>{@link Method#INFORMED_RRT_STAR}: RRT* that, once a path of cost {@code c} exists,
 *       samples only the prolate hyperspheroid of points that could improve it (Gammell et
 *       al., IROS 2014).</li>
 * </ul>
 *
 * Every planner takes a pluggable {@link Sampler}, so a learned sampling distribution
 * ({@link LearnedSampler}) drops in for the uniform one; collision checks are counted.
 */
public final class SamplingPlanners {

    private SamplingPlanners() {
    }

    /** The planner. */
    public enum Method {
        /** Probabilistic roadmap. */
        PRM,
        /** Rapidly-exploring random tree. */
        RRT,
        /** Bidirectional greedy RRT. */
        RRT_CONNECT,
        /** Asymptotically optimal RRT. */
        RRT_STAR,
        /** RRT* with informed sampling. */
        INFORMED_RRT_STAR
    }

    /** A configuration space with a collision checker. */
    public interface Space {

        /** @return the dimension */
        int dimension();

        /** @return the lower bounds */
        double[] lower();

        /** @return the upper bounds */
        double[] upper();

        /**
         * @param q a configuration
         * @return whether it is collision-free
         */
        boolean free(double[] q);
    }

    /** A sampling distribution over a space. */
    @FunctionalInterface
    public interface Sampler {

        /**
         * @param random the random source
         * @return a configuration
         */
        double[] sample(SplittableRandom random);

        /**
         * The uniform distribution over a space's box.
         *
         * @param space the space
         * @return the sampler
         */
        static Sampler uniform(Space space) {
            double[] lo = space.lower();
            double[] hi = space.upper();
            return r -> {
                double[] q = new double[lo.length];
                for (int i = 0; i < q.length; i++) {
                    q[i] = lo[i] + (hi[i] - lo[i]) * r.nextDouble();
                }
                return q;
            };
        }
    }

    /**
     * A planning result.
     *
     * @param path the configurations from start to goal, empty if none
     * @param cost the path length
     * @param firstSolution samples drawn until the first path, -1 if none
     * @param samples samples drawn in total
     * @param collisionChecks configurations checked for collision
     */
    public record Result(List<double[]> path, double cost, int firstSolution, int samples, long collisionChecks) {

        /** @return whether a path was found */
        public boolean found() {
            return !path.isEmpty();
        }
    }

    /**
     * A point robot in a 2-D field.
     *
     * @param field the field
     * @return the space
     */
    public static Space point(PlanningMaps.Field field) {
        return new Space() {
            @Override
            public int dimension() {
                return 2;
            }

            @Override
            public double[] lower() {
                return new double[] {0, 0};
            }

            @Override
            public double[] upper() {
                return new double[] {1, 1};
            }

            @Override
            public boolean free(double[] q) {
                return field.free(q[0], q[1]);
            }
        };
    }

    /**
     * A planar arm with its base at a point of a 2-D field; a configuration is the joint angles
     * (relative), each in {@code [-pi, pi]}, and it is free if every link segment (checked at
     * ten points per link) is inside the field and outside obstacles.
     *
     * @param field the field
     * @param base the base position
     * @param lengths the link lengths
     * @return the space
     */
    public static Space arm(PlanningMaps.Field field, double[] base, double[] lengths) {
        int d = lengths.length;
        return new Space() {
            @Override
            public int dimension() {
                return d;
            }

            @Override
            public double[] lower() {
                double[] lo = new double[d];
                Arrays.fill(lo, -Math.PI);
                return lo;
            }

            @Override
            public double[] upper() {
                double[] hi = new double[d];
                Arrays.fill(hi, Math.PI);
                return hi;
            }

            @Override
            public boolean free(double[] q) {
                double x = base[0];
                double y = base[1];
                double a = 0;
                for (int i = 0; i < d; i++) {
                    a += q[i];
                    double nx = x + lengths[i] * Math.cos(a);
                    double ny = y + lengths[i] * Math.sin(a);
                    for (int s = 1; s <= 10; s++) {
                        double t = s / 10.0;
                        if (!field.free(x + t * (nx - x), y + t * (ny - y))) {
                            return false;
                        }
                    }
                    x = nx;
                    y = ny;
                }
                return true;
            }
        };
    }

    /**
     * Plans a path.
     *
     * @param space the space
     * @param sampler the sampling distribution
     * @param start the start
     * @param goal the goal
     * @param method the planner
     * @param maxSamples the sample budget
     * @param step the RRT extension step (and collision-check resolution times 10)
     * @param goalBias the probability of sampling the goal (RRT family)
     * @param stopAtFirst stop as soon as a path exists (otherwise use the whole budget)
     * @param random the random source
     * @return the result
     */
    public static Result plan(Space space, Sampler sampler, double[] start, double[] goal, Method method,
            int maxSamples, double step, double goalBias, boolean stopAtFirst, SplittableRandom random) {
        Checker c = new Checker(space, step / 10);
        return switch (method) {
            case PRM -> prm(c, sampler, start, goal, maxSamples, step, stopAtFirst, random);
            case RRT_CONNECT -> connect(c, sampler, start, goal, maxSamples, step, random);
            default -> rrt(c, sampler, start, goal, method, maxSamples, step, goalBias, stopAtFirst, random);
        };
    }

    /** Counts collision checks and checks segments at a fixed resolution. */
    private static final class Checker {
        final Space space;
        final double resolution;
        long checks;

        Checker(Space space, double resolution) {
            this.space = space;
            this.resolution = resolution;
        }

        boolean free(double[] q) {
            checks++;
            return space.free(q);
        }

        boolean segment(double[] a, double[] b) {
            double d = distance(a, b);
            int n = Math.max(1, (int) Math.ceil(d / resolution));
            for (int i = 1; i <= n; i++) {
                if (!free(lerp(a, b, i / (double) n))) {
                    return false;
                }
            }
            return true;
        }
    }

    /**
     * @param a a configuration
     * @param b a configuration
     * @return the Euclidean distance
     */
    public static double distance(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return Math.sqrt(s);
    }

    private static double[] lerp(double[] a, double[] b, double t) {
        double[] q = new double[a.length];
        for (int i = 0; i < q.length; i++) {
            q[i] = a[i] + t * (b[i] - a[i]);
        }
        return q;
    }

    private static double[] steer(double[] from, double[] to, double step) {
        double d = distance(from, to);
        return d <= step ? to.clone() : lerp(from, to, step / d);
    }

    private static int nearest(List<double[]> nodes, double[] q) {
        int best = 0;
        double bd = Double.POSITIVE_INFINITY;
        for (int i = 0; i < nodes.size(); i++) {
            double d = distance(nodes.get(i), q);
            if (d < bd) {
                bd = d;
                best = i;
            }
        }
        return best;
    }

    private static double pathCost(List<double[]> path) {
        double c = 0;
        for (int i = 1; i < path.size(); i++) {
            c += distance(path.get(i - 1), path.get(i));
        }
        return c;
    }

    private static Result prm(Checker c, Sampler sampler, double[] start, double[] goal, int maxSamples, double step,
            boolean stopAtFirst, SplittableRandom random) {
        List<double[]> nodes = new ArrayList<>();
        List<List<int[]>> edges = new ArrayList<>();
        nodes.add(start.clone());
        nodes.add(goal.clone());
        edges.add(new ArrayList<>());
        edges.add(new ArrayList<>());
        int d = start.length;
        double radius = 3 * step;
        int first = -1;
        List<double[]> best = List.of();
        int drawn = 0;
        int checkEvery = 25;
        while (drawn < maxSamples) {
            double[] q = sampler.sample(random);
            drawn++;
            if (!c.free(q)) {
                continue;
            }
            int id = nodes.size();
            nodes.add(q);
            edges.add(new ArrayList<>());
            double r = Math.max(radius, 2.5 * Math.pow(Math.log(id + 1) / (id + 1), 1.0 / d));
            for (int j = 0; j < id; j++) {
                double dist = distance(nodes.get(j), q);
                if (dist <= r && c.segment(nodes.get(j), q)) {
                    edges.get(j).add(new int[] {id});
                    edges.get(id).add(new int[] {j});
                }
            }
            if (drawn % checkEvery == 0 || drawn == maxSamples) {
                List<double[]> p = dijkstra(nodes, edges);
                if (!p.isEmpty()) {
                    if (first < 0) {
                        first = drawn;
                    }
                    best = p;
                    if (stopAtFirst) {
                        break;
                    }
                }
            }
        }
        return new Result(best, best.isEmpty() ? Double.POSITIVE_INFINITY : pathCost(best), first, drawn, c.checks);
    }

    private static List<double[]> dijkstra(List<double[]> nodes, List<List<int[]>> edges) {
        int n = nodes.size();
        double[] dist = new double[n];
        Arrays.fill(dist, Double.POSITIVE_INFINITY);
        int[] parent = new int[n];
        Arrays.fill(parent, -1);
        dist[0] = 0;
        PriorityQueue<double[]> q = new PriorityQueue<>((a, b) -> Double.compare(a[0], b[0]));
        q.add(new double[] {0, 0});
        while (!q.isEmpty()) {
            double[] top = q.poll();
            int u = (int) top[1];
            if (top[0] > dist[u]) {
                continue;
            }
            if (u == 1) {
                break;
            }
            for (int[] e : edges.get(u)) {
                int v = e[0];
                double nd = dist[u] + distance(nodes.get(u), nodes.get(v));
                if (nd < dist[v]) {
                    dist[v] = nd;
                    parent[v] = u;
                    q.add(new double[] {nd, v});
                }
            }
        }
        if (Double.isInfinite(dist[1])) {
            return List.of();
        }
        List<double[]> path = new ArrayList<>();
        for (int v = 1; v != -1; v = parent[v]) {
            path.add(nodes.get(v));
        }
        Collections.reverse(path);
        return path;
    }

    private static Result rrt(Checker c, Sampler sampler, double[] start, double[] goal, Method method,
            int maxSamples, double step, double goalBias, boolean stopAtFirst, SplittableRandom random) {
        boolean star = method == Method.RRT_STAR || method == Method.INFORMED_RRT_STAR;
        boolean informed = method == Method.INFORMED_RRT_STAR;
        int d = start.length;
        List<double[]> nodes = new ArrayList<>();
        List<Integer> parent = new ArrayList<>();
        List<Double> cost = new ArrayList<>();
        nodes.add(start.clone());
        parent.add(-1);
        cost.add(0.0);
        int goalNode = -1;
        int first = -1;
        int drawn = 0;
        double cMin = distance(start, goal);
        double[][] frame = informed ? ellipseFrame(start, goal) : null;
        double[] lo = c.space.lower();
        double[] hi = c.space.upper();
        double gamma = 2 * Math.pow(1 + 1.0 / d, 1.0 / d) * Math.pow(volume(lo, hi), 1.0 / d);
        while (drawn < maxSamples) {
            drawn++;
            double cBest = goalNode >= 0 ? cost.get(goalNode) : Double.POSITIVE_INFINITY;
            double[] q;
            if (random.nextDouble() < goalBias) {
                q = goal.clone();
            } else if (informed && Double.isFinite(cBest)) {
                q = informedSample(start, goal, cBest, cMin, frame, random);
            } else {
                q = sampler.sample(random);
            }
            int near = nearest(nodes, q);
            double[] x = steer(nodes.get(near), q, step);
            if (!c.free(x) || !c.segment(nodes.get(near), x)) {
                continue;
            }
            int bestParent = near;
            double bestCost = cost.get(near) + distance(nodes.get(near), x);
            List<Integer> neighbours = new ArrayList<>();
            if (star) {
                double r = Math.min(gamma * Math.pow(Math.log(nodes.size() + 1) / (nodes.size() + 1), 1.0 / d),
                    3 * step);
                for (int j = 0; j < nodes.size(); j++) {
                    if (distance(nodes.get(j), x) <= r) {
                        neighbours.add(j);
                    }
                }
                for (int j : neighbours) {
                    double cc = cost.get(j) + distance(nodes.get(j), x);
                    if (cc < bestCost - 1e-12 && c.segment(nodes.get(j), x)) {
                        bestCost = cc;
                        bestParent = j;
                    }
                }
            }
            int id = nodes.size();
            nodes.add(x);
            parent.add(bestParent);
            cost.add(bestCost);
            if (star) {
                for (int j : neighbours) {
                    double cc = bestCost + distance(x, nodes.get(j));
                    if (cc < cost.get(j) - 1e-12 && c.segment(x, nodes.get(j))) {
                        double delta = cost.get(j) - cc;
                        parent.set(j, id);
                        propagate(j, delta, parent, cost);
                    }
                }
            }
            if (distance(x, goal) <= step && c.segment(x, goal)) {
                double gc = cost.get(id) + distance(x, goal);
                if (goalNode < 0) {
                    goalNode = nodes.size();
                    nodes.add(goal.clone());
                    parent.add(id);
                    cost.add(gc);
                    first = drawn;
                    if (stopAtFirst || !star) {
                        break;
                    }
                } else if (gc < cost.get(goalNode)) {
                    parent.set(goalNode, id);
                    cost.set(goalNode, gc);
                }
            }
        }
        if (goalNode < 0) {
            return new Result(List.of(), Double.POSITIVE_INFINITY, -1, drawn, c.checks);
        }
        List<double[]> path = new ArrayList<>();
        for (int v = goalNode; v != -1; v = parent.get(v)) {
            path.add(nodes.get(v));
        }
        Collections.reverse(path);
        return new Result(path, pathCost(path), first, drawn, c.checks);
    }

    private static void propagate(int root, double delta, List<Integer> parent, List<Double> cost) {
        cost.set(root, cost.get(root) - delta);
        List<Integer> stack = new ArrayList<>();
        stack.add(root);
        while (!stack.isEmpty()) {
            int u = stack.remove(stack.size() - 1);
            for (int v = 0; v < parent.size(); v++) {
                if (parent.get(v) == u && v != root) {
                    cost.set(v, cost.get(v) - delta);
                    stack.add(v);
                }
            }
        }
    }

    private static double volume(double[] lo, double[] hi) {
        double v = 1;
        for (int i = 0; i < lo.length; i++) {
            v *= hi[i] - lo[i];
        }
        return v;
    }

    private static double[][] ellipseFrame(double[] start, double[] goal) {
        int d = start.length;
        double cm = distance(start, goal);
        List<double[]> basis = new ArrayList<>();
        double[] a1 = new double[d];
        for (int i = 0; i < d; i++) {
            a1[i] = (goal[i] - start[i]) / cm;
        }
        basis.add(a1);
        for (int e = 0; e < d && basis.size() < d; e++) {
            double[] v = new double[d];
            v[e] = 1;
            for (double[] b : basis) {
                double p = GeometryLinalg.dot(v, b);
                for (int i = 0; i < d; i++) {
                    v[i] -= p * b[i];
                }
            }
            double n = GeometryLinalg.norm(v);
            if (n > 1e-6) {
                for (int i = 0; i < d; i++) {
                    v[i] /= n;
                }
                basis.add(v);
            }
        }
        double[][] frame = new double[d][d];
        for (int j = 0; j < d; j++) {
            for (int i = 0; i < d; i++) {
                frame[i][j] = basis.get(j)[i];
            }
        }
        return frame;
    }

    private static double[] informedSample(double[] start, double[] goal, double cBest, double cMin, double[][] frame,
            SplittableRandom random) {
        int d = start.length;
        double[] ball = new double[d];
        double nn = 0;
        for (int i = 0; i < d; i++) {
            ball[i] = random.nextGaussian();
            nn += ball[i] * ball[i];
        }
        double radius = Math.pow(random.nextDouble(), 1.0 / d) / Math.sqrt(nn);
        double r1 = cBest / 2;
        double rk = Math.sqrt(Math.max(0, cBest * cBest - cMin * cMin)) / 2;
        double[] scaled = new double[d];
        for (int i = 0; i < d; i++) {
            scaled[i] = ball[i] * radius * (i == 0 ? r1 : rk);
        }
        double[] q = GeometryLinalg.multiply(frame, scaled);
        for (int i = 0; i < d; i++) {
            q[i] += (start[i] + goal[i]) / 2;
        }
        return q;
    }

    private static Result connect(Checker c, Sampler sampler, double[] start, double[] goal, int maxSamples,
            double step, SplittableRandom random) {
        List<double[]> ta = new ArrayList<>();
        List<Integer> pa = new ArrayList<>();
        List<double[]> tb = new ArrayList<>();
        List<Integer> pb = new ArrayList<>();
        ta.add(start.clone());
        pa.add(-1);
        tb.add(goal.clone());
        pb.add(-1);
        boolean swapped = false;
        int drawn = 0;
        while (drawn < maxSamples) {
            drawn++;
            double[] q = sampler.sample(random);
            int added = extend(c, ta, pa, q, step);
            if (added >= 0) {
                double[] target = ta.get(added);
                int reached = -1;
                while (true) {
                    int before = tb.size();
                    int e = extend(c, tb, pb, target, step);
                    if (e < 0) {
                        break;
                    }
                    if (distance(tb.get(e), target) < 1e-12) {
                        reached = e;
                        break;
                    }
                    if (tb.size() == before) {
                        break;
                    }
                }
                if (reached >= 0) {
                    List<double[]> fromA = trace(ta, pa, added);
                    List<double[]> fromB = trace(tb, pb, reached);
                    Collections.reverse(fromA);
                    List<double[]> path = new ArrayList<>(fromA);
                    path.addAll(fromB.subList(1, fromB.size()));
                    if (swapped) {
                        Collections.reverse(path);
                    }
                    return new Result(path, pathCost(path), drawn, drawn, c.checks);
                }
            }
            List<double[]> t = ta;
            ta = tb;
            tb = t;
            List<Integer> p = pa;
            pa = pb;
            pb = p;
            swapped = !swapped;
        }
        return new Result(List.of(), Double.POSITIVE_INFINITY, -1, drawn, c.checks);
    }

    private static int extend(Checker c, List<double[]> tree, List<Integer> parent, double[] q, double step) {
        int near = nearest(tree, q);
        double[] x = steer(tree.get(near), q, step);
        if (!c.free(x) || !c.segment(tree.get(near), x)) {
            return -1;
        }
        tree.add(x);
        parent.add(near);
        return tree.size() - 1;
    }

    private static List<double[]> trace(List<double[]> tree, List<Integer> parent, int node) {
        List<double[]> out = new ArrayList<>();
        for (int v = node; v != -1; v = parent.get(v)) {
            out.add(tree.get(v));
        }
        return out;
    }
}
