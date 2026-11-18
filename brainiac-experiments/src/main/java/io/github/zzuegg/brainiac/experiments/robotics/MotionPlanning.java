package io.github.zzuegg.brainiac.experiments.robotics;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.PlanningMaps;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.planning.GraphSearch;
import io.github.zzuegg.brainiac.experimental.planning.InverseKinematics;
import io.github.zzuegg.brainiac.experimental.planning.LearnedIk;
import io.github.zzuegg.brainiac.experimental.planning.LearnedSampler;
import io.github.zzuegg.brainiac.experimental.planning.SamplingPlanners;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Wave 12 items 12.3.14 and 12.3.15.
 *
 * <ul>
 *   <li>{@link #sampling}: PRM, RRT, RRT-Connect, RRT* and informed RRT* against grid A*
 *       (resolution-complete) for a point robot in narrow-passage fields and a 3-link planar
 *       arm among discs, by success, samples to the first path, path cost and collision
 *       checks;</li>
 *   <li>{@link #learnedSampling}: RRT with uniform samples against RRT whose samples come half
 *       from a CVAE trained on grid-optimal paths of other narrow-passage problems;</li>
 *   <li>{@link #inverseKinematics}: pseudo-inverse, damped least squares, CCD and FABRIK on 2-,
 *       3- and 7-link arms for random and near-singular targets;</li>
 *   <li>{@link #learnedIk}: a conditional coupling flow against an MLP regressor on a redundant
 *       3-link arm, by error, error after one damped least-squares step, and the number of
 *       elbow branches covered.</li>
 * </ul>
 */
public final class MotionPlanning {

    private MotionPlanning() {
    }

    /**
     * A sampling-planner row.
     *
     * @param problem the problem family
     * @param method the planner
     * @param success the share solved within the budget
     * @param firstSolution mean samples to the first path (solved problems)
     * @param cost mean path cost (solved problems)
     * @param collisionChecks mean collision checks
     */
    public record PlanRow(String problem, String method, double success, double firstSolution, double cost,
            double collisionChecks) {
    }

    /**
     * Sampling planners against grid A*.
     *
     * @param problems problems per family
     * @param budget samples per planner
     * @param seed the seed
     * @return rows
     */
    public static List<PlanRow> sampling(int problems, int budget, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<PlanRow> rows = new ArrayList<>();
        for (String family : new String[] {"point, narrow passage", "3-link arm"}) {
            boolean arm = family.startsWith("3");
            SamplingPlanners.Method[] methods = SamplingPlanners.Method.values();
            double[][] acc = new double[methods.length + 1][4];
            int[] solved = new int[methods.length + 1];
            for (int p = 0; p < problems; p++) {
                SamplingPlanners.Space space;
                double[] start;
                double[] goal;
                PlanningMaps.Field field;
                if (arm) {
                    field = PlanningMaps.clutter(4, r);
                    double[] base = {0.5, 0.5};
                    double[] lengths = {0.18, 0.14, 0.1};
                    field = withoutBase(field, base);
                    space = SamplingPlanners.arm(field, base, lengths);
                    start = freeConfig(space, r);
                    goal = freeConfig(space, r);
                } else {
                    field = PlanningMaps.narrowPassage(0.05, 4, r);
                    space = SamplingPlanners.point(field);
                    start = field.start();
                    goal = field.goal();
                }
                SamplingPlanners.Sampler uniform = SamplingPlanners.Sampler.uniform(space);
                double step = arm ? 0.3 : 0.05;
                for (int m = 0; m < methods.length; m++) {
                    boolean star = methods[m] == SamplingPlanners.Method.RRT_STAR
                        || methods[m] == SamplingPlanners.Method.INFORMED_RRT_STAR;
                    SamplingPlanners.Result res = SamplingPlanners.plan(space, uniform, start, goal, methods[m], budget,
                        step, 0.05, !star, new SplittableRandom(seed + 31L * p + m));
                    acc[m][3] += res.collisionChecks();
                    if (res.found()) {
                        solved[m]++;
                        acc[m][0] += res.firstSolution();
                        acc[m][1] += res.cost();
                    }
                }
                double[] g = arm ? gridArm(space, start, goal, 24, acc[methods.length])
                    : gridPoint(field, start, goal, 64, acc[methods.length]);
                if (g[0] > 0) {
                    solved[methods.length]++;
                    acc[methods.length][1] += g[1];
                }
            }
            for (int m = 0; m <= methods.length; m++) {
                String name = m < methods.length ? methods[m].name() : "grid A*";
                int s = Math.max(1, solved[m]);
                rows.add(new PlanRow(family, name, solved[m] / (double) problems,
                    m < methods.length ? acc[m][0] / s : Double.NaN, acc[m][1] / s, acc[m][3] / problems));
            }
        }
        return rows;
    }

    private static PlanningMaps.Field withoutBase(PlanningMaps.Field f, double[] base) {
        List<double[]> keep = new ArrayList<>();
        for (double[] d : f.discs()) {
            if (Math.hypot(d[0] - base[0], d[1] - base[1]) > d[2] + 0.05) {
                keep.add(d);
            }
        }
        return new PlanningMaps.Field(keep.toArray(new double[0][]), f.boxes(), f.start(), f.goal());
    }

    private static double[] freeConfig(SamplingPlanners.Space space, SplittableRandom r) {
        SamplingPlanners.Sampler u = SamplingPlanners.Sampler.uniform(space);
        while (true) {
            double[] q = u.sample(r);
            if (space.free(q)) {
                return q;
            }
        }
    }

    private static double[] gridPoint(PlanningMaps.Field field, double[] start, double[] goal, int cells,
            double[] acc) {
        PlanningMaps.Grid g = field.raster(cells);
        int s = cell(start[1], cells) * cells + cell(start[0], cells);
        int t = cell(goal[1], cells) * cells + cell(goal[0], cells);
        acc[3] += cells * cells;
        GraphSearch.Result res = GraphSearch.grid(g.blocked(), cells, cells, s, t, GraphSearch.Method.ASTAR,
            GraphSearch.Heuristic.OCTILE, 1, true);
        return new double[] {res.found() ? 1 : 0, res.cost() / cells};
    }

    private static int cell(double v, int cells) {
        return Math.max(0, Math.min(cells - 1, (int) Math.floor(v * cells)));
    }

    private static double[] gridArm(SamplingPlanners.Space space, double[] start, double[] goal, int bins,
            double[] acc) {
        double w = 2 * Math.PI / bins;
        int[] s = new int[3];
        int[] t = new int[3];
        for (int i = 0; i < 3; i++) {
            s[i] = Math.min(bins - 1, (int) Math.floor((start[i] + Math.PI) / w));
            t[i] = Math.min(bins - 1, (int) Math.floor((goal[i] + Math.PI) / w));
        }
        boolean[] free = new boolean[bins * bins * bins];
        for (int a = 0; a < bins; a++) {
            for (int b = 0; b < bins; b++) {
                for (int c = 0; c < bins; c++) {
                    free[(a * bins + b) * bins + c] = space.free(new double[] {-Math.PI + (a + 0.5) * w,
                        -Math.PI + (b + 0.5) * w, -Math.PI + (c + 0.5) * w});
                }
            }
        }
        acc[3] += free.length;
        GraphSearch.GenericResult<int[]> res = GraphSearch.search(s, q -> q[0] == t[0] && q[1] == t[1] && q[2] == t[2],
            q -> {
                List<int[]> out = new ArrayList<>(6);
                for (int d = 0; d < 3; d++) {
                    for (int sg = -1; sg <= 1; sg += 2) {
                        int[] n = q.clone();
                        n[d] += sg;
                        if (n[d] < 0 || n[d] >= bins || !free[(n[0] * bins + n[1]) * bins + n[2]]) {
                            continue;
                        }
                        out.add(n);
                    }
                }
                return out;
            }, q -> (q[0] * bins + q[1]) * bins + q[2],
            q -> Math.abs(q[0] - t[0]) + Math.abs(q[1] - t[1]) + Math.abs(q[2] - t[2]), GraphSearch.Method.ASTAR, 1,
            1_000_000);
        return new double[] {res.found() ? 1 : 0, res.cost() * w};
    }

    /**
     * A learned-sampler row.
     *
     * @param method the sampler
     * @param success the share solved within the budget
     * @param firstSolution mean samples to the first path (solved problems)
     */
    public record SamplerRow(String method, double success, double firstSolution) {
    }

    /** The problem encoding: occupancy along the wall line {@code x = 0.5} at 64 heights, start, goal. */
    private static double[] condition(PlanningMaps.Field f) {
        double[] c = new double[68];
        for (int i = 0; i < 64; i++) {
            c[i] = f.free(0.5, (i + 0.5) / 64) ? 0 : 1;
        }
        c[64] = f.start()[0];
        c[65] = f.start()[1];
        c[66] = f.goal()[0];
        c[67] = f.goal()[1];
        return c;
    }

    /**
     * Uniform against learned sampling for RRT in narrow passages.
     *
     * @param trainProblems training problems (grid-optimal paths)
     * @param testProblems test problems
     * @param budget samples per run
     * @param steps CVAE training steps
     * @param seed the seed
     * @return rows
     */
    public static List<SamplerRow> learnedSampling(int trainProblems, int testProblems, int budget, int steps,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double gap = 0.03;
        List<double[]> xs = new ArrayList<>();
        List<double[]> cs = new ArrayList<>();
        int cells = 64;
        for (int p = 0; p < trainProblems; p++) {
            PlanningMaps.Field f = PlanningMaps.narrowPassage(gap, 4, r);
            PlanningMaps.Grid g = f.raster(cells);
            int s = cell(f.start()[1], cells) * cells + cell(f.start()[0], cells);
            int t = cell(f.goal()[1], cells) * cells + cell(f.goal()[0], cells);
            GraphSearch.Result res = GraphSearch.grid(g.blocked(), cells, cells, s, t, GraphSearch.Method.ASTAR,
                GraphSearch.Heuristic.OCTILE, 1, true);
            double[] c = condition(f);
            for (int k = 0; k < res.path().size(); k += 2) {
                int cell = res.path().get(k);
                xs.add(new double[] {(cell % cells + 0.5) / cells, (cell / cells + 0.5) / cells});
                cs.add(c);
            }
        }
        LearnedSampler sampler = new LearnedSampler(2, 68, 2, 64, seed + 1);
        sampler.fit(xs, cs, steps, 128, 0.1f, 3e-3f, seed + 2);
        double[][] acc = new double[3][2];
        int[] solved = new int[3];
        for (int p = 0; p < testProblems; p++) {
            PlanningMaps.Field f = PlanningMaps.narrowPassage(gap, 4, r);
            SamplingPlanners.Space space = SamplingPlanners.point(f);
            SamplingPlanners.Sampler uniform = SamplingPlanners.Sampler.uniform(space);
            SamplingPlanners.Sampler[] samplers = {uniform, sampler.sampler(condition(f), uniform, 0.5),
                sampler.sampler(condition(f), uniform, 0.9)};
            for (int m = 0; m < samplers.length; m++) {
                SamplingPlanners.Result res = SamplingPlanners.plan(space, samplers[m], f.start(), f.goal(),
                    SamplingPlanners.Method.RRT, budget, 0.05, 0.05, true, new SplittableRandom(seed + 7L * p));
                if (res.found()) {
                    solved[m]++;
                    acc[m][0] += res.firstSolution();
                }
            }
        }
        String[] names = {"RRT uniform", "RRT learned 50%", "RRT learned 90%"};
        List<SamplerRow> rows = new ArrayList<>();
        for (int m = 0; m < 3; m++) {
            rows.add(new SamplerRow(names[m], solved[m] / (double) testProblems, acc[m][0] / Math.max(1, solved[m])));
        }
        return rows;
    }

    /**
     * An IK row.
     *
     * @param arm the arm
     * @param targets the target family
     * @param method the solver
     * @param success the share reaching 1e-3
     * @param iterations mean iterations
     */
    public record IkRow(String arm, String targets, String method, double success, double iterations) {
    }

    private static InverseKinematics arm(double[] lengths) {
        double[] lo = new double[lengths.length];
        double[] hi = new double[lengths.length];
        java.util.Arrays.fill(lo, Math.toRadians(-170));
        java.util.Arrays.fill(hi, Math.toRadians(170));
        return new InverseKinematics(lengths, lo, hi).maxStep(0.5);
    }

    /**
     * Numerical IK solvers.
     *
     * @param targets targets per family
     * @param seed the seed
     * @return rows
     */
    public static List<IkRow> inverseKinematics(int targets, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] arms = {{1, 1}, {1, 0.8, 0.6}, {0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3}};
        List<IkRow> rows = new ArrayList<>();
        for (double[] lengths : arms) {
            InverseKinematics ik = arm(lengths);
            double reach = 0;
            for (double l : lengths) {
                reach += l;
            }
            for (String family : new String[] {"random", "near-singular"}) {
                InverseKinematics.Method[] methods = InverseKinematics.Method.values();
                double[] success = new double[methods.length];
                double[] iters = new double[methods.length];
                for (int t = 0; t < targets; t++) {
                    double[] target;
                    if (family.equals("random")) {
                        double[] q = new double[lengths.length];
                        for (int j = 0; j < q.length; j++) {
                            q[j] = Math.toRadians(-160 + 320 * r.nextDouble());
                        }
                        target = ik.endEffector(q);
                    } else {
                        double a = 2 * Math.PI * r.nextDouble();
                        double d = 0.995 * reach;
                        target = new double[] {d * Math.cos(a), d * Math.sin(a)};
                    }
                    double[] init = new double[lengths.length];
                    for (int j = 0; j < init.length; j++) {
                        init[j] = 0.3 * r.nextGaussian();
                    }
                    for (int m = 0; m < methods.length; m++) {
                        InverseKinematics.Solution s = ik.solve(methods[m], target, init, 200, 1e-3, 0.05);
                        success[m] += s.error() < 1e-3 ? 1 : 0;
                        iters[m] += s.iterations();
                    }
                }
                for (int m = 0; m < methods.length; m++) {
                    rows.add(new IkRow(lengths.length + "-link", family, methods[m].name(), success[m] / targets,
                        iters[m] / targets));
                }
            }
        }
        return rows;
    }

    /**
     * A learned-IK row.
     *
     * @param method the model
     * @param error the mean end-effector error of the raw proposals
     * @param refined the mean error after one damped least-squares step
     * @param branches the mean number of distinct elbow-sign branches among refined proposals
     *     within 0.01
     */
    public record LearnedIkRow(String method, double error, double refined, double branches) {
    }

    /**
     * A conditional flow against an MLP regressor on a redundant 3-link arm.
     *
     * @param targets test targets
     * @param samples proposals per target
     * @param steps training steps for each model
     * @param seed the seed
     * @return rows
     */
    public static List<LearnedIkRow> learnedIk(int targets, int samples, int steps, long seed) {
        InverseKinematics ik = arm(new double[] {1, 0.8, 0.6});
        LearnedIk flow = new LearnedIk(ik, 4, 48, seed);
        flow.fit(steps, 128, 3e-3f, seed + 1);
        SimpleMlp mlp = SimpleMlp.tanh(seed + 2, 2, 64, 64, 3);
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom tr = new SplittableRandom(seed + 3);
        double[] lo = ik.lower();
        double[] hi = ik.upper();
        for (int s = 0; s < steps; s++) {
            int b = 256;
            float[] x = new float[b * 2];
            float[] y = new float[b * 3];
            for (int i = 0; i < b; i++) {
                double[] q = new double[3];
                for (int j = 0; j < 3; j++) {
                    q[j] = lo[j] + (hi[j] - lo[j]) * tr.nextDouble();
                    y[i * 3 + j] = (float) q[j];
                }
                double[] e = ik.endEffector(q);
                x[2 * i] = (float) e[0];
                x[2 * i + 1] = (float) e[1];
            }
            Var loss = mlp.forward(Var.of(Tensor.of(Shape.of(b, 2), x))).sub(Var.of(Tensor.of(Shape.of(b, 3), y)))
                .square().mean();
            loss.backward();
            opt.step(mlp.parameters());
        }
        SplittableRandom r = new SplittableRandom(seed + 4);
        double[][] acc = new double[2][3];
        for (int t = 0; t < targets; t++) {
            double[] q0 = new double[3];
            for (int j = 0; j < 3; j++) {
                q0[j] = lo[j] + (hi[j] - lo[j]) * r.nextDouble();
            }
            double[] target = ik.endEffector(q0);
            double[][] fq = flow.sample(target, samples, 0.8, r);
            float[] mq = mlp.predict(Tensor.of(Shape.of(1, 2), (float) target[0], (float) target[1])).toArray();
            double[][] mlpq = new double[samples][];
            for (int k = 0; k < samples; k++) {
                mlpq[k] = new double[] {mq[0], mq[1], mq[2]};
            }
            double[][][] sets = {fq, mlpq};
            for (int m = 0; m < 2; m++) {
                Set<String> branches = new HashSet<>();
                for (double[] q : sets[m]) {
                    acc[m][0] += ik.error(q, target) / samples;
                    double[] qq = q.clone();
                    ik.refine(qq, target, 0.05);
                    double e = ik.error(qq, target);
                    acc[m][1] += e / samples;
                    if (e < 0.01) {
                        branches.add((qq[1] > 0 ? "+" : "-") + (qq[2] > 0 ? "+" : "-"));
                    }
                }
                acc[m][2] += branches.size();
            }
        }
        List<LearnedIkRow> rows = new ArrayList<>();
        String[] names = {"conditional flow", "MLP regressor"};
        for (int m = 0; m < 2; m++) {
            rows.add(new LearnedIkRow(names[m], acc[m][0] / targets, acc[m][1] / targets, acc[m][2] / targets));
        }
        return rows;
    }
}
