package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.deeprl.LearningCurveHarness;
import io.github.zzuegg.brainiac.experimental.deeprl.Mbpo;
import io.github.zzuegg.brainiac.experimental.deeprl.ModernSac;
import io.github.zzuegg.brainiac.experimental.explore.Hiro;
import io.github.zzuegg.brainiac.experimental.explore.PointMaze;
import io.github.zzuegg.brainiac.experimental.rl.CemPlanner;
import io.github.zzuegg.brainiac.experimental.rl.HighUtdSac;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.Td3;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.Pendulum;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Model-based policy optimisation and goal-conditioned hierarchies (wave 11, 11.6.9-11.6.10).
 *
 * <ul>
 *   <li>{@link #mbpo}: pendulum return at equal environment steps for {@link Mbpo} (short
 *       branched rollouts from a probabilistic ensemble into SAC's buffer) against
 *       {@link ModernSac} at one and at five updates per step, DroQ at five, and a CEM planner on
 *       the true dynamics (PETS's planner with a perfect model, the ceiling);</li>
 *   <li>{@link #hierarchy}: success rate on the continuous {@link PointMaze} for {@link Hiro}
 *       (with and without off-policy correction), the HAC-style hindsight hierarchy, flat TD3 with
 *       hindsight relabelling, and the core TD3 without it.</li>
 * </ul>
 */
public final class ModelBasedAndHierarchical {

    private ModelBasedAndHierarchical() {
    }

    private static final float LR = 3e-3f;

    /**
     * 11.6.9: MBPO against model-free SAC at equal steps.
     *
     * @param steps environment steps
     * @param seeds seeds
     * @param width hidden width
     * @param utd the high update-to-data ratio of MBPO, SAC and DroQ
     * @return row to {IQM return, gradient updates per step, seconds per run}
     */
    public static Map<String, double[]> mbpo(long steps, int seeds, int width, int utd) {
        int start = (int) Math.min(400, Math.max(100, steps / 5));
        RlNetworkFactory n = RlNetworkFactory.mlp(Activation.RELU, width, width);
        LearningCurveHarness.Report report = LearningCurveHarness.of(() -> new Pendulum(200)).budgets(steps)
                .seeds(seeds).evalEpisodes(3)
                .add("SAC (UTD 1)", s -> ModernSac.of(3, 1, n, n).learningStarts(start).learningRate(LR).seed(s),
                        ModernSac::updates)
                .add("SAC (UTD " + utd + ")", s -> ModernSac.of(3, 1, n, n).updateToData(utd).learningStarts(start)
                        .learningRate(LR).seed(s), ModernSac::updates)
                .add("DroQ (UTD " + utd + ")", s -> HighUtdSac.droq(3, 1, 0.01f, width, width).updateToData(utd)
                        .learningStarts(start).learningRate(LR).seed(s), HighUtdSac::updates)
                .add("MBPO (UTD " + utd + ")", s -> {
                    Mbpo m = Mbpo.of(3, 1, n, n, 5, 64, s).rolloutLength(1, 3).chunk(200);
                    m.sac().updateToData(utd).learningStarts(start).learningRate(LR);
                    return m;
                }, Mbpo::updates)
                .run();
        Map<String, double[]> out = new LinkedHashMap<>();
        report.curves().forEach((name, c) -> out.put(name, new double[] {c.finalIqm(), c.updatesPerStep(),
            c.meanSeconds()}));
        double[] cem = new double[seeds];
        for (int s = 0; s < seeds; s++) {
            cem[s] = Rl.evaluate(trueModelCem(s), new Pendulum(200), 1, 1_000 + s);
        }
        out.put("CEM (true model)", new double[] {LearningCurveHarness.iqm(cem), 0, 0});
        return out;
    }

    private static double wrap(double angle) {
        return ((angle + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    }

    /** A CEM planner on the pendulum's own equations. */
    static Agent trueModelCem(long seed) {
        return CemPlanner.of((obs, act) -> {
            int b = obs.shape().dim(0);
            float[] o = obs.toArray();
            float[] a = act.toArray();
            float[] out = new float[b * 3];
            for (int i = 0; i < b; i++) {
                double angle = Math.atan2(o[i * 3 + 1], o[i * 3]);
                double u = Math.max(-2, Math.min(2, a[i]));
                double speed = o[i * 3 + 2] + (15 * Math.sin(angle) + 3 * u) * 0.05;
                speed = Math.max(-8, Math.min(8, speed));
                angle += speed * 0.05;
                out[i * 3] = (float) Math.cos(angle);
                out[i * 3 + 1] = (float) Math.sin(angle);
                out[i * 3 + 2] = (float) speed;
            }
            return Tensor.of(Shape.of(b, 3), out);
        }, (obs, act) -> {
            double th = wrap(Math.atan2(obs[1], obs[0]));
            double u = Math.max(-2, Math.min(2, act[0]));
            return -(th * th + 0.1 * obs[2] * obs[2] + 0.001 * u * u);
        }, (Space.Box) new Pendulum(200).actionSpace()).horizon(15).search(100, 10, 3).seed(seed);
    }

    private static double successRate(Agent agent, Environment env, int episodes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int ok = 0;
        for (int e = 0; e < episodes; e++) {
            float[] obs = env.reset(r.split());
            agent.startEpisode();
            Environment.Step s;
            do {
                s = env.step(agent.act(obs, false));
                obs = s.observation();
            } while (!s.done());
            if (s.terminated()) {
                ok++;
            }
        }
        return ok / (double) episodes;
    }

    /**
     * 11.6.10: hierarchies against flat learners on a point maze with a sparse reward.
     *
     * @param layout the maze
     * @param maxSteps steps per episode
     * @param steps environment steps per learner
     * @param seeds seeds
     * @param width hidden width of every network
     * @param modes the {@link Hiro} modes to run
     * @param coreTd3 also run the core TD3 without hindsight
     * @return row to mean success rate over seeds (5 greedy episodes each)
     */
    public static Map<String, Double> hierarchy(PointMaze.Layout layout, int maxSteps, long steps, int seeds,
            int width, Hiro.Mode[] modes, boolean coreTd3) {
        Map<String, Double> out = new LinkedHashMap<>();
        RlNetworkFactory n = RlNetworkFactory.mlp(Activation.RELU, width, width);
        for (Hiro.Mode m : modes) {
            double total = 0;
            for (int s = 0; s < seeds; s++) {
                PointMaze env = maze(layout, maxSteps);
                Hiro h = Hiro.of(m, env, n, 10, s);
                h.fit(env, steps, null);
                total += successRate(h, maze(layout, maxSteps), 5, 100 + s);
            }
            out.put(m.toString(), total / seeds);
        }
        if (!coreTd3) {
            return out;
        }
        double total = 0;
        for (int s = 0; s < seeds; s++) {
            PointMaze env = maze(layout, maxSteps);
            Learner td3 = Td3.td3(4, 2, n).learningStarts(500).learningRate(1e-3f).seed(s);
            td3.fit(env, steps, null);
            total += successRate(td3, maze(layout, maxSteps), 5, 100 + s);
        }
        out.put("TD3 (core, no hindsight)", total / seeds);
        return out;
    }

    private static PointMaze maze(PointMaze.Layout layout, int maxSteps) {
        return layout == PointMaze.Layout.U_MAZE ? PointMaze.uMaze(maxSteps)
                : new PointMaze(layout, maxSteps, new float[] {-0.7f, -0.7f}, null);
    }

    /**
     * Prints the tables at longer budgets.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        System.out.println(fmt(mbpo(5_000, 3, 64, 5)));
        System.out.println(hierarchy(PointMaze.Layout.U_MAZE, 100, 20_000, 3, 64, Hiro.Mode.values(), true));
    }

    /**
     * @param table a result table
     * @return one line
     */
    public static String fmt(Map<String, double[]> table) {
        StringBuilder sb = new StringBuilder();
        table.forEach((k, v) -> sb.append(k).append(String.format(Locale.ROOT, "=%.1f(%.1fupd/step,%.1fs) ", v[0],
                v[1], v[2])));
        return sb.toString().trim();
    }
}
