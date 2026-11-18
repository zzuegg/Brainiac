package io.github.zzuegg.brainiac.experiments.robotics;

import io.github.zzuegg.brainiac.experimental.explore.StepwiseTd3;
import io.github.zzuegg.brainiac.experimental.planning.ControlBarrierFilter;
import io.github.zzuegg.brainiac.experimental.rl.ConstrainedPolicyOptimization;
import io.github.zzuegg.brainiac.experimental.rl.CostedEnvironment;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * W12 12.3.16: safe control of a point robot with double-integrator dynamics that must cross
 * the unit square past disc obstacles lying on the straight line to the goal. A control
 * barrier function filter ({@link ControlBarrierFilter}) wraps a nominal PD controller and an
 * RL learner (TD3, filtered during training as in Cheng et al., AAAI 2019); the baselines are
 * the unfiltered controllers and PPO-Lagrangian, which only learns to avoid cost. Each row
 * reports success, constraint violations (steps inside an obstacle) and return; the RL rows
 * also report the violations during training.
 */
public final class SafeControl {

    /** The time step. */
    public static final double DT = 0.05;

    /**
     * One method.
     *
     * @param method the method
     * @param success the share of episodes that reached the goal
     * @param violations the mean steps inside an obstacle per episode
     * @param meanReturn the mean return
     * @param trainingViolations steps inside an obstacle during training (0 for controllers)
     * @param interventions the share of steps the filter changed the action
     */
    public record Row(String method, double success, double violations, double meanReturn, double trainingViolations,
            double interventions) {
    }

    private SafeControl() {
    }

    /** @return the obstacles {@code (x, y, r)} */
    public static double[][] obstacles() {
        return new double[][] {{0.5, 0.5, 0.14}, {0.3, 0.3, 0.07}, {0.72, 0.66, 0.07}};
    }

    /**
     * Creates the environment.
     *
     * @param horizon the episode length
     * @return the environment
     */
    public static CostedEnvironment environment(int horizon) {
        return new Arena(obstacles(), horizon);
    }

    /** The double-integrator arena. Observation {@code (px, py, vx, vy, gx - px, gy - py)}. */
    private static final class Arena implements CostedEnvironment {

        private final double[][] discs;
        private final int horizon;
        private final double[] p = new double[2];
        private final double[] v = new double[2];
        private final double[] g = new double[2];
        private int t;
        private float cost;

        Arena(double[][] discs, int horizon) {
            this.discs = discs;
            this.horizon = horizon;
        }

        @Override
        public Space observationSpace() {
            return Space.box(6, -2f, 2f);
        }

        @Override
        public Space actionSpace() {
            return Space.box(2, -1f, 1f);
        }

        @Override
        public float[] reset(SplittableRandom random) {
            p[0] = 0.08;
            p[1] = 0.5 + random.nextDouble(-0.08, 0.08);
            g[0] = 0.92;
            g[1] = 0.5 + random.nextDouble(-0.08, 0.08);
            v[0] = 0;
            v[1] = 0;
            t = 0;
            cost = 0;
            return observation();
        }

        private float[] observation() {
            return new float[] {(float) p[0], (float) p[1], (float) v[0], (float) v[1], (float) (g[0] - p[0]),
                (float) (g[1] - p[1])};
        }

        @Override
        public Step step(float[] action) {
            double before = Math.hypot(g[0] - p[0], g[1] - p[1]);
            for (int i = 0; i < 2; i++) {
                double u = Math.max(-1, Math.min(1, action[i]));
                p[i] += v[i] * DT + 0.5 * u * DT * DT;
                v[i] += u * DT;
                v[i] = Math.max(-1, Math.min(1, v[i]));
                if (p[i] < 0 || p[i] > 1) {
                    p[i] = Math.max(0, Math.min(1, p[i]));
                    v[i] = 0;
                }
            }
            cost = 0;
            for (double[] d : discs) {
                if (Math.hypot(p[0] - d[0], p[1] - d[1]) < d[2]) {
                    cost = 1;
                }
            }
            t++;
            double after = Math.hypot(g[0] - p[0], g[1] - p[1]);
            boolean reached = after < 0.05;
            float reward = (float) (10 * (before - after) + (reached ? 1 : 0));
            return new Step(observation(), reward, reached, !reached && t >= horizon);
        }

        @Override
        public float lastCost() {
            return cost;
        }
    }

    /**
     * The nominal PD controller toward the goal.
     *
     * @param kp the position gain
     * @param kd the velocity gain
     * @return the controller
     */
    public static Agent pd(double kp, double kd) {
        return (o, explore) -> new float[] {(float) (kp * o[4] - kd * o[2]), (float) (kp * o[5] - kd * o[3])};
    }

    /**
     * The analytic or learned barrier filter around a policy.
     *
     * @param policy the policy
     * @param learned use the learned barrier
     * @param seed the seed
     * @return the filter
     */
    public static ControlBarrierFilter filter(Agent policy, boolean learned, long seed) {
        ControlBarrierFilter.Barrier b = learned ? learnedBarrier(seed)
            : ControlBarrierFilter.Barrier.analyticDiscs(obstacles(), 0.02);
        return ControlBarrierFilter.doubleIntegrator(policy, b, 4, 4, new double[] {-1, -1}, new double[] {1, 1}, 0, 2);
    }

    private static ControlBarrierFilter.LearnedBarrier learnedBarrier(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<double[]> safe = new ArrayList<>();
        List<double[]> unsafe = new ArrayList<>();
        double[][] discs = obstacles();
        while (safe.size() < 1500 || unsafe.size() < 400) {
            double[] q = {r.nextDouble(), r.nextDouble()};
            double clearance = Double.POSITIVE_INFINITY;
            for (double[] d : discs) {
                clearance = Math.min(clearance, Math.hypot(q[0] - d[0], q[1] - d[1]) - d[2]);
            }
            if (clearance > 0.03 && safe.size() < 1500) {
                safe.add(q);
            } else if (clearance < 0 && unsafe.size() < 400) {
                unsafe.add(q);
            }
        }
        ControlBarrierFilter.LearnedBarrier b = new ControlBarrierFilter.LearnedBarrier(32, seed);
        b.fit(safe, unsafe, 0.05f, 1500, 3e-3f, seed + 1);
        return b;
    }

    /**
     * Runs a policy.
     *
     * @param agent the agent
     * @param episodes episodes
     * @param seed the seed
     * @return {@code {success, violations, return}}
     */
    public static double[] evaluate(Agent agent, int episodes, long seed) {
        CostedEnvironment env = environment(200);
        SplittableRandom random = new SplittableRandom(seed);
        double ok = 0;
        double cost = 0;
        double ret = 0;
        for (int e = 0; e < episodes; e++) {
            agent.startEpisode();
            float[] o = env.reset(random.split());
            Environment.Step s;
            do {
                s = env.step(agent.act(o, false));
                ret += s.reward();
                cost += env.lastCost();
                o = s.observation();
            } while (!s.done());
            ok += s.terminated() ? 1 : 0;
        }
        return new double[] {ok / episodes, cost / episodes, ret / episodes};
    }

    /**
     * The PD controller with and without the analytic and learned filters.
     *
     * @param episodes evaluation episodes
     * @param seed the seed
     * @return PD, PD + analytic CBF, PD + learned CBF
     */
    public static List<Row> controllers(int episodes, long seed) {
        List<Row> rows = new ArrayList<>();
        Agent pd = pd(3, 2.5);
        double[] e = evaluate(pd, episodes, seed);
        rows.add(new Row("PD", e[0], e[1], e[2], 0, 0));
        for (boolean learned : new boolean[] {false, true}) {
            ControlBarrierFilter f = filter(pd, learned, seed);
            long[] steps = new long[1];
            e = evaluate(counting(f, steps), episodes, seed);
            double share = f.interventions() / (double) Math.max(1, steps[0]);
            rows.add(new Row(learned ? "PD + learned CBF" : "PD + analytic CBF", e[0], e[1], e[2], 0, share));
        }
        return rows;
    }

    private static Agent counting(Agent a, long[] steps) {
        return new Agent() {
            @Override
            public float[] act(float[] observation, boolean explore) {
                steps[0]++;
                return a.act(observation, explore);
            }

            @Override
            public void startEpisode() {
                a.startEpisode();
            }
        };
    }

    /**
     * TD3 unfiltered and filtered during training and evaluation, and PPO-Lagrangian.
     *
     * @param trainSteps environment steps per learner
     * @param episodes evaluation episodes
     * @param costLimit PPO-Lagrangian's cost limit per episode
     * @param seed the seed
     * @return TD3, TD3 + CBF, PPO-Lagrangian
     */
    public static List<Row> learners(int trainSteps, int episodes, float costLimit, long seed) {
        List<Row> rows = new ArrayList<>();
        for (boolean shielded : new boolean[] {false, true}) {
            StepwiseTd3 td3 = StepwiseTd3.of(6, 2, RlNetworkFactory.mlp(Activation.RELU, 32, 32), 1e-3f, seed)
                .discount(0.98f).learningStarts(500).batchSize(64);
            ControlBarrierFilter f = filter(td3::unitAction, false, seed);
            CostedEnvironment env = environment(150);
            SplittableRandom r = new SplittableRandom(seed);
            float[] o = env.reset(r.split());
            double violations = 0;
            for (int s = 0; s < trainSteps; s++) {
                float[] u;
                if (s < 500) {
                    u = new float[] {(float) r.nextDouble(-1, 1), (float) r.nextDouble(-1, 1)};
                } else {
                    u = td3.unitAction(o, true);
                }
                if (shielded) {
                    u = f.filter(o, u);
                }
                Environment.Step st = env.step(u);
                violations += env.lastCost();
                td3.remember(o, u, st.reward(), st.observation(), st.terminated());
                o = st.done() ? env.reset(r.split()) : st.observation();
                if (s >= 500) {
                    td3.update();
                }
            }
            Agent eval = shielded ? filter(td3::unitAction, false, seed) : td3::unitAction;
            double[] e = evaluate(eval, episodes, seed + 7);
            rows.add(new Row(shielded ? "TD3 + CBF" : "TD3", e[0], e[1], e[2], violations, 0));
        }
        ConstrainedPolicyOptimization l = ConstrainedPolicyOptimization.of(
            ConstrainedPolicyOptimization.Method.PPO_LAGRANGIAN, Nets.mlp(6, 2, seed, 32), Nets.mlp(6, 1, seed + 1, 32),
            Nets.mlp(6, 1, seed + 2, 32), Space.box(2, -1f, 1f), costLimit).rolloutSteps(500).lagrangeRate(0.5f)
            .seed(seed);
        CostedEnvironment env = environment(150);
        Counting counted = new Counting(env);
        l.fit(counted, trainSteps);
        double[] e = evaluate(l, episodes, seed + 7);
        rows.add(new Row("PPO-Lagrangian", e[0], e[1], e[2], counted.total, 0));
        return rows;
    }

    /** Counts the cost during training. */
    private static final class Counting implements CostedEnvironment {

        private final CostedEnvironment inner;
        private double total;

        Counting(CostedEnvironment inner) {
            this.inner = inner;
        }

        @Override
        public Space observationSpace() {
            return inner.observationSpace();
        }

        @Override
        public Space actionSpace() {
            return inner.actionSpace();
        }

        @Override
        public float[] reset(SplittableRandom random) {
            return inner.reset(random);
        }

        @Override
        public Step step(float[] action) {
            Step s = inner.step(action);
            total += inner.lastCost();
            return s;
        }

        @Override
        public float lastCost() {
            return inner.lastCost();
        }
    }
}
