package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.evolve.CmaEs;
import io.github.zzuegg.brainiac.experimental.rl.CemPlanner;
import io.github.zzuegg.brainiac.experimental.rl.ControlSystem;
import io.github.zzuegg.brainiac.experimental.rl.ControlSystems;
import io.github.zzuegg.brainiac.experimental.rl.DifferentiableSim;
import io.github.zzuegg.brainiac.experimental.rl.DomainRandomization;
import io.github.zzuegg.brainiac.experimental.rl.LinearQuadraticControl;
import io.github.zzuegg.brainiac.experimental.rl.ModelBasedDiffusionPlanner;
import io.github.zzuegg.brainiac.experimental.rl.MovementPrimitive;
import io.github.zzuegg.brainiac.experimental.rl.Pilco;
import io.github.zzuegg.brainiac.experimental.rl.RapidMotorAdaptation;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.Sac;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Ppo;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Classical and model-based control on small known-physics systems
 * ({@link ControlSystems}):
 * <ul>
 *   <li>planners with the true model: LQR, iLQR and DDP in receding-horizon (MPC) form, model-based
 *       diffusion, MPPI and the cross-entropy method, by cost, success and time per step
 *       (Li and Todorov 2004; Tassa et al. 2012; Pan et al. 2024,
 *       <a href="https://arxiv.org/abs/2407.01573">arXiv:2407.01573</a>);</li>
 *   <li>open-loop iLQR on the mass-spring reach of {@link DifferentiablePhysicsControl}, against
 *       its amortised backprop-through-the-simulator controller;</li>
 *   <li>sim-to-real: a policy trained on nominal physics, a domain-randomised one and rapid motor
 *       adaptation, on held-out physics and after a mid-episode change (Tobin et al. 2017,
 *       <a href="https://arxiv.org/abs/1703.06907">arXiv:1703.06907</a>; Kumar et al. 2021,
 *       <a href="https://arxiv.org/abs/2107.04034">arXiv:2107.04034</a>);</li>
 *   <li>data efficiency: PILCO (Deisenroth and Rasmussen 2011) against model-free SAC and PPO
 *       given ten times as many real steps.</li>
 *   <li>movement primitives (Ijspeert et al. 2013) against behaviour cloning on a 2-D reach to
 *       goals outside the demonstrations, and their reward-weighted refinement.</li>
 * </ul>
 */
public final class ClassicalControl {

    private ClassicalControl() {
    }

    /**
     * One planner's mean over the seeds.
     *
     * @param method the planner
     * @param cost the mean episode cost
     * @param success the fraction of episodes ending in the goal region
     * @param millisPerStep wall time per control step
     * @param rolloutsPerDecision model rollouts per decision (0 for the gradient planners)
     */
    public record PlannerScore(String method, double cost, double success, double millisPerStep,
            int rolloutsPerDecision) {
    }

    /** The tasks of the planner comparison. */
    public enum Task {
        /** Pendulum swing-up from hanging down. */
        PENDULUM_SWING_UP,
        /** Cart-pole balance from near upright. */
        CART_POLE_BALANCE,
        /** Point mass around a disc obstacle (a non-convex cost). */
        POINT_MASS
    }

    private static ControlSystem system(Task task) {
        return switch (task) {
            case PENDULUM_SWING_UP -> ControlSystems.squashed(ControlSystems.pendulum());
            case CART_POLE_BALANCE -> ControlSystems.cartPole(0.1, 0.5, 1, false);
            case POINT_MASS -> ControlSystems.pointMass();
        };
    }

    /**
     * The core {@link CemPlanner} on a {@link ControlSystem}'s true model.
     *
     * @param s the system
     * @param horizon planning horizon
     * @param samples candidates per round
     * @param rounds refinement rounds
     * @param seed the seed
     * @return the planner
     */
    public static ControlSystem.Planner cem(ControlSystem s, int horizon, int samples, int rounds, long seed) {
        int n = s.stateSize();
        int m = s.actionSize();
        CemPlanner p = CemPlanner.of((obs, act) -> {
            int b = obs.shape().dim(0);
            float[] o = obs.toArray();
            float[] a = act.toArray();
            float[] out = new float[b * n];
            for (int i = 0; i < b; i++) {
                double[] y = s.step(slice(o, i * n, n), slice(a, i * m, m));
                for (int j = 0; j < n; j++) {
                    out[i * n + j] = (float) y[j];
                }
            }
            return Tensor.of(Shape.of(b, n), out);
        }, (obs, act) -> -s.cost(slice(obs, 0, n), slice(act, 0, m)),
                Space.box(ControlSystem.toFloat(s.actionLow()), ControlSystem.toFloat(s.actionHigh())))
                .horizon(horizon).search(samples, Math.max(2, samples / 10), rounds).seed(seed);
        return new ControlSystem.Planner() {
            @Override
            public double[] act(double[] x) {
                float[] a = p.act(ControlSystem.toFloat(x), false);
                return slice(a, 0, a.length);
            }

            @Override
            public void reset() {
                p.startEpisode();
            }
        };
    }

    private static double[] slice(float[] v, int from, int len) {
        double[] out = new double[len];
        for (int i = 0; i < len; i++) {
            out[i] = v[from + i];
        }
        return out;
    }

    /** LQR on the linearisation at the origin (the upright, resting cart-pole). */
    private static ControlSystem.Planner lqr(ControlSystem s, double actionCost) {
        double[][][] ab = s.jacobians(new double[s.stateSize()], new double[s.actionSize()]);
        int n = s.stateSize();
        double[][] q = new double[n][n];
        double[] origin = new double[n];
        double[][] hess = ControlSystem.secondOrder(x -> s.cost(x, new double[s.actionSize()]), origin);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                q[i][j] = 0.5 * hess[1 + i][j];
            }
        }
        double[][] k = LinearQuadraticControl.lqr(ab[0], ab[1], q, new double[][] {{actionCost}}, 500);
        return x -> {
            double u = 0;
            for (int j = 0; j < n; j++) {
                u -= k[0][j] * x[j];
            }
            return new double[] {u};
        };
    }

    /**
     * Runs every planner on a task with the true model, closed loop.
     *
     * @param task the task
     * @param steps control steps per episode
     * @param horizon planning horizon
     * @param samples rollouts per level or round of the sampling planners
     * @param levels noise levels / rounds (sampling planners use {@code samples x levels}
     *     rollouts per decision)
     * @param seeds episodes (start states and planner seeds)
     * @return one row per planner
     */
    public static List<PlannerScore> planners(Task task, int steps, int horizon, int samples, int levels,
            int seeds) {
        ControlSystem s = system(task);
        List<String> names = new ArrayList<>(List.of("iLQR-MPC", "DDP-MPC", "MBD", "MPPI", "CEM"));
        if (task == Task.CART_POLE_BALANCE) {
            names.add(0, "LQR");
        }
        List<PlannerScore> out = new ArrayList<>();
        for (String name : names) {
            double cost = 0;
            double success = 0;
            double millis = 0;
            for (int seed = 1; seed <= seeds; seed++) {
                ControlSystem.Planner p = switch (name) {
                    case "LQR" -> lqr(s, 1e-4);
                    case "iLQR-MPC" -> LinearQuadraticControl.ilqr(s, horizon).mpc(2);
                    case "DDP-MPC" -> LinearQuadraticControl.ddp(s, horizon).mpc(2);
                    case "MBD" -> ModelBasedDiffusionPlanner.of(s, horizon, samples, levels, 0.1, seed);
                    case "MPPI" -> ModelBasedDiffusionPlanner.mppi(s, horizon, samples, levels, 0.3, 0.1, seed);
                    default -> cem(s, horizon, samples, levels, seed);
                };
                ControlSystem.Episode e = s.run(p, s.start(new SplittableRandom(seed)), steps);
                cost += e.cost() / seeds;
                success += (e.success() ? 1.0 : 0.0) / seeds;
                millis += e.millisPerStep() / seeds;
            }
            int rollouts = name.equals("MBD") || name.equals("MPPI") || name.equals("CEM") ? samples * levels : 0;
            out.add(new PlannerScore(name, cost, success, millis, rollouts));
        }
        return out;
    }

    /**
     * Open-loop iLQR on the mass-spring reach, one optimisation per goal.
     *
     * @param goals goals drawn as in {@link DifferentiablePhysicsControl}
     * @param iterations iLQR iterations
     * @param seed the goal seed
     * @return the mean final ball-to-goal distance
     */
    public static double massSpringIlqr(int goals, int iterations, long seed) {
        DifferentiableSim sim = DifferentiableSim.massSpring();
        float[] g = sim.goals(goals, new SplittableRandom(seed)).toArray();
        double total = 0;
        for (int i = 0; i < goals; i++) {
            ControlSystem s = ControlSystems.massSpring(sim, g[2 * i], g[2 * i + 1]);
            LinearQuadraticControl.Trajectory t = LinearQuadraticControl.ilqr(s, sim.horizon())
                    .optimize(s.start(new SplittableRandom(0)), null, iterations);
            double[] end = t.states()[sim.horizon()];
            total += Math.hypot(end[4] - g[2 * i], end[5] - g[2 * i + 1]);
        }
        return total / goals;
    }

    /**
     * The sim-to-real comparison, means over seeds.
     *
     * @param nominal a linear policy trained (CMA-ES) on the nominal physics only
     * @param randomised a linear policy trained on the randomised family
     * @param rmaOracle RMA's base policy given the true physics latent (the privileged bound)
     * @param rmaAdapted RMA with the latent estimated online from history
     * @param nominalAfterChange return over the 50 steps after a mid-episode physics change
     * @param randomisedAfterChange the same for the randomised policy
     * @param rmaAfterChange the same for adapted RMA
     */
    public record SimToReal(double nominal, double randomised, double rmaOracle, double rmaAdapted,
            double nominalAfterChange, double randomisedAfterChange, double rmaAfterChange) {
    }

    private static final double[] LOW = {0.05, 0.25, -1};
    private static final double[] HIGH = {0.5, 1.0, 1};
    private static final double[] NOMINAL = {0.1, 0.5, 0.26};
    private static final int EPISODE = 200;

    /**
     * Cart-pole balance whose pole mass, half-length and actuator gain are drawn per episode.
     * The gain is {@code sign(p) (0.3 + 2.7 |p|)}: the motor's polarity is unknown as well as
     * its strength, so one policy cannot fit the whole family. Falling (|angle| above 0.8 or
     * the cart off the track) ends the episode and charges a cost of 2 per lost step.
     */
    private static DomainRandomization family(double[] low, double[] high) {
        return DomainRandomization.of(p -> ControlSystems.cartPole(p[0], p[1],
                Math.signum(p[2]) * (0.3 + 2.7 * Math.abs(p[2])), false), low, high, EPISODE)
                .failWhen(x -> Math.abs(x[2]) > 0.8 || Math.abs(x[0]) > 2.4, 2);
    }

    /**
     * Return of a linear policy over the steps from {@code from} on (an episode that failed
     * earlier is charged the failure cost of those steps).
     */
    private static double linearReturn(DomainRandomization e, double[] w, SplittableRandom r, int from) {
        e.reset(r);
        double total = 0;
        Environment.Step s;
        do {
            double[] x = e.state();
            double u = w[4];
            for (int j = 0; j < 4; j++) {
                u += w[j] * x[j];
            }
            int t = e.time();
            s = e.step(new float[] {(float) Math.max(-10, Math.min(10, u))});
            if (t >= from) {
                total += s.reward();
            }
        } while (!s.done());
        if (e.time() < from) {
            total -= e.failureCost(from);
        }
        return total;
    }

    private static double[] trainLinear(DomainRandomization e, int evaluations, int episodes, long seed) {
        int[] generation = {0};
        return CmaEs.of(new double[5], 0.5).seed(seed).minimize(w -> {
            double total = 0;
            for (int k = 0; k < episodes; k++) {
                total += linearReturn(e, w, new SplittableRandom(seed * 7919 + generation[0] / 16 * 31 + k), 0);
            }
            generation[0]++;
            return -total / episodes;
        }, evaluations);
    }

    /**
     * Domain randomisation and RMA against a nominal-physics policy, on held-out episodes of the
     * randomised family, and over the last 50 steps of episodes whose physics are redrawn at
     * step 150 (the episode is cut at step 200; an episode that fell before the change is
     * charged the failure cost of those 50 steps).
     *
     * @param evaluations CMA-ES evaluations for every policy
     * @param episodes episodes per CMA-ES evaluation (randomised training)
     * @param tests held-out test episodes
     * @param seeds seeds
     * @return the means over seeds
     */
    public static SimToReal simToReal(int evaluations, int episodes, int tests, int seeds) {
        double[] sums = new double[7];
        for (long seed = 1; seed <= seeds; seed++) {
            double[] wn = trainLinear(family(NOMINAL, NOMINAL), evaluations, 1, seed);
            double[] wd = trainLinear(family(LOW, HIGH), evaluations, episodes, seed);
            DomainRandomization train = family(LOW, HIGH);
            RapidMotorAdaptation rma = RapidMotorAdaptation.of(train, 2, 8, seed).trainBase(evaluations, episodes)
                    .trainAdaptation(3, 30, 500);
            long testSeed = 1000 + seed;
            DomainRandomization test = family(LOW, HIGH);
            DomainRandomization change = family(LOW, HIGH).changeAt(150);
            for (int k = 0; k < tests; k++) {
                sums[0] += linearReturn(test, wn, new SplittableRandom(testSeed + 104729L * k), 0) / tests;
                sums[1] += linearReturn(test, wd, new SplittableRandom(testSeed + 104729L * k), 0) / tests;
                sums[4] += linearReturn(change, wn, new SplittableRandom(testSeed + 104729L * k), 150) / tests;
                sums[5] += linearReturn(change, wd, new SplittableRandom(testSeed + 104729L * k), 150) / tests;
            }
            sums[2] += rma.evaluate(tests, testSeed, false);
            sums[3] += rma.evaluate(tests, testSeed, true);
            train.changeAt(150);
            sums[6] += rma.evaluateFrom(tests, testSeed, 150);
        }
        for (int i = 0; i < sums.length; i++) {
            sums[i] /= seeds;
        }
        return new SimToReal(sums[0], sums[1], sums[2], sums[3], sums[4], sums[5], sums[6]);
    }

    private static final double[] PILCO_STD = {0.05, 0.05, 0.15, 0.05};
    private static final int PILCO_REPEAT = 3;
    private static final int PILCO_DECISIONS = 30;
    private static final int PILCO_TESTS = 10;

    /**
     * PILCO's learning curve on cart-pole balance from a Gaussian start (pole angle standard
     * deviation 0.15 rad), episodes of 90 steps (1.8 s); scores are the mean total system cost
     * of 10 test episodes (lower is better), means over seeds.
     *
     * @param zeroPolicy no force at all
     * @param curve the cost after each PILCO iteration
     * @param steps the real steps used by then (one random episode plus one per iteration)
     * @param lqr LQR on the true linearised model (the reference)
     */
    public record DataEfficiency(double zeroPolicy, double[] curve, long[] steps, double lqr) {
    }

    /**
     * Runs PILCO.
     *
     * @param iterations PILCO iterations (real episodes after the random one)
     * @param evaluations CMA-ES evaluations of PILCO's analytic objective per iteration
     * @param maxData the most transitions the GP models are conditioned on
     * @param seeds seeds
     * @return the means over seeds
     */
    public static DataEfficiency pilco(int iterations, int evaluations, int maxData, int seeds) {
        ControlSystem s = ControlSystems.cartPole(0.1, 0.5, 1, false);
        ControlSystem gaussian = ControlSystems.withStart(s, new double[4], PILCO_STD);
        int steps = PILCO_REPEAT * PILCO_DECISIONS;
        double zero = 0;
        double reference = 0;
        double[] curve = new double[iterations];
        long[] used = new long[iterations];
        ControlSystem.Planner lqr = lqr(s, 1e-2);
        for (long seed = 1; seed <= seeds; seed++) {
            Pilco p = Pilco.of(s, new double[4], PILCO_STD, PILCO_REPEAT, PILCO_DECISIONS, seed)
                    .saturatingCost(new double[4], new double[] {1, 0, 16, 0}).maxData(maxData).randomEpisodes(1);
            for (int k = 0; k < PILCO_TESTS; k++) {
                long test = 500 + seed * 31 + k;
                zero += p.episode(false, false, new SplittableRandom(test)) / PILCO_TESTS / seeds;
                reference += gaussian.run(lqr, gaussian.start(new SplittableRandom(test)), steps).cost()
                        / PILCO_TESTS / seeds;
            }
            for (int it = 0; it < iterations; it++) {
                p.learn(1, evaluations);
                for (int k = 0; k < PILCO_TESTS; k++) {
                    curve[it] += p.episode(false, false, new SplittableRandom(500 + seed * 31 + k)) / PILCO_TESTS
                            / seeds;
                }
                used[it] = p.realSteps();
            }
        }
        return new DataEfficiency(zero, curve, used, reference);
    }

    /**
     * Model-free baselines on the same task and test episodes as {@link #pilco}.
     *
     * @param sac SAC's mean test cost
     * @param ppo the core PPO's mean test cost
     */
    public record ModelFree(double sac, double ppo) {
    }

    /**
     * Trains SAC and the core PPO on the PILCO task.
     *
     * @param steps real environment steps
     * @param seeds seeds
     * @return the mean test costs over seeds
     */
    public static ModelFree modelFree(long steps, int seeds) {
        ControlSystem gaussian = ControlSystems.withStart(ControlSystems.cartPole(0.1, 0.5, 1, false), new double[4],
                PILCO_STD);
        int length = PILCO_REPEAT * PILCO_DECISIONS;
        Environment env = gaussian.asEnvironment(length);
        double sac = 0;
        double ppo = 0;
        for (long seed = 1; seed <= seeds; seed++) {
            Sac a = Sac.of(4, 1, RlNetworkFactory.mlp(Activation.RELU, 32, 32)).learningStarts(200)
                    .learningRate(3e-3f).seed(seed);
            a.fit(env, steps);
            Ppo b = Ppo.of(Nets.mlp(4, 1, seed, 32, 32), Nets.mlp(4, 1, seed + 1, 32, 32), env.actionSpace())
                    .rolloutSteps(length * 4).seed(seed);
            b.fit(env, steps);
            sac -= Rl.evaluate(a, env, PILCO_TESTS, 500 + seed * 31) / seeds;
            ppo -= Rl.evaluate(b, env, PILCO_TESTS, 500 + seed * 31) / seeds;
        }
        return new ModelFree(sac, ppo);
    }

    private static final int REACH_STEPS = 101;
    private static final double REACH_DT = 0.01;

    /**
     * A demonstrated reach from the origin to {@code goal}: a minimum-jerk progress along the
     * straight line plus a sideways arc of {@code bend} times the distance (as when reaching over
     * something), with Gaussian jitter of standard deviation {@code jitter} on every sample.
     */
    private static double[][] demo(double[] goal, double bend, double jitter, SplittableRandom r) {
        double[][] y = new double[REACH_STEPS][2];
        double len = Math.hypot(goal[0], goal[1]);
        double nx = -goal[1] / len;
        double ny = goal[0] / len;
        for (int t = 0; t < REACH_STEPS; t++) {
            double u = t / (double) (REACH_STEPS - 1);
            double m = u * u * u * (10 - 15 * u + 6 * u * u);
            double side = bend * len * Math.sin(Math.PI * m);
            boolean end = t == 0 || t == REACH_STEPS - 1;
            y[t][0] = m * goal[0] + side * nx + (end ? 0 : jitter * r.nextGaussian());
            y[t][1] = m * goal[1] + side * ny + (end ? 0 : jitter * r.nextGaussian());
        }
        return y;
    }

    private static double[] goal(SplittableRandom r, double minAngle, double maxAngle, double minR, double maxR) {
        double a = r.nextDouble(minAngle, maxAngle);
        double d = r.nextDouble(minR, maxR);
        return new double[] {d * Math.cos(a), d * Math.sin(a)};
    }

    /**
     * Movement primitives against behaviour cloning for one demonstration count, means over
     * seeds and 20 test goals each.
     *
     * @param demos demonstrations
     * @param dmpGoalError mean final distance to the goal of the DMP
     * @param bcGoalError the same for the behaviour-cloning MLP
     * @param dmpPathError mean distance from the noise-free demonstrator's path to the goal
     * @param bcPathError the same for behaviour cloning
     * @param dmpJerk RMS jerk of the DMP's trajectories
     * @param bcJerk RMS jerk of behaviour cloning's trajectories
     * @param demoJerk RMS jerk of the (jittered) demonstrations
     * @param idealJerk RMS jerk of the noise-free demonstrator on the test goals
     */
    public record PrimitiveScore(int demos, double dmpGoalError, double bcGoalError, double dmpPathError,
            double bcPathError, double dmpJerk, double bcJerk, double demoJerk, double idealJerk) {
    }

    /**
     * Fits a DMP (2 dimensions, 15 basis functions) and an MLP behaviour-cloning policy
     * {@code (position, goal, phase) -> velocity} to demonstrations whose goals lie at angles
     * 20 to 70 degrees and distances 0.8 to 1.2, then tests both on goals at 10 to 80 degrees
     * and 0.5 to 1.5 (mostly unseen).
     *
     * @param demos the demonstration count
     * @param bcSteps Adam steps for behaviour cloning
     * @param seeds seeds
     * @return the scores
     */
    public static PrimitiveScore primitives(int demos, int bcSteps, int seeds) {
        double[] sums = new double[8];
        int tests = 20;
        for (long seed = 1; seed <= seeds; seed++) {
            SplittableRandom r = new SplittableRandom(seed * 7);
            List<double[][]> data = new ArrayList<>();
            List<double[]> goals = new ArrayList<>();
            for (int k = 0; k < demos; k++) {
                double[] g = goal(r, 0.35, 1.22, 0.8, 1.2);
                goals.add(g);
                data.add(demo(g, r.nextDouble(0.25, 0.35), 0.002, r));
                sums[6] += MovementPrimitive.jerk(data.get(k), REACH_DT) / demos;
            }
            MovementPrimitive dmp = MovementPrimitive.of(2, 15, 25).fit(data, REACH_DT);
            Model bc = behaviourCloning(data, goals, bcSteps, seed);
            SplittableRandom tr = new SplittableRandom(seed * 13 + 1);
            for (int k = 0; k < tests; k++) {
                double[] g = goal(tr, 0.17, 1.4, 0.5, 1.5);
                double[][] ideal = demo(g, 0.3, 0, tr);
                double[][] a = dmp.rollout(new double[2], g, REACH_STEPS, REACH_DT);
                double[][] b = cloneRollout(bc, g);
                sums[0] += distance(a[REACH_STEPS - 1], g) / tests;
                sums[1] += distance(b[REACH_STEPS - 1], g) / tests;
                sums[2] += pathError(a, ideal) / tests;
                sums[3] += pathError(b, ideal) / tests;
                sums[4] += MovementPrimitive.jerk(a, REACH_DT) / tests;
                sums[5] += MovementPrimitive.jerk(b, REACH_DT) / tests;
                sums[7] += MovementPrimitive.jerk(ideal, REACH_DT) / tests;
            }
        }
        for (int i = 0; i < sums.length; i++) {
            sums[i] /= seeds;
        }
        return new PrimitiveScore(demos, sums[0], sums[1], sums[2], sums[3], sums[4], sums[5], sums[6],
                sums[7]);
    }

    private static double distance(double[] a, double[] b) {
        return Math.hypot(a[0] - b[0], a[1] - b[1]);
    }

    private static double pathError(double[][] a, double[][] b) {
        double total = 0;
        for (int t = 0; t < a.length; t++) {
            total += distance(a[t], b[t]) / a.length;
        }
        return total;
    }

    private static float[] cloneInput(double[] y, double[] g, int t) {
        return new float[] {(float) y[0], (float) y[1], (float) g[0], (float) g[1], t / (float) (REACH_STEPS - 1)};
    }

    private static Model behaviourCloning(List<double[][]> data, List<double[]> goals, int steps, long seed) {
        List<float[]> xs = new ArrayList<>();
        List<float[]> ys = new ArrayList<>();
        for (int k = 0; k < data.size(); k++) {
            double[][] y = data.get(k);
            for (int t = 0; t + 1 < y.length; t++) {
                xs.add(cloneInput(y[t], goals.get(k), t));
                ys.add(new float[] {(float) (y[t + 1][0] - y[t][0]) * 10, (float) (y[t + 1][1] - y[t][1]) * 10});
            }
        }
        Model net = Nets.mlp(5, 2, Activation.TANH, seed, 64, 64);
        Optimizer opt = Optimizer.adam(1e-3f);
        SplittableRandom r = new SplittableRandom(seed + 3);
        int batch = 64;
        for (int it = 0; it < steps; it++) {
            float[] in = new float[batch * 5];
            float[] out = new float[batch * 2];
            for (int b = 0; b < batch; b++) {
                int k = r.nextInt(xs.size());
                System.arraycopy(xs.get(k), 0, in, b * 5, 5);
                System.arraycopy(ys.get(k), 0, out, b * 2, 2);
            }
            net.forward(Var.of(Tensor.of(Shape.of(batch, 5), in)), true).sub(Tensor.of(Shape.of(batch, 2), out))
                    .square().mean().backward();
            opt.step(net.parameters());
        }
        return net;
    }

    private static double[][] cloneRollout(Model bc, double[] g) {
        double[][] y = new double[REACH_STEPS][];
        y[0] = new double[2];
        for (int t = 0; t + 1 < REACH_STEPS; t++) {
            float[] v = bc.forward(Var.of(Tensor.of(Shape.of(1, 5), cloneInput(y[t], g, t))), false).value().toArray();
            y[t + 1] = new double[] {y[t][0] + v[0] / 10, y[t][1] + v[1] / 10};
        }
        return y;
    }

    /**
     * Reward-weighted refinement of a one-demonstration DMP to pass through a via-point the
     * demonstration does not visit: the midpoint of the arc mirrored to the other side of the
     * straight line, for a new goal.
     *
     * @param iterations improvement iterations
     * @param rollouts rollouts per iteration
     * @param seeds seeds
     * @return {via-point error before, after, goal error after}, means over seeds
     */
    public static double[] refinement(int iterations, int rollouts, int seeds) {
        double[] out = new double[3];
        for (long seed = 1; seed <= seeds; seed++) {
            SplittableRandom r = new SplittableRandom(seed * 7);
            double[] g0 = goal(r, 0.35, 1.22, 0.8, 1.2);
            MovementPrimitive dmp = MovementPrimitive.of(2, 15, 25).fit(List.<double[][]>of(demo(g0, 0.3, 0.002, r)), REACH_DT);
            double[] g = goal(r, 0.35, 1.22, 0.8, 1.2);
            double[] via = demo(g, -0.2, 0, r)[REACH_STEPS / 2];
            int mid = REACH_STEPS / 2;
            ToDoubleFunction<double[][]> reward = y -> -distance(y[mid], via) - distance(y[REACH_STEPS - 1], g);
            double scale = 0;
            for (double w : dmp.weights()) {
                scale += w * w / dmp.weights().length;
            }
            out[0] += distance(dmp.rollout(new double[2], g, REACH_STEPS, REACH_DT)[mid], via) / seeds;
            MovementPrimitive better = dmp.improve(reward, new double[2], g, REACH_STEPS, REACH_DT, iterations,
                    rollouts, 0.3 * Math.sqrt(scale), 0.5, seed);
            double[][] y = better.rollout(new double[2], g, REACH_STEPS, REACH_DT);
            out[1] += distance(y[mid], via) / seeds;
            out[2] += distance(y[REACH_STEPS - 1], g) / seeds;
        }
        return out;
    }
}
