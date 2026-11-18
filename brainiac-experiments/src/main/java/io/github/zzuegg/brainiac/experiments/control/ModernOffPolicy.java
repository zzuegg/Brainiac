package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.deeprl.Bro;
import io.github.zzuegg.brainiac.experimental.deeprl.ContinuousMountainCar;
import io.github.zzuegg.brainiac.experimental.deeprl.CrossQ;
import io.github.zzuegg.brainiac.experimental.deeprl.LearningCurveHarness;
import io.github.zzuegg.brainiac.experimental.deeprl.ModernSac;
import io.github.zzuegg.brainiac.experimental.deeprl.MrQ;
import io.github.zzuegg.brainiac.experimental.deeprl.SimbaV2;
import io.github.zzuegg.brainiac.experimental.deeprl.Td7;
import io.github.zzuegg.brainiac.experimental.rl.HighUtdSac;
import io.github.zzuegg.brainiac.experimental.rl.HlGaussValue;
import io.github.zzuegg.brainiac.experimental.rl.ResidualRlNetworks;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.Sac;
import io.github.zzuegg.brainiac.experimental.rl.Td3;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Pendulum;
import io.github.zzuegg.brainiac.rl.Ppo;
import io.github.zzuegg.brainiac.rl.RlListener;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Tracker;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * The 2023-2025 off-policy recipes for continuous control, compared on one learning-curve
 * harness ({@link LearningCurveHarness}: equal environment steps, several seeds, interquartile
 * means with bootstrap intervals, gradient updates and wall-clock per run):
 *
 * <ul>
 *   <li>{@link #crossQ}: CrossQ at one update per step against SAC, SAC with naively
 *       batch-normalised critics, the high-UTD REDQ and DroQ, and TD3 (W10 4.2).</li>
 *   <li>{@link #broScaling}: BRO against SAC and the wave 5 SimBa+HL-Gauss SAC as the critic
 *       widens, with BRO's optimism and resets ablated (W10 4.3).</li>
 *   <li>{@link #simbaV2Scaling}: SimbaV2 against SimBa as the replay ratio grows, with the critic
 *       parameter norm after training (W10 4.4).</li>
 *   <li>{@link #td7Ablation}: TD7 and each component removed, down to TD3 (W10 4.7).</li>
 *   <li>{@link #mrq}: MR.Q with and without its model-based losses against TD3 and SAC (W10 4.6).</li>
 *   <li>{@link #fairBudget}: the wave 1 continuous-control lineup (PPO, SAC, TD3, DDPG) re-run
 *       with the new recipes at budgets where the comparison means something, with Kendall's tau
 *       between budgets (W10 4.14).</li>
 * </ul>
 *
 * <p>Tasks: the pendulum swing-up (200 steps, return about -1,200 for random play) and the
 * continuous mountain car (sparse +100 at the flag, action cost). Test budgets are a few
 * thousand steps; {@link #main} runs the long curves.
 */
public final class ModernOffPolicy {

    private ModernOffPolicy() {
    }

    /** The tasks. */
    public enum Task {
        /** The pendulum swing-up, 200-step episodes. */
        PENDULUM(3, () -> new Pendulum(200)),
        /** The continuous mountain car, 300-step episodes. */
        MOUNTAIN_CAR(2, () -> new ContinuousMountainCar(300));

        private final int observations;
        private final Supplier<Environment> factory;

        Task(int observations, Supplier<Environment> factory) {
            this.observations = observations;
            this.factory = factory;
        }

        /** @return values per observation */
        public int observations() {
            return observations;
        }

        /** @return a fresh environment */
        public Environment create() {
            return factory.get();
        }
    }

    private static final float LR = 3e-3f;

    private static int warmup(long steps) {
        return (int) Math.min(500, Math.max(100, steps / 5));
    }

    /** A learner that acts at random, for the floor of every table. */
    static final class RandomLearner implements Learner {
        private final Agent agent;
        private final SplittableRandom random;

        RandomLearner(Environment env, long seed) {
            this.agent = Agent.random(env.actionSpace(), seed);
            this.random = new SplittableRandom(seed);
        }

        @Override
        public float[] act(float[] observation, boolean explore) {
            return agent.act(observation, explore);
        }

        @Override
        public RlResult fit(Environment env, long steps, RlListener listener) {
            Tracker t = new Tracker(listener);
            float[] obs = env.reset(random.split());
            while (t.steps() < steps) {
                Environment.Step s = env.step(agent.act(obs, true));
                obs = t.step(s) ? env.reset(random.split()) : s.observation();
            }
            return t.result();
        }
    }

    private static LearningCurveHarness harness(Task task, long[] budgets, int seeds) {
        return LearningCurveHarness.of(task::create).budgets(budgets).seeds(seeds).evalEpisodes(3);
    }

    private static ModernSac sac(Task task, int width, long steps, long seed) {
        RlNetworkFactory n = RlNetworkFactory.mlp(Activation.RELU, width, width);
        return ModernSac.of(task.observations(), 1, n, n).learningStarts(warmup(steps)).learningRate(LR).seed(seed);
    }

    private static Td3 td3(Task task, int width, long steps, long seed) {
        return Td3.td3(task.observations(), 1, RlNetworkFactory.mlp(Activation.RELU, width, width))
                .learningStarts(warmup(steps)).learningRate(LR).seed(seed);
    }

    /**
     * W10 4.2: CrossQ against SAC, naive batch normalisation, REDQ and DroQ at a high replay
     * ratio, and TD3.
     *
     * @param task the task
     * @param steps environment steps
     * @param seeds seeds
     * @param width hidden width
     * @param highUtd REDQ's and DroQ's updates per step
     * @return the curves (one checkpoint)
     */
    public static LearningCurveHarness.Report crossQ(Task task, long steps, int seeds, int width, int highUtd) {
        int obs = task.observations();
        int start = warmup(steps);
        return harness(task, new long[] {steps}, seeds)
                .add("SAC (UTD 1)", s -> sac(task, width, steps, s), ModernSac::updates)
                .add("CrossQ (UTD 1)", s -> CrossQ.learner(obs, 1, width, 2 * width).learningStarts(start)
                        .seed(s), ModernSac::updates)
                .add("SAC + naive BN critic", s -> CrossQ.naiveBatchNorm(obs, 1, width, 2 * width)
                        .learningStarts(start).seed(s), ModernSac::updates)
                .add("REDQ (UTD " + highUtd + ")", s -> HighUtdSac.redq(obs, 1, 5, 2, width, width)
                        .updateToData(highUtd).learningStarts(start).learningRate(LR).seed(s), HighUtdSac::updates)
                .add("DroQ (UTD " + highUtd + ")", s -> HighUtdSac.droq(obs, 1, 0.01f, width, width)
                        .updateToData(highUtd).learningStarts(start).learningRate(LR).seed(s), HighUtdSac::updates)
                .add("TD3", s -> td3(task, width, steps, s), Td3::updates)
                .run();
    }

    /**
     * W10 4.3: BRO against SAC and the wave 5 SimBa SAC with HL-Gauss critics as the critic
     * widens; at the widest critic, BRO without optimism and without resets.
     *
     * @param task the task
     * @param steps environment steps
     * @param seeds seeds
     * @param widths critic widths
     * @param replayRatio BRO's updates per step
     * @return the curves
     */
    public static LearningCurveHarness.Report broScaling(Task task, long steps, int seeds, int[] widths,
            int replayRatio) {
        int obs = task.observations();
        int start = warmup(steps);
        LearningCurveHarness h = harness(task, new long[] {steps}, seeds);
        int resetEvery = (int) Math.max(200, steps * replayRatio / 3);
        for (int w : widths) {
            h.add("SAC w" + w, s -> {
                RlNetworkFactory critic = RlNetworkFactory.mlp(Activation.RELU, w, w);
                return ModernSac.of(obs, 1, RlNetworkFactory.mlp(Activation.RELU, 32, 32), critic)
                        .learningStarts(start).learningRate(LR).seed(s);
            }, ModernSac::updates);
            h.add("SimBa+HL-Gauss SAC w" + w, s -> Sac.of(obs, 1, ResidualRlNetworks.simba(w, 1),
                    HlGaussValue.of(-200f, 10f, 51)).learningStarts(start).learningRate(LR).seed(s), Sac::updates);
            h.add("BRO w" + w, s -> Bro.learner(obs, 1, w, new Bro.Options(true, resetEvery, 8, replayRatio))
                    .learningStarts(start).learningRate(LR).seed(s), ModernSac::updates);
        }
        int wide = widths[widths.length - 1];
        h.add("BRO w" + wide + " no optimism", s -> Bro.learner(obs, 1, wide,
                new Bro.Options(false, resetEvery, 8, replayRatio)).learningStarts(start).learningRate(LR).seed(s),
                ModernSac::updates);
        h.add("BRO w" + wide + " no resets", s -> Bro.learner(obs, 1, wide, new Bro.Options(true, 0, 8, replayRatio))
                .learningStarts(start).learningRate(LR).seed(s), ModernSac::updates);
        return h.run();
    }

    /**
     * One row of the SimbaV2 scaling table.
     *
     * @param name the row
     * @param iqm the interquartile mean return over seeds
     * @param criticNorm the mean parameter norm of the first critic after training
     * @param updatesPerStep gradient updates per environment step
     */
    public record ScalingRow(String name, double iqm, double criticNorm, double updatesPerStep) {
    }

    /**
     * W10 4.4: SimbaV2 against SimBa (pre-LayerNorm residual SAC) at several replay ratios,
     * with the critics' parameter norm after training.
     *
     * @param task the task
     * @param steps environment steps
     * @param seeds seeds
     * @param width critic width
     * @param ratios replay ratios
     * @return the rows
     */
    public static Map<String, ScalingRow> simbaV2Scaling(Task task, long steps, int seeds, int width, int[] ratios) {
        int obs = task.observations();
        int start = warmup(steps);
        Map<String, ScalingRow> out = new LinkedHashMap<>();
        for (int r : ratios) {
            double[] simbaRet = new double[seeds];
            double[] simbaNorm = new double[seeds];
            double[] v2Ret = new double[seeds];
            double[] v2Norm = new double[seeds];
            for (int s = 0; s < seeds; s++) {
                long seed = s + 1;
                ModernSac simba = ModernSac.of(obs, 1, ResidualRlNetworks.simba(32, 1), ResidualRlNetworks.simba(width, 1))
                        .updateToData(r).learningStarts(start).learningRate(LR).seed(seed);
                simba.fit(task.create(), steps);
                simbaRet[s] = io.github.zzuegg.brainiac.rl.Rl.evaluate(simba, task.create(), 3, 10_000 + seed);
                simbaNorm[s] = SimbaV2.meanParameterNorm(simba.criticModels().get(0));
                ModernSac v2 = SimbaV2.learner(obs, 1, width, r).learningStarts(start).learningRate(LR).seed(seed);
                v2.fit(task.create(), steps);
                v2Ret[s] = io.github.zzuegg.brainiac.rl.Rl.evaluate(v2, task.create(), 3, 10_000 + seed);
                v2Norm[s] = SimbaV2.meanParameterNorm(v2.criticModels().get(0));
            }
            out.put("SimBa UTD " + r, new ScalingRow("SimBa UTD " + r, LearningCurveHarness.iqm(simbaRet),
                    mean(simbaNorm), r));
            out.put("SimbaV2 UTD " + r, new ScalingRow("SimbaV2 UTD " + r, LearningCurveHarness.iqm(v2Ret),
                    mean(v2Norm), r));
        }
        return out;
    }

    private static double mean(double[] x) {
        double s = 0;
        for (double v : x) {
            s += v;
        }
        return s / x.length;
    }

    /**
     * W10 4.7: TD7 with every component, then without each, and with none (TD3 with hard target
     * updates), evaluated at several checkpoints so the variance of the deployed policy can be
     * read off.
     *
     * @param task the task
     * @param budgets checkpoints
     * @param seeds seeds
     * @param width hidden width
     * @return the curves
     */
    public static LearningCurveHarness.Report td7Ablation(Task task, long[] budgets, int seeds, int width) {
        int obs = task.observations();
        long steps = budgets[budgets.length - 1];
        int start = warmup(steps);
        LearningCurveHarness h = harness(task, budgets, seeds);
        Map<String, Td7.Components> rows = new LinkedHashMap<>();
        rows.put("TD7", Td7.Components.all());
        rows.put("TD7 - SALE", new Td7.Components(false, true, true, true));
        rows.put("TD7 - LAP", new Td7.Components(true, false, true, true));
        rows.put("TD7 - clipping", new Td7.Components(true, true, false, true));
        rows.put("TD7 - checkpoints", new Td7.Components(true, true, true, false));
        rows.put("TD7 none (TD3, hard targets)", Td7.Components.none());
        rows.forEach((name, c) -> h.add(name, s -> Td7.of(obs, 1, width, 16, c).learningStarts(start)
                .learningRate(LR).targetEvery(100).seed(s), Td7::updates));
        h.add("TD3 (wave 1)", s -> td3(task, width, steps, s), Td3::updates);
        return h.run();
    }

    /**
     * W10 4.6 (continuous rows): MR.Q, MR.Q without the model-based losses, TD3 and SAC with one
     * configuration.
     *
     * @param task the task
     * @param steps environment steps
     * @param seeds seeds
     * @param width hidden width
     * @return the curves
     */
    public static LearningCurveHarness.Report mrq(Task task, long steps, int seeds, int width) {
        int obs = task.observations();
        int start = warmup(steps);
        return harness(task, new long[] {steps}, seeds)
                .add("MR.Q", s -> MrQ.continuous(obs, 1, width, 16).learningStarts(start).learningRate(LR)
                        .targetEvery(100).seed(s), MrQ::updates)
                .add("MR.Q - model losses", s -> MrQ.continuous(obs, 1, width, 16).modelLosses(false)
                        .learningStarts(start).learningRate(LR).targetEvery(100).seed(s), MrQ::updates)
                .add("TD3", s -> td3(task, width, steps, s), Td3::updates)
                .add("SAC", s -> sac(task, width, steps, s), ModernSac::updates)
                .run();
    }

    /**
     * W10 4.14: the wave 1 lineup and the wave 10 recipes on one harness at several budgets.
     *
     * @param task the task
     * @param budgets checkpoints, increasing
     * @param seeds seeds
     * @param width hidden width (critic widths of BRO and SimbaV2 are twice this)
     * @return the curves
     */
    public static LearningCurveHarness.Report fairBudget(Task task, long[] budgets, int seeds, int width) {
        int obs = task.observations();
        long steps = budgets[budgets.length - 1];
        int start = warmup(budgets[0]);
        RlNetworkFactory n = RlNetworkFactory.mlp(Activation.RELU, width, width);
        return harness(task, budgets, seeds)
                .add("random", s -> new RandomLearner(task.create(), s))
                .add("PPO (core)", s -> {
                    Model pi = Nets.mlp(obs, 1, s, width, width);
                    Model v = Nets.mlp(obs, 1, s + 1, width, width);
                    return Ppo.of(pi, v, task.create().actionSpace()).rolloutSteps(400).seed(s);
                })
                .add("SAC", s -> Sac.of(obs, 1, n).learningStarts(start).learningRate(LR).seed(s), Sac::updates)
                .add("TD3", s -> td3(task, width, steps, s).learningStarts(start), Td3::updates)
                .add("DDPG", s -> Td3.ddpg(obs, 1, n).learningStarts(start).learningRate(LR).seed(s), Td3::updates)
                .add("CrossQ", s -> CrossQ.learner(obs, 1, width, 2 * width).learningStarts(start).seed(s), ModernSac::updates)
                .add("BRO (RR 2)", s -> Bro.learner(obs, 1, 2 * width, new Bro.Options(true,
                        (int) Math.max(500, steps * 2 / 3), 8, 2)).learningStarts(start).learningRate(LR).seed(s),
                        ModernSac::updates)
                .add("SimbaV2", s -> SimbaV2.learner(obs, 1, 2 * width, 1).learningStarts(start).learningRate(LR)
                        .seed(s), ModernSac::updates)
                .add("TD7", s -> Td7.of(obs, 1, width, 16, Td7.Components.all()).learningStarts(start)
                        .learningRate(LR).targetEvery(100).seed(s), Td7::updates)
                .add("MR.Q", s -> MrQ.continuous(obs, 1, width, 16).learningStarts(start).learningRate(LR)
                        .targetEvery(100).seed(s), MrQ::updates)
                .run();
    }

    /**
     * Probability that row {@code x} beats row {@code y} at a checkpoint, over seeds.
     *
     * @param r the report
     * @param x the first row
     * @param y the second row
     * @param checkpoint the checkpoint index
     * @return the probability of improvement
     */
    public static double probabilityOfImprovement(LearningCurveHarness.Report r, String x, String y, int checkpoint) {
        return LearningCurveHarness.probabilityOfImprovement(new double[][] {r.curve(x).at(checkpoint)},
                new double[][] {r.curve(y).at(checkpoint)});
    }

    /**
     * The long runs: 5 seeds, budgets up to 20k steps on the pendulum and the mountain car.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        for (Task task : Task.values()) {
            LearningCurveHarness.Report fair = fairBudget(task, new long[] {2_000, 5_000, 10_000, 20_000}, 5, 32);
            System.out.println("W10 4.14 fair budget, " + task + ":\n" + fair.table());
            System.out.printf("Kendall tau 2k vs 20k %.2f, 5k vs 20k %.2f, 10k vs 20k %.2f%n", fair.rankStability(0, 3),
                    fair.rankStability(1, 3), fair.rankStability(2, 3));
            System.out.printf("P(SAC > random) at 20k %.2f, P(CrossQ > SAC) at 5k %.2f%n",
                    probabilityOfImprovement(fair, "SAC", "random", 3), probabilityOfImprovement(fair, "CrossQ", "SAC", 1));
        }
        System.out.println("W10 4.2 CrossQ, pendulum 10k:\n" + crossQ(Task.PENDULUM, 10_000, 5, 32, 10).table());
        System.out.println("W10 4.3 BRO widths, pendulum 10k:\n"
                + broScaling(Task.PENDULUM, 10_000, 5, new int[] {32, 64, 128}, 2).table());
        System.out.println("W10 4.3 BRO widths, mountain car 10k:\n"
                + broScaling(Task.MOUNTAIN_CAR, 10_000, 5, new int[] {32, 128}, 2).table());
        System.out.println("W10 4.4 SimbaV2 UTD, pendulum 10k: "
                + simbaV2Scaling(Task.PENDULUM, 10_000, 5, 64, new int[] {1, 2, 8}).values());
        System.out.println("W10 4.7 TD7 ablation, pendulum:\n"
                + td7Ablation(Task.PENDULUM, new long[] {5_000, 7_500, 10_000}, 5, 32).table());
        System.out.println("W10 4.6 MR.Q, pendulum 10k:\n" + mrq(Task.PENDULUM, 10_000, 5, 32).table());
        System.out.println("W10 4.6 MR.Q, mountain car 10k:\n" + mrq(Task.MOUNTAIN_CAR, 10_000, 5, 32).table());
    }
}
