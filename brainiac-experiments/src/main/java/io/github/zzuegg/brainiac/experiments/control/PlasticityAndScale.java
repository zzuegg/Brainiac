package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.deeprl.Bro;
import io.github.zzuegg.brainiac.experimental.deeprl.LearningCurveHarness;
import io.github.zzuegg.brainiac.experimental.deeprl.ModernSac;
import io.github.zzuegg.brainiac.experimental.deeprl.MultiSkipNetwork;
import io.github.zzuegg.brainiac.experimental.deeprl.Plasticity;
import io.github.zzuegg.brainiac.experimental.deeprl.SelfPredictive;
import io.github.zzuegg.brainiac.experimental.deeprl.SimbaV2;
import io.github.zzuegg.brainiac.experimental.deeprl.ValueScalingStudy;
import io.github.zzuegg.brainiac.experimental.eval.CurveExtrapolator;
import io.github.zzuegg.brainiac.experimental.rl.ResidualRlNetworks;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.train.PsgdKron;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.CartPole;
import io.github.zzuegg.brainiac.rl.Dqn;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.GridWorld;
import io.github.zzuegg.brainiac.rl.Pendulum;
import io.github.zzuegg.brainiac.rl.Ppo;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Schedule;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Plasticity and scale in value-based deep RL:
 *
 * <ul>
 *   <li>{@link #plasticity}: dormant neurons, feature rank and return of a DQN at several replay
 *       ratios with no intervention, ReDo, last-layer resets, plasticity injection,
 *       shrink-and-perturb (the {@code WarmStartGap} operation) and BBF-style resets; and
 *       {@link #taskSwitch}, the same interventions when the goal moves mid-training (W10 4.10).</li>
 *   <li>{@link #depth}: DQN, PPO and SAC with plain deep MLPs, SimBa residual networks and
 *       multi-skip networks, each with Adam or the Kronecker-factored PSGD optimizer, at several
 *       depths, with the gradient-norm spread across layers (W10 4.11).</li>
 *   <li>{@link #scaling}: held-out TD error over a width x batch-size grid of TD critics, the
 *       passive-critic experiment and the power-law fits (W10 4.17); {@link #onlineScaling} is
 *       the SAC return grid with BRO and SimbaV2 as regularised points (Run only).</li>
 * </ul>
 */
public final class PlasticityAndScale {

    private PlasticityAndScale() {
    }

    /** The interventions against plasticity loss. */
    public enum Intervention {
        /** Nothing. */
        NONE,
        /** ReDo every few hundred updates (tau 0.1). */
        REDO,
        /** Reset the Q head's last two layers periodically. */
        RESET_LAST,
        /** Plasticity injection once, at a third of training. */
        INJECTION,
        /** Shrink-and-perturb encoder and head periodically (alpha 0.8). */
        SHRINK_PERTURB,
        /** BBF-style resets (fresh head, encoder shrunk halfway). */
        BBF_RESETS
    }

    /**
     * One plasticity run's outcome.
     *
     * @param ret the greedy return at the end (IQM over seeds)
     * @param dormant the dormant share of the encoder and head at the end (mean)
     * @param featureRank the head's effective feature rank at the end (mean)
     * @param weightNorm the head's weight norm at the end (mean)
     */
    public record Outcome(double ret, double dormant, double featureRank, double weightNorm) {
    }

    private static SelfPredictive dqn(int obs, int actions, long steps, int replayRatio, long seed) {
        return SelfPredictive.of(obs, actions, SelfPredictive.mlpEncoder(64), 32, 64)
                .learningStarts((int) Math.min(500, steps / 5)).epsilon((int) (steps / 3), 0.05f)
                .replayRatio(replayRatio).seed(seed);
    }

    private static Consumer<SelfPredictive> hook(Intervention iv, long totalUpdates, long seed) {
        int every = (int) Math.max(100, totalUpdates / 4);
        long injectAt = Math.max(1, totalUpdates / 3);
        Optimizer none = null;
        return agent -> {
            long u = agent.updates();
            switch (iv) {
                case REDO -> {
                    if (u % every == 0) {
                        Tensor batch = agent.sampleObservations(128);
                        if (batch != null) {
                            Plasticity.redo(agent.encoder(), batch, 0.1f, none, seed + u);
                            Plasticity.redo(agent.head(), agent.encoder().predict(batch), 0.1f, none, seed + u + 1);
                        }
                    }
                }
                case RESET_LAST -> {
                    if (u % every == 0) {
                        Plasticity.resetLast(agent.head(), 2, seed + u);
                    }
                }
                case INJECTION -> {
                    if (u == injectAt && !agent.injected()) {
                        agent.injectPlasticity();
                    }
                }
                case SHRINK_PERTURB -> {
                    if (u % every == 0) {
                        SelfPredictive.shrinkPerturb(agent.encoder(), SelfPredictive.mlpEncoder(64)
                                .create(agent.encoder().inputSize(), 32, seed + u), 0.8f);
                    }
                }
                default -> { }
            }
        };
    }

    private static SelfPredictive withIntervention(SelfPredictive agent, Intervention iv, long totalUpdates,
            long seed) {
        if (iv == Intervention.BBF_RESETS) {
            return agent.resets((int) Math.max(100, totalUpdates / 4), 0.5f);
        }
        return agent.onUpdate(hook(iv, totalUpdates, seed));
    }

    private static double[] diagnose(SelfPredictive agent) {
        Tensor batch = agent.sampleObservations(256);
        if (batch == null) {
            return new double[] {Double.NaN, Double.NaN, Double.NaN};
        }
        double dormant = Plasticity.dormantFraction(agent.encoder(), batch, 0.1f);
        Tensor z = agent.encoder().predict(batch);
        double rank = Plasticity.featureRank(agent.head(), z);
        return new double[] {dormant, rank, Plasticity.weightNorm(agent.head())};
    }

    /**
     * W10 4.10: interventions against plasticity loss at several replay ratios.
     *
     * @param env builds the task
     * @param steps environment steps
     * @param replayRatios replay ratios
     * @param seeds seeds
     * @return "intervention @ RR k" to outcome
     */
    public static Map<String, Outcome> plasticity(Supplier<Environment> env, long steps, int[] replayRatios,
            int seeds) {
        Environment probe = env.get();
        int obs = probe.observationSpace().size();
        int actions = ((io.github.zzuegg.brainiac.rl.Space.Discrete) probe.actionSpace()).n();
        Map<String, Outcome> out = new LinkedHashMap<>();
        for (int rr : replayRatios) {
            for (Intervention iv : Intervention.values()) {
                double[] ret = new double[seeds];
                double[] diag = new double[3];
                for (int s = 0; s < seeds; s++) {
                    long seed = s + 1;
                    long totalUpdates = steps * rr;
                    SelfPredictive agent = withIntervention(dqn(obs, actions, steps, rr, seed), iv, totalUpdates, seed);
                    agent.fit(env.get(), steps);
                    ret[s] = Rl.evaluate(agent, env.get(), 5, 10_000 + seed);
                    double[] d = diagnose(agent);
                    for (int k = 0; k < 3; k++) {
                        diag[k] += d[k] / seeds;
                    }
                }
                out.put(iv + " @ RR " + rr, new Outcome(LearningCurveHarness.iqm(ret), diag[0], diag[1], diag[2]));
            }
        }
        return out;
    }

    /** @return the first grid of the task switch: goal bottom right */
    public static GridWorld gridA() {
        return GridWorld.of("S....", ".....", "....G").maxSteps(20);
    }

    /** @return the second grid: the goal moves to the bottom left */
    public static GridWorld gridB() {
        return GridWorld.of("S....", ".....", "G....").maxSteps(20);
    }

    /**
     * W10 4.10, task switch: train on {@link #gridA()}, then continue on {@link #gridB()}; the
     * return on B measures how well each intervention keeps the network able to learn.
     *
     * @param stepsPerTask environment steps on each grid
     * @param replayRatio the replay ratio
     * @param seeds seeds
     * @return intervention to outcome on grid B
     */
    public static Map<String, Outcome> taskSwitch(long stepsPerTask, int replayRatio, int seeds) {
        Map<String, Outcome> out = new LinkedHashMap<>();
        for (Intervention iv : Intervention.values()) {
            double[] ret = new double[seeds];
            double[] diag = new double[3];
            for (int s = 0; s < seeds; s++) {
                long seed = s + 1;
                long totalUpdates = 2 * stepsPerTask * replayRatio;
                SelfPredictive agent = withIntervention(dqn(15, 4, stepsPerTask, replayRatio, seed)
                        .epsilon((int) (stepsPerTask / 3), 0.1f), iv, totalUpdates, seed);
                agent.fit(gridA(), stepsPerTask);
                agent.fit(gridB(), stepsPerTask);
                ret[s] = Rl.evaluate(agent, gridB(), 1, 10_000 + seed);
                double[] d = diagnose(agent);
                for (int k = 0; k < 3; k++) {
                    diag[k] += d[k] / seeds;
                }
            }
            out.put(iv.toString(), new Outcome(LearningCurveHarness.iqm(ret), diag[0], diag[1], diag[2]));
        }
        return out;
    }

    /** The network families of the depth study. */
    public enum Family {
        /** A plain MLP of the given depth. */
        PLAIN,
        /** SimBa residual blocks (pre-LayerNorm). */
        SIMBA,
        /** Multi-skip: the embedding broadcast to every layer. */
        MULTI_SKIP
    }

    /** The agents of the depth study. */
    public enum Algorithm {
        /** Core DQN on cart-pole. */
        DQN,
        /** Core PPO on cart-pole. */
        PPO,
        /** SAC on the pendulum. */
        SAC
    }

    /**
     * One depth-study cell.
     *
     * @param ret the IQM greedy return
     * @param gradientSpread the largest over the smallest per-layer gradient norm (multi-skip and
     *     plain trunks; NaN for SimBa)
     */
    public record DepthOutcome(double ret, double gradientSpread) {
    }

    private static RlNetworkFactory family(Family f, int width, int depth) {
        return switch (f) {
            case PLAIN -> MultiSkipNetwork.factory(width, depth, false);
            case MULTI_SKIP -> MultiSkipNetwork.factory(width, depth, true);
            case SIMBA -> ResidualRlNetworks.simba(width, Math.max(1, depth / 2));
        };
    }

    /**
     * W10 4.11: return against depth for each network family and optimizer.
     *
     * @param algorithm the agent and task
     * @param depths hidden depths
     * @param steps environment steps
     * @param seeds seeds
     * @param width the width
     * @return "family/optimizer @ depth" to outcome
     */
    public static Map<String, DepthOutcome> depth(Algorithm algorithm, int[] depths, long steps, int seeds, int width) {
        Map<String, DepthOutcome> out = new LinkedHashMap<>();
        for (int depth : depths) {
            for (Family f : Family.values()) {
                for (boolean kron : new boolean[] {false, true}) {
                    double[] ret = new double[seeds];
                    double spread = 0;
                    for (int s = 0; s < seeds; s++) {
                        long seed = s + 1;
                        RlNetworkFactory nets = family(f, width, depth);
                        Supplier<Optimizer> opt = () -> kron ? PsgdKron.of(3e-4f) : Optimizer.adam(1e-3f);
                        Model probeNet;
                        switch (algorithm) {
                            case DQN -> {
                                Dqn d = Dqn.of(() -> nets.create(4, 2, seed), 2).optimizer(opt.get())
                                        .learningStarts((int) Math.min(500, steps / 5))
                                        .epsilon(Schedule.linear(1, 0.05, steps / 3)).seed(seed);
                                d.fit(new CartPole(200), steps);
                                ret[s] = Rl.evaluate(d, new CartPole(200), 5, 10_000 + seed);
                                probeNet = d.network();
                            }
                            case PPO -> {
                                Model pi = nets.create(4, 2, seed);
                                Ppo p = Ppo.of(pi, nets.create(4, 1, seed + 1), new CartPole(200).actionSpace())
                                        .optimizer(opt.get()).rolloutSteps(256).seed(seed);
                                p.fit(new CartPole(200), steps);
                                ret[s] = Rl.evaluate(p, new CartPole(200), 5, 10_000 + seed);
                                probeNet = pi;
                            }
                            default -> {
                                ModernSac sac = ModernSac.of(3, 1, nets, nets).optimizer(opt)
                                        .learningStarts((int) Math.min(500, steps / 5)).seed(seed);
                                sac.fit(new Pendulum(200), steps);
                                ret[s] = Rl.evaluate(sac, new Pendulum(200), 3, 10_000 + seed);
                                probeNet = sac.actor();
                            }
                        }
                        int in = probeNet.inputSize();
                        Tensor batch = Tensor.normal(io.github.zzuegg.brainiac.tensor.Shape.of(64, in), 0, 1,
                                new SplittableRandom(seed));
                        spread += MultiSkipNetwork.gradientSpread(probeNet, batch) / seeds;
                    }
                    out.put(f + "/" + (kron ? "Kron" : "Adam") + " @ depth " + depth,
                            new DepthOutcome(LearningCurveHarness.iqm(ret), spread));
                }
            }
        }
        return out;
    }

    /**
     * The value-scaling outcome.
     *
     * @param grid the width x batch grid
     * @param passive {own targets, wide critic's targets, the wide critic}
     * @param bestBatchLaw {a, b} of {@code batch* = a width^b}
     * @param widthLaw the held-out TD error against width at the largest batch
     */
    public record Scaling(List<ValueScalingStudy.Point> grid, ValueScalingStudy.Point[] passive, double[] bestBatchLaw,
            CurveExtrapolator.Fit widthLaw) {
    }

    /** A noisy cart-pole controller: push towards the pole's lean, a third of the time at random. */
    static Agent behaviour(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        return (o, explore) -> new float[] {r.nextDouble() < 0.3 ? r.nextInt(2) : o[2] + 0.5f * o[3] > 0 ? 1 : 0};
    }

    /**
     * W10 4.17: TD critics on a fixed behaviour policy's cart-pole data.
     *
     * @param episodes training episodes (held-out: a quarter as many)
     * @param widths widths
     * @param batches batch sizes
     * @param steps gradient steps per cell
     * @param seed the seed
     * @return the outcome
     */
    public static Scaling scaling(int episodes, int[] widths, int[] batches, int steps, long seed) {
        float gamma = 0.95f;
        ValueScalingStudy.Dataset train = ValueScalingStudy.collect(new CartPole(200), behaviour(seed), episodes, gamma,
                seed);
        ValueScalingStudy.Dataset held = ValueScalingStudy.collect(new CartPole(200), behaviour(seed + 1),
                Math.max(2, episodes / 4), gamma, seed + 1);
        List<ValueScalingStudy.Point> grid = ValueScalingStudy.grid(train, held, 2, widths, batches, steps, 1e-3f, gamma,
                seed);
        ValueScalingStudy.Point[] passive = ValueScalingStudy.passive(train, held, 2, widths[0],
                widths[widths.length - 1], batches[batches.length - 1], steps, 1e-3f, gamma, seed);
        return new Scaling(grid, passive, ValueScalingStudy.fitBestBatch(grid),
                widths.length >= 3 ? ValueScalingStudy.fitWidthLaw(grid, batches[batches.length - 1]) : null);
    }

    /**
     * W10 4.17, online: SAC return over critic width x batch size on the pendulum, with BRO and
     * SimbaV2 at the widest critic as regularised points (Run only; expensive).
     *
     * @param widths critic widths
     * @param batches batch sizes
     * @param steps environment steps
     * @param seeds seeds
     * @return "config" to IQM return
     */
    public static Map<String, Double> onlineScaling(int[] widths, int[] batches, long steps, int seeds) {
        Map<String, Double> out = new LinkedHashMap<>();
        int start = (int) Math.min(500, steps / 5);
        for (int w : widths) {
            for (int b : batches) {
                LearningCurveHarness.Report r = LearningCurveHarness.of(() -> new Pendulum(200)).budgets(steps)
                        .seeds(seeds).add("SAC", s -> ModernSac.of(3, 1, RlNetworkFactory.mlp(Activation.RELU, 32, 32),
                                RlNetworkFactory.mlp(Activation.RELU, w, w)).batchSize(b).learningStarts(start)
                                .learningRate(3e-3f).seed(s)).run();
                out.put("SAC w" + w + " b" + b, r.curve("SAC").finalIqm());
            }
        }
        int wide = widths[widths.length - 1];
        LearningCurveHarness.Report r = LearningCurveHarness.of(() -> new Pendulum(200)).budgets(steps).seeds(seeds)
                .add("BRO", s -> Bro.learner(3, 1, wide, new Bro.Options(true, 0, 8, 1)).learningStarts(start)
                        .learningRate(3e-3f).seed(s))
                .add("SimbaV2", s -> SimbaV2.learner(3, 1, wide, 1).learningStarts(start).learningRate(3e-3f).seed(s))
                .run();
        out.put("BRO w" + wide, r.curve("BRO").finalIqm());
        out.put("SimbaV2 w" + wide, r.curve("SimbaV2").finalIqm());
        return out;
    }

    /**
     * The long runs.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        System.out.println("W10 4.10 cart-pole 10k: " + plasticity(() -> new CartPole(200), 10_000, new int[] {1, 4, 16},
                5));
        System.out.println("W10 4.10 task switch 5k + 5k, RR 4: " + taskSwitch(5_000, 4, 5));
        System.out.println("W10 4.11 DQN depth: " + depth(Algorithm.DQN, new int[] {2, 4, 8, 16}, 10_000, 5, 64));
        System.out.println("W10 4.11 PPO depth: " + depth(Algorithm.PPO, new int[] {2, 4, 8, 16}, 20_000, 5, 64));
        System.out.println("W10 4.11 SAC depth: " + depth(Algorithm.SAC, new int[] {2, 4, 8}, 5_000, 5, 64));
        Scaling sc = scaling(100, new int[] {32, 64, 128, 256, 512}, new int[] {32, 64, 128, 256, 512, 1024}, 3_000, 1);
        System.out.println("W10 4.17 grid: " + sc.grid());
        System.out.println("W10 4.17 passive: " + java.util.Arrays.toString(sc.passive()));
        System.out.printf("W10 4.17 best batch = %.2f width^%.2f; width law %s%n", sc.bestBatchLaw()[0],
                sc.bestBatchLaw()[1], sc.widthLaw());
        System.out.println("W10 4.17 online SAC: " + onlineScaling(new int[] {32, 128, 512}, new int[] {32, 128, 512},
                10_000, 5));
    }
}
