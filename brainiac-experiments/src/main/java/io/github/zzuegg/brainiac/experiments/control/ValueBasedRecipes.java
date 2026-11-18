package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.deeprl.DrQ;
import io.github.zzuegg.brainiac.experimental.deeprl.EfficientZero;
import io.github.zzuegg.brainiac.experimental.deeprl.LearningCurveHarness;
import io.github.zzuegg.brainiac.experimental.deeprl.ModernSac;
import io.github.zzuegg.brainiac.experimental.deeprl.MrQ;
import io.github.zzuegg.brainiac.experimental.deeprl.PixelObservations;
import io.github.zzuegg.brainiac.experimental.deeprl.Pqn;
import io.github.zzuegg.brainiac.experimental.deeprl.SelfPredictive;
import io.github.zzuegg.brainiac.experimental.rl.DreamerAgent;
import io.github.zzuegg.brainiac.experimental.rl.KeyDoorGrid;
import io.github.zzuegg.brainiac.experimental.rl.MuZeroLite;
import io.github.zzuegg.brainiac.experimental.rl.RainbowDqn;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.CartPole;
import io.github.zzuegg.brainiac.rl.Dqn;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.GridWorld;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Pendulum;
import io.github.zzuegg.brainiac.rl.Ppo;
import io.github.zzuegg.brainiac.rl.Schedule;
import io.github.zzuegg.brainiac.rl.Space;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.function.Supplier;

/**
 * Value-based recipes for discrete actions, on one learning-curve harness
 * ({@link LearningCurveHarness}):
 *
 * <ul>
 *   <li>{@link #pqn}: PQN (no replay, no target network, LayerNorm, Q(lambda) over vectorised
 *       rollouts) against core DQN, Rainbow and core PPO at equal environment steps, with
 *       LayerNorm and lambda ablated (W10 4.5).</li>
 *   <li>{@link #sampleEfficient}: "Atari 100k in miniature" on pixel grids: SPR, BBF, DrQ and
 *       SPR+DrQ against a plain double DQN, core DQN, Rainbow and Dreamer, and BBF without
 *       resets (W10 4.8, 4.9); {@link #pixelPendulum} is the continuous DrQ row (W10 4.9).</li>
 *   <li>{@link #modelBased}: EfficientZero and each addition removed, MuZero-lite, SPR and
 *       Dreamer (W10 4.12).</li>
 *   <li>{@link #mrqDiscrete}: MR.Q's discrete rows with the continuous configuration (W10 4.6).</li>
 * </ul>
 */
public final class ValueBasedRecipes {

    private ValueBasedRecipes() {
    }

    /** The discrete tasks. */
    public enum Task {
        /** Cart-pole, 200-step episodes (return = steps balanced). */
        CARTPOLE(() -> new CartPole(200)),
        /** A 3 x 5 grid with a wall, goal reward 1, 25 steps, one-hot cells. */
        GRID(ValueBasedRecipes::grid),
        /** The same grid rendered as a 3 x 6 x 10 image. */
        PIXEL_GRID(() -> PixelObservations.grid(grid(), 2)),
        /** A 4 x 4 key-and-door grid (key always visible) rendered as a 3 x 8 x 8 image. */
        PIXEL_KEY_DOOR(() -> PixelObservations.keyDoor(new KeyDoorGrid(4, 30, 30), 4, 2));

        private final Supplier<Environment> factory;

        Task(Supplier<Environment> factory) {
            this.factory = factory;
        }

        /** @return a fresh environment */
        public Environment create() {
            return factory.get();
        }

        /** @return values per observation */
        public int observations() {
            return create().observationSpace().size();
        }

        /** @return discrete actions */
        public int actions() {
            return ((Space.Discrete) create().actionSpace()).n();
        }

        /** @return the image geometry, or null for vector observations */
        PixelObservations pixels() {
            return create() instanceof PixelObservations p ? p : null;
        }
    }

    /** @return the grid: start top left, goal bottom right, a wall in the middle row */
    public static GridWorld grid() {
        return GridWorld.of("S....", ".###.", "....G").maxSteps(25);
    }

    private static int warmup(long steps) {
        return (int) Math.min(500, Math.max(100, steps / 5));
    }

    private static LearningCurveHarness harness(Task task, long steps, int seeds) {
        return LearningCurveHarness.of(task::create).budgets(steps).seeds(seeds).evalEpisodes(5);
    }

    /** An MLP encoder for vectors and pixels alike (convolutions are too slow for the CPU budget). */
    private static RlNetworkFactory encoder(Task task) {
        return SelfPredictive.mlpEncoder(64);
    }

    private static Dqn coreDqn(Task task, long steps, long seed) {
        int obs = task.observations();
        return Dqn.of(() -> Nets.mlp(obs, task.actions(), Activation.RELU, seed, 64, 64), task.actions())
                .learningStarts(warmup(steps)).epsilon(Schedule.linear(1, 0.05, steps / 3)).seed(seed);
    }

    private static RainbowDqn rainbow(Task task, long steps, long seed) {
        int obs = task.observations();
        return RainbowDqn.of(() -> Nets.mlp(obs, task.actions(), Activation.RELU, seed, 64, 64), task.actions())
                .learningStarts(warmup(steps)).epsilon(Schedule.linear(1, 0.05, steps / 3)).seed(seed);
    }

    private static SelfPredictive base(Task task, long steps, long seed) {
        return SelfPredictive.of(task.observations(), task.actions(), encoder(task), 32, 64)
                .learningStarts(warmup(steps)).epsilon((int) (steps / 3), 0.05f).seed(seed);
    }

    /**
     * W10 4.5: PQN against DQN, Rainbow and PPO.
     *
     * @param task the task
     * @param steps environment steps (summed over PQN's parallel environments)
     * @param seeds seeds
     * @param envs PQN's parallel environments
     * @return the curves
     */
    public static LearningCurveHarness.Report pqn(Task task, long steps, int seeds, int envs) {
        int obs = task.observations();
        return harness(task, steps, seeds)
                .add("PQN (LayerNorm, lambda 0.65)", s -> Pqn.of(obs, task.actions(), 64).environments(task::create, envs)
                        .seed(s), Pqn::updates)
                .add("PQN no LayerNorm", s -> Pqn.of(obs, task.actions(), 64).layerNorm(false)
                        .environments(task::create, envs).seed(s), Pqn::updates)
                .add("PQN lambda 0", s -> Pqn.of(obs, task.actions(), 64).lambda(0).environments(task::create, envs)
                        .seed(s), Pqn::updates)
                .add("DQN (core, replay + target)", s -> coreDqn(task, steps, s), Dqn::updates)
                .add("Rainbow", s -> rainbow(task, steps, s), RainbowDqn::updates)
                .add("PPO (core)", s -> Ppo.of(Nets.mlp(obs, task.actions(), s, 64, 64), Nets.mlp(obs, 1, s + 1, 64, 64),
                        task.create().actionSpace()).rolloutSteps(256).seed(s))
                .run();
    }

    /**
     * W10 4.8 and 4.9: the sample-efficiency recipes on a pixel task.
     *
     * @param task the task (a pixel grid)
     * @param steps environment steps
     * @param seeds seeds
     * @param replayRatio BBF's replay ratio
     * @return the curves
     */
    public static LearningCurveHarness.Report sampleEfficient(Task task, long steps, int seeds, int replayRatio) {
        PixelObservations p = task.pixels();
        int resetEvery = (int) Math.max(200, steps * replayRatio / 4);
        int pad = Math.max(1, p == null ? 1 : p.height() / 8);
        LearningCurveHarness h = harness(task, steps, seeds)
                .add("DQN (double, n-step)", s -> base(task, steps, s), SelfPredictive::updates)
                .add("DQN (core)", s -> coreDqn(task, steps, s), Dqn::updates)
                .add("Rainbow", s -> rainbow(task, steps, s), RainbowDqn::updates);
        if (p != null) {
            h.add("DrQ (K=M=2)", s -> base(task, steps, s).augment(DrQ.randomShift(p.channels(), p.height(), p.width(),
                    pad, s), 2, 2), SelfPredictive::updates);
        }
        h.add("SPR (k=3)", s -> base(task, steps, s).spr(3, 1f), SelfPredictive::updates);
        if (p != null) {
            h.add("SPR + DrQ", s -> base(task, steps, s).spr(3, 1f).augment(DrQ.randomShift(p.channels(), p.height(),
                    p.width(), pad, s), 1, 1), SelfPredictive::updates);
        }
        h.add("BBF (SPR, RR " + replayRatio + ", resets)", s -> base(task, steps, s).spr(3, 1f).replayRatio(replayRatio)
                .resets(resetEvery, 0.5f).bbfSchedules(resetEvery / 2).weightDecay(0.1f).targetTau(0.005f),
                SelfPredictive::updates);
        h.add("BBF without resets", s -> base(task, steps, s).spr(3, 1f).replayRatio(replayRatio)
                .bbfSchedules((int) (steps * replayRatio / 2)).weightDecay(0.1f).targetTau(0.005f),
                SelfPredictive::updates);
        h.add("DQN at RR " + replayRatio + " (no resets)", s -> base(task, steps, s).replayRatio(replayRatio),
                SelfPredictive::updates);
        h.add("Dreamer", s -> DreamerAgent.of(task.observations(), task.actions(), 64, s));
        return h.run();
    }

    /**
     * W10 4.9, continuous row: SAC on a pixel pendulum with and without DrQ's random shifts.
     *
     * @param steps environment steps
     * @param seeds seeds
     * @return the curves
     */
    public static LearningCurveHarness.Report pixelPendulum(long steps, int seeds) {
        Supplier<PixelObservations> env = () -> PixelObservations.pendulum(new Pendulum(200), 12);
        PixelObservations p = env.get();
        int obs = p.channels() * p.height() * p.width();
        RlNetworkFactory actor = RlNetworkFactory.mlp(Activation.RELU, 64, 64);
        return LearningCurveHarness.of(env).budgets(steps).seeds(seeds).evalEpisodes(3)
                .add("SAC (pixels)", s -> ModernSac.of(obs, 1, actor, actor).nStep(3).learningStarts(warmup(steps))
                        .learningRate(3e-3f).seed(s), ModernSac::updates)
                .add("DrQ-v2-style SAC (shifts, 3-step)", s -> DrQ.sac(p, 1, actor, actor, s).learningStarts(warmup(steps))
                        .learningRate(3e-3f), ModernSac::updates)
                .run();
    }

    /**
     * W10 4.12: EfficientZero, each addition removed, MuZero-lite, SPR and Dreamer.
     *
     * @param task the task
     * @param steps environment steps
     * @param seeds seeds
     * @param simulations MCTS simulations per move
     * @return the curves
     */
    public static LearningCurveHarness.Report modelBased(Task task, long steps, int seeds, int simulations) {
        int obs = task.observations();
        int a = task.actions();
        RlNetworkFactory nets = RlNetworkFactory.mlp(Activation.RELU, 64);
        int warm = warmup(steps);
        LearningCurveHarness h = harness(task, steps, seeds);
        h.add("EfficientZero", s -> EfficientZero.of(obs, a, nets, 16, EfficientZero.Components.all(), s)
                .simulations(simulations).warmup(warm), EfficientZero::updates);
        h.add("EZ - consistency", s -> EfficientZero.of(obs, a, nets, 16,
                new EfficientZero.Components(false, true, true), s).simulations(simulations).warmup(warm),
                EfficientZero::updates);
        h.add("EZ - value prefix", s -> EfficientZero.of(obs, a, nets, 16,
                new EfficientZero.Components(true, false, true), s).simulations(simulations).warmup(warm),
                EfficientZero::updates);
        h.add("EZ - off-policy correction", s -> EfficientZero.of(obs, a, nets, 16,
                new EfficientZero.Components(true, true, false), s).simulations(simulations).warmup(warm),
                EfficientZero::updates);
        h.add("EZ none (MuZero-lite structure)", s -> EfficientZero.of(obs, a, nets, 16,
                EfficientZero.Components.none(), s).simulations(simulations).warmup(warm), EfficientZero::updates);
        h.add("MuZeroLite (wave 1)", s -> MuZeroLite.of(obs, a, nets, 16).simulations(simulations).warmup(warm)
                .optimizer(Optimizer.adam(3e-3f)).seed(s));
        h.add("SPR", s -> base(task, steps, s).spr(3, 1f), SelfPredictive::updates);
        h.add("Dreamer", s -> DreamerAgent.of(obs, a, 64, s));
        return h.run();
    }

    /**
     * W10 4.6, discrete rows: MR.Q with the same configuration as on the continuous tasks,
     * against core DQN and MR.Q without its model losses.
     *
     * @param task the task
     * @param steps environment steps
     * @param seeds seeds
     * @return the curves
     */
    public static LearningCurveHarness.Report mrqDiscrete(Task task, long steps, int seeds) {
        int obs = task.observations();
        int start = warmup(steps);
        return harness(task, steps, seeds)
                .add("MR.Q", s -> MrQ.discrete(obs, task.actions(), 32, 16).learningStarts(start).learningRate(3e-3f)
                        .targetEvery(100).seed(s), MrQ::updates)
                .add("MR.Q - model losses", s -> MrQ.discrete(obs, task.actions(), 32, 16).modelLosses(false)
                        .learningStarts(start).learningRate(3e-3f).targetEvery(100).seed(s), MrQ::updates)
                .add("DQN (core)", s -> coreDqn(task, steps, s), Dqn::updates)
                .run();
    }

    /**
     * The long runs (5 seeds).
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        System.out.println("W10 4.5 PQN, cart-pole 20k:\n" + pqn(Task.CARTPOLE, 20_000, 5, 8).table());
        System.out.println("W10 4.5 PQN, grid 10k:\n" + pqn(Task.GRID, 10_000, 5, 8).table());
        System.out.println("W10 4.5 PQN, pixel grid 10k:\n" + pqn(Task.PIXEL_GRID, 10_000, 5, 8).table());
        System.out.println("W10 4.8/4.9 pixel grid 10k:\n" + sampleEfficient(Task.PIXEL_GRID, 10_000, 5, 4).table());
        System.out.println("W10 4.8/4.9 pixel key-door 10k:\n"
                + sampleEfficient(Task.PIXEL_KEY_DOOR, 10_000, 5, 4).table());
        System.out.println("W10 4.9 pixel pendulum 10k:\n" + pixelPendulum(10_000, 5).table());
        System.out.println("W10 4.12 pixel grid 5k:\n" + modelBased(Task.PIXEL_GRID, 5_000, 5, 8).table());
        System.out.println("W10 4.12 cart-pole 5k:\n" + modelBased(Task.CARTPOLE, 5_000, 5, 8).table());
        System.out.println("W10 4.6 MR.Q cart-pole 10k:\n" + mrqDiscrete(Task.CARTPOLE, 10_000, 5).table());
        System.out.println("W10 4.6 MR.Q pixel grid 10k:\n" + mrqDiscrete(Task.PIXEL_GRID, 10_000, 5).table());
    }
}
