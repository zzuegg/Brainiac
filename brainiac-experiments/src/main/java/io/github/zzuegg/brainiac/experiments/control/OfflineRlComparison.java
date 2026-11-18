package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.BehaviorCloning;
import io.github.zzuegg.brainiac.experimental.rl.Cql;
import io.github.zzuegg.brainiac.experimental.rl.DecisionTransformer;
import io.github.zzuegg.brainiac.experimental.rl.FlowQLearning;
import io.github.zzuegg.brainiac.experimental.rl.Iql;
import io.github.zzuegg.brainiac.experimental.rl.OfflineDataset;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.Td3Bc;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.GridWorld;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Pendulum;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Offline RL from logged data of mixed quality: mostly random play with a fraction of
 * noisy-expert episodes, never interacting with the environment during training. Behaviour
 * cloning copies the average logged behaviour; filtered behaviour cloning copies only the
 * best episodes; CQL and IQL learn values that stay within the data and improve on it;
 * the Decision Transformer conditions on a high target return. Varying the expert
 * fraction shows where each method breaks.
 *
 * <p>Papers: Kumar et al., <a href="https://arxiv.org/abs/2006.04779">arXiv:2006.04779</a>
 * (CQL); Kostrikov et al., <a href="https://arxiv.org/abs/2110.06169">arXiv:2110.06169</a> (IQL);
 * Chen et al., <a href="https://arxiv.org/abs/2106.01345">arXiv:2106.01345</a> (Decision Transformer).
 *
 * <p>{@link #pendulum} adds the continuous-control counterpart in the style of the D4RL suites:
 * Pendulum data sets of four qualities (random, medium, medium-replay, expert) and
 * {@link Td3Bc} (Fujimoto and Gu, <a href="https://arxiv.org/abs/2106.06860">arXiv:2106.06860</a>)
 * against behaviour cloning, flow Q-learning and IQL/CQL on five torque levels. The Decision
 * Transformer here is discrete-only and is left out of that table.
 */
@SuppressWarnings("exports")
public final class OfflineRlComparison {

    private OfflineRlComparison() {
    }

    /** @return the task: a 3 x 5 open grid, goal in the far corner, 20-step episodes */
    public static GridWorld grid() {
        return GridWorld.of("S....", ".....", "....G").maxSteps(20);
    }

    /**
     * The behaviour policy's expert: right along the top row, then down.
     *
     * @param noise the chance of a random action
     * @param seed the seed
     * @return the agent
     */
    public static Agent expert(double noise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        return (obs, explore) -> {
            if (r.nextDouble() < noise) {
                return new float[] {r.nextInt(4)};
            }
            int cell = 0;
            while (obs[cell] == 0) {
                cell++;
            }
            return new float[] {cell % 5 < 4 ? 1 : 2};
        };
    }

    /**
     * Trains every method on one dataset.
     *
     * @param randomEpisodes logged random episodes
     * @param expertEpisodes logged noisy-expert episodes
     * @param steps gradient steps per method
     * @param seed the seed
     * @return method to greedy return; "data" is the mean logged return
     */
    public static Map<String, Double> run(int randomEpisodes, int expertEpisodes, int steps, long seed) {
        GridWorld env = grid();
        int obs = 15;
        OfflineDataset data = OfflineDataset.collect(env, Agent.random(env.actionSpace(), seed), randomEpisodes, seed)
                .merge(OfflineDataset.collect(env, expert(0.5, seed), expertEpisodes, seed + 1));
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("data", data.meanReturn());
        out.put("BC", Rl.evaluate(BehaviorCloning.discrete(Nets.mlp(obs, 4, Activation.RELU, seed, 32), 4)
                .seed(seed).train(data, steps / 4), env, 1, seed));
        out.put("filtered BC (top 10%)", Rl.evaluate(BehaviorCloning.discrete(
                Nets.mlp(obs, 4, Activation.RELU, seed, 32), 4).seed(seed).train(data.best(0.1), steps / 4), env, 1,
                seed));
        out.put("CQL", Rl.evaluate(Cql.of(() -> Nets.mlp(obs, 4, Activation.RELU, seed, 32), 4).alpha(0.5f)
                .discount(0.9f).seed(seed).train(data, steps), env, 1, seed));
        out.put("IQL", Rl.evaluate(Iql.of(obs, 4, RlNetworkFactory.mlp(Activation.RELU, 32)).seed(seed)
                .train(data, steps), env, 1, seed));
        DecisionTransformer dt = DecisionTransformer.of(obs, 4, 4, 20, 32, 1, seed).seed(seed)
                .train(data, steps / 4);
        out.put("Decision Transformer", dt.play(env, (float) data.maxReturn(), seed));
        return out;
    }

    /** The quality of a logged Pendulum data set, after the D4RL naming. */
    public enum PendulumData {
        /** uniformly random torques */
        RANDOM,
        /** the controller with 80% of its actions replaced by random ones */
        MEDIUM,
        /** a "training run" towards medium: episodes from random to medium quality */
        MEDIUM_REPLAY,
        /** the controller with small Gaussian noise */
        EXPERT
    }

    /**
     * One method's result.
     *
     * @param normalised the return, 0 for the random policy and 1 for the noise-free controller
     * @param seconds the training wall-clock
     */
    public record OfflineRow(double normalised, double seconds) {
    }

    private static final float[] TORQUES = {-1, -0.5f, 0, 0.5f, 1};

    /** Pendulum with actions in [-1, 1], scaled to the torque limit 2. */
    static final class UnitPendulum implements Environment {
        private final Pendulum env = new Pendulum(200);

        @Override
        public Space observationSpace() {
            return env.observationSpace();
        }

        @Override
        public Space actionSpace() {
            return Space.box(1, -1, 1);
        }

        @Override
        public float[] reset(SplittableRandom random) {
            return env.reset(random);
        }

        @Override
        public Step step(float[] action) {
            return env.step(new float[] {2 * Math.max(-1, Math.min(1, action[0]))});
        }
    }

    /** The same task with five torque levels as discrete actions. */
    static final class LeveledPendulum implements Environment {
        private final UnitPendulum env = new UnitPendulum();

        static int nearest(float action) {
            int best = 0;
            for (int k = 1; k < TORQUES.length; k++) {
                if (Math.abs(TORQUES[k] - action) < Math.abs(TORQUES[best] - action)) {
                    best = k;
                }
            }
            return best;
        }

        @Override
        public Space observationSpace() {
            return env.observationSpace();
        }

        @Override
        public Space actionSpace() {
            return new Space.Discrete(TORQUES.length);
        }

        @Override
        public float[] reset(SplittableRandom random) {
            return env.reset(random);
        }

        @Override
        public Step step(float[] action) {
            return env.step(new float[] {TORQUES[Math.round(action[0])]});
        }
    }

    /**
     * A hand-made swing-up controller for Pendulum with actions in [-1, 1]: energy pumping far
     * from the top, a PD controller near it, with a share of random actions and Gaussian noise.
     *
     * @param randomShare the chance of a uniformly random action
     * @param noise Gaussian noise on the other actions
     * @param seed the seed
     * @return the controller
     */
    public static Agent pendulumController(double randomShare, double noise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        return (o, explore) -> {
            if (r.nextDouble() < randomShare) {
                return new float[] {(float) r.nextDouble(-1, 1)};
            }
            double angle = Math.atan2(o[1], o[0]);
            double speed = o[2];
            double u;
            if (o[0] > 0.85) {
                u = -(10 * angle + 2 * speed) / 2;
            } else {
                double energy = speed * speed / 2 + 15 * o[0];
                u = (speed >= 0 ? 1 : -1) * 0.5 * (15 - energy);
            }
            return new float[] {(float) Math.max(-1, Math.min(1, u + noise * r.nextGaussian()))};
        };
    }

    /**
     * Collects a Pendulum data set.
     *
     * @param kind the quality
     * @param episodes 200-step episodes
     * @param seed the seed
     * @return the data, actions in [-1, 1]
     */
    public static OfflineDataset pendulumData(PendulumData kind, int episodes, long seed) {
        UnitPendulum env = new UnitPendulum();
        return switch (kind) {
            case RANDOM -> OfflineDataset.collect(env, Agent.random(env.actionSpace(), seed), episodes, seed);
            case MEDIUM -> OfflineDataset.collect(env, pendulumController(0.8, 0.3, seed), episodes, seed);
            case EXPERT -> OfflineDataset.collect(env, pendulumController(0, 0.1, seed), episodes, seed);
            case MEDIUM_REPLAY -> {
                OfflineDataset d = null;
                for (int e = 0; e < episodes; e++) {
                    double share = 1 - 0.2 * e / Math.max(1, episodes - 1);
                    OfflineDataset one = OfflineDataset.collect(env, pendulumController(share, 0.3, seed + e), 1,
                            seed + e);
                    d = d == null ? one : d.merge(one);
                }
                yield d;
            }
        };
    }

    private static OfflineDataset leveledPendulum(OfflineDataset data) {
        List<List<OfflineDataset.Step>> out = new ArrayList<>();
        for (List<OfflineDataset.Step> t : data.trajectories()) {
            List<OfflineDataset.Step> d = new ArrayList<>();
            for (OfflineDataset.Step s : t) {
                d.add(new OfflineDataset.Step(s.observation(), new float[] {LeveledPendulum.nearest(s.action()[0])},
                        s.reward(), s.nextObservation(), s.terminated()));
            }
            out.add(d);
        }
        return OfflineDataset.of(out);
    }

    private static OfflineRow timed(Supplier<Agent> train, Environment env, int episodes, long seed, double floor,
            double top) {
        long t = System.nanoTime();
        Agent a = train.get();
        double seconds = (System.nanoTime() - t) / 1e9;
        return new OfflineRow((Rl.evaluate(a, env, episodes, seed) - floor) / (top - floor), seconds);
    }

    /**
     * TD3+BC against the other offline methods on one Pendulum data set (10.4.1).
     *
     * @param kind the data quality
     * @param episodes logged 200-step episodes
     * @param steps gradient steps per method
     * @param seed the seed
     * @return method to normalised return and training time; "data" holds the data's own return
     */
    public static Map<String, OfflineRow> pendulum(PendulumData kind, int episodes, int steps, long seed) {
        OfflineDataset data = pendulumData(kind, episodes, seed);
        UnitPendulum env = new UnitPendulum();
        int eval = 5;
        double floor = Rl.evaluate(Agent.random(env.actionSpace(), seed), env, eval, seed);
        double top = Rl.evaluate(pendulumController(0, 0, seed), env, eval, seed);
        Map<String, OfflineRow> out = new LinkedHashMap<>();
        out.put("data", new OfflineRow((data.meanReturn() - floor) / (top - floor), 0));
        float[] low = {-1};
        float[] high = {1};
        RlNetworkFactory nets = RlNetworkFactory.mlp(Activation.RELU, 64, 64);
        out.put("BC", timed(() -> BehaviorCloning.continuous(Nets.mlp(3, 1, Activation.RELU, seed, 64, 64))
                .seed(seed).train(data, steps), env, eval, seed, floor, top));
        out.put("TD3+BC", timed(() -> Td3Bc.of(3, low, high, nets, 1e-3f).seed(seed).train(data, steps), env, eval,
                seed, floor, top));
        out.put("TD3+BC alpha 0 (BC)", timed(() -> Td3Bc.of(3, low, high, nets, 1e-3f).alpha(0).seed(seed)
                .train(data, steps), env, eval, seed, floor, top));
        out.put("FQL", timed(() -> FlowQLearning.of(3, 1, 1, 64, seed).alpha(0.3f).train(data, steps), env, eval,
                seed, floor, top));
        OfflineDataset discrete = leveledPendulum(data);
        LeveledPendulum leveled = new LeveledPendulum();
        out.put("IQL (5 levels)", timed(() -> Iql.of(3, TORQUES.length, nets).seed(seed).train(discrete, steps),
                leveled, eval, seed, floor, top));
        out.put("CQL (5 levels)", timed(() -> Cql.of(() -> Nets.mlp(3, TORQUES.length, Activation.RELU, seed, 64,
                64), TORQUES.length).alpha(0.5f).seed(seed).train(discrete, steps), leveled, eval, seed, floor, top));
        return out;
    }

    /**
     * Runs the Pendulum table on all four data sets.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        for (PendulumData kind : PendulumData.values()) {
            System.out.printf("%s: %s%n", kind, pendulum(kind, 50, 20_000, 1));
        }
    }
}
