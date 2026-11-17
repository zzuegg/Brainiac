package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.ActionChunking;
import io.github.zzuegg.brainiac.experimental.rl.BehaviorCloning;
import io.github.zzuegg.brainiac.experimental.rl.Cql;
import io.github.zzuegg.brainiac.experimental.rl.FlowActionPolicy;
import io.github.zzuegg.brainiac.experimental.rl.FlowQLearning;
import io.github.zzuegg.brainiac.experimental.rl.Iql;
import io.github.zzuegg.brainiac.experimental.rl.OfflineDataset;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.WindyReach;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Offline RL with continuous actions, the continuous counterpart of
 * {@link OfflineRlComparison}: a fixed data set on {@link WindyReach} (push a cart to a
 * target against a head wind; dense reward) mixing a few noisy controller runs with many
 * random-action runs, and policies learned from it without further interaction. Scores are
 * normalized returns: 0 is the random policy, 1 the noise-free controller.
 *
 * <p>Compared: MSE behaviour cloning, flow behaviour cloning ({@link FlowActionPolicy}),
 * {@link FlowQLearning} with single actions and with action chunks of 4 (Q-chunking), and
 * the discrete-action {@link Iql} and {@link Cql} on the same data with each action snapped
 * to the nearest of five force levels.
 */
@SuppressWarnings("exports")
public final class ContinuousOfflineRl {
    private static final float[] LEVELS = {-1, -0.5f, 0, 0.5f, 1};

    private ContinuousOfflineRl() {
    }

    /** The task: target 1, head wind 0.5, 40 steps. */
    static WindyReach task() {
        return new WindyReach(1, -0.5, 40);
    }

    /** The same task with five force levels as discrete actions. */
    static final class LeveledReach implements Environment {
        private final WindyReach env = task();

        static int nearest(float action) {
            int best = 0;
            for (int k = 1; k < LEVELS.length; k++) {
                if (Math.abs(LEVELS[k] - action) < Math.abs(LEVELS[best] - action)) {
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
            return new Space.Discrete(LEVELS.length);
        }

        @Override
        public float[] reset(SplittableRandom random) {
            return env.reset(random);
        }

        @Override
        public Step step(float[] action) {
            return env.step(new float[] {LEVELS[Math.round(action[0])]});
        }
    }

    /**
     * A PD controller with wind compensation.
     *
     * @param noise Gaussian action noise
     * @param seed the noise seed
     * @return the controller
     */
    public static Agent controller(double noise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        return (o, explore) -> new float[] {(float) Math.max(-1, Math.min(1,
                2 * o[0] - 1.5 * o[1] + 0.25 + noise * r.nextGaussian()))};
    }

    /**
     * The mixed data set.
     *
     * @param good runs of the noisy controller (noise 0.2)
     * @param random runs of uniformly random actions
     * @param seed the seed
     * @return the data
     */
    public static OfflineDataset data(int good, int random, long seed) {
        WindyReach env = task();
        return OfflineDataset.collect(env, controller(0.2, seed), good, seed)
                .merge(OfflineDataset.collect(env, Agent.random(env.actionSpace(), seed + 1), random, seed + 1));
    }

    private static OfflineDataset leveled(OfflineDataset data) {
        List<List<OfflineDataset.Step>> out = new ArrayList<>();
        for (List<OfflineDataset.Step> t : data.trajectories()) {
            List<OfflineDataset.Step> d = new ArrayList<>();
            for (OfflineDataset.Step s : t) {
                d.add(new OfflineDataset.Step(s.observation(), new float[] {LeveledReach.nearest(s.action()[0])},
                        s.reward(), s.nextObservation(), s.terminated()));
            }
            out.add(d);
        }
        return OfflineDataset.of(out);
    }

    /**
     * Runs the comparison.
     *
     * @param good noisy-controller runs in the data
     * @param random random runs in the data
     * @param steps gradient steps per method
     * @param seed the seed
     * @return normalized return per method, plus the data's own
     */
    public static Map<String, Double> run(int good, int random, int steps, long seed) {
        OfflineDataset data = data(good, random, seed);
        WindyReach env = task();
        int episodes = 5;   // the task is deterministic from its single start
        double floor = Rl.evaluate(Agent.random(env.actionSpace(), seed), env, episodes, seed);
        double top = Rl.evaluate(controller(0, seed), env, episodes, seed);
        Map<String, Double> raw = new LinkedHashMap<>();
        raw.put("data", data.meanReturn());
        raw.put("BC", Rl.evaluate(BehaviorCloning.continuous(Nets.mlp(2, 1, Activation.RELU, seed, 64, 64))
                .seed(seed).train(data, steps), env, episodes, seed));
        FlowActionPolicy flowBc = FlowActionPolicy.of(2, 1, 1, 64, 2, seed).execute(1);
        flowBc.train(ActionChunking.pairs(data, 1), steps, 64);
        raw.put("flow BC", Rl.evaluate(flowBc, env, episodes, seed));
        raw.put("FQL", Rl.evaluate(FlowQLearning.of(2, 1, 1, 64, seed).alpha(0.3f).tau(0.05f).discount(0.95f)
                .train(data, steps), env, episodes, seed));
        raw.put("FQL + Q-chunking (h=4)", Rl.evaluate(FlowQLearning.of(2, 1, 4, 64, seed).alpha(0.3f).tau(0.05f)
                .discount(0.95f).train(data, steps), env, episodes, seed));
        OfflineDataset discrete = leveled(data);
        LeveledReach leveledEnv = new LeveledReach();
        raw.put("IQL (5 levels)", Rl.evaluate(Iql.of(2, LEVELS.length, RlNetworkFactory.mlp(Activation.RELU, 64, 64))
                .discount(0.95f).seed(seed).train(discrete, steps), leveledEnv, episodes, seed));
        raw.put("CQL (5 levels)", Rl.evaluate(Cql.of(() -> Nets.mlp(2, LEVELS.length, Activation.RELU, seed, 64, 64),
                LEVELS.length).alpha(0.5f).discount(0.95f).seed(seed).train(discrete, steps), leveledEnv, episodes, seed));
        Map<String, Double> out = new LinkedHashMap<>();
        raw.forEach((k, v) -> out.put(k, (v - floor) / (top - floor)));
        return out;
    }
}
