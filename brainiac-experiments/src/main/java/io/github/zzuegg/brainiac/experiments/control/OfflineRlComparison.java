package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.BehaviorCloning;
import io.github.zzuegg.brainiac.experimental.rl.Cql;
import io.github.zzuegg.brainiac.experimental.rl.DecisionTransformer;
import io.github.zzuegg.brainiac.experimental.rl.Iql;
import io.github.zzuegg.brainiac.experimental.rl.OfflineDataset;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.GridWorld;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Rl;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

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
 */
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
}
