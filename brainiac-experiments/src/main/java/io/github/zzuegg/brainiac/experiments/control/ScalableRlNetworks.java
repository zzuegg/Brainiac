package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.HlGaussValue;
import io.github.zzuegg.brainiac.experimental.rl.ResidualRlNetworks;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.Sac;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Pendulum;
import io.github.zzuegg.brainiac.rl.Rl;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Scaling RL networks, on the {@link ContinuousControlComparison} pendulum: SAC with plain
 * MLP actor and critics and MSE critic loss, against SAC with SimBa networks
 * ({@link ResidualRlNetworks#simba}) and HL-Gauss value classification
 * ({@link HlGaussValue}), and the two ablations (one change each), at the same number of
 * environment steps. Each entry also reports the critic's parameter count.
 */
@SuppressWarnings("exports")
public final class ScalableRlNetworks {
    private ScalableRlNetworks() {
    }

    /** A result: the evaluation return and the parameters of one critic. */
    public record Outcome(double meanReturn, long criticParameters) {
    }

    /**
     * Trains and evaluates each configuration.
     *
     * @param steps environment steps
     * @param episodeLength steps per pendulum episode
     * @param width the hidden width of both network kinds
     * @param ablations whether to also run the two single-change ablations
     * @param seed the seed
     * @return per configuration, the outcome, plus a random policy's return
     */
    public static Map<String, Outcome> run(int steps, int episodeLength, int width, boolean ablations, long seed) {
        Pendulum env = new Pendulum(episodeLength);
        RlNetworkFactory mlp = RlNetworkFactory.mlp(Activation.RELU, width, width);
        RlNetworkFactory simba = ResidualRlNetworks.simba(width, 1);
        float gamma = 0.95f;
        HlGaussValue bins = HlGaussValue.of(-16 / (1 - gamma), 1, 51);
        Map<String, Sac> learners = new LinkedHashMap<>();
        learners.put("MLP + MSE", Sac.of(3, 1, mlp));
        if (ablations) {
            learners.put("MLP + HL-Gauss", Sac.of(3, 1, mlp, bins));
            learners.put("SimBa + MSE", Sac.of(3, 1, simba));
        }
        learners.put("SimBa + HL-Gauss", Sac.of(3, 1, simba, bins));
        Map<String, Outcome> out = new LinkedHashMap<>();
        out.put("random", new Outcome(Rl.evaluate(Agent.random(env.actionSpace(), seed), env, 5, seed), 0));
        for (Map.Entry<String, Sac> e : learners.entrySet()) {
            Sac sac = e.getValue().discount(gamma).learningStarts(Math.min(500, steps / 4)).learningRate(3e-3f)
                    .seed(seed);
            sac.fit(env, steps);
            boolean classify = e.getKey().endsWith("HL-Gauss");
            long params = (e.getKey().startsWith("MLP") ? mlp : simba).create(4, classify ? bins.bins() : 1, seed)
                    .parameterCount();
            out.put(e.getKey(), new Outcome(Rl.evaluate(sac, env, 5, seed), params));
        }
        return out;
    }
}
