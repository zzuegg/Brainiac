package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.CemPlanner;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.experimental.rl.Sac;
import io.github.zzuegg.brainiac.experimental.rl.Td3;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Pendulum;
import io.github.zzuegg.brainiac.rl.Ppo;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Continuous control on the pendulum swing-up with the same environment-step budget:
 * the core on-policy PPO against the off-policy actor-critics SAC, TD3 and DDPG, and
 * model-based CEM planning on a dynamics model learned from the same number of random
 * steps. Off-policy methods reuse every transition many times and win at small budgets;
 * planning with a learned model is the most sample-efficient of all, at the price of
 * compute per action.
 *
 * <p>Papers: Haarnoja et al., <a href="https://arxiv.org/abs/1801.01290">arXiv:1801.01290</a>
 * (SAC); Fujimoto et al., <a href="https://arxiv.org/abs/1802.09477">arXiv:1802.09477</a> (TD3);
 * Lillicrap et al., <a href="https://arxiv.org/abs/1509.02971">arXiv:1509.02971</a> (DDPG); Chua
 * et al., <a href="https://arxiv.org/abs/1805.12114">arXiv:1805.12114</a> (PETS).
 */
public final class ContinuousControlComparison {

    private ContinuousControlComparison() {
    }

    /**
     * Trains every method and evaluates it greedily.
     *
     * @param steps environment steps per method
     * @param episodeLength pendulum episode length
     * @param seed the seed
     * @return method name to mean evaluation return (higher is better; "random" first)
     */
    public static Map<String, Double> run(int steps, int episodeLength, long seed) {
        Pendulum env = new Pendulum(episodeLength);
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("random", Rl.evaluate(Agent.random(env.actionSpace(), seed), env, 3, seed));
        RlNetworkFactory nets = RlNetworkFactory.mlp(Activation.RELU, 32, 32);
        int warmup = Math.min(300, steps / 4);
        Map<String, Learner> learners = new LinkedHashMap<>();
        learners.put("PPO (core)", Ppo.of(Nets.mlp(3, 1, seed, 32, 32), Nets.mlp(3, 1, seed + 1, 32, 32),
                env.actionSpace()).rolloutSteps(200).seed(seed));
        learners.put("SAC", Sac.of(3, 1, nets).learningStarts(warmup).learningRate(3e-3f).seed(seed));
        learners.put("TD3", Td3.td3(3, 1, nets).learningStarts(warmup).learningRate(3e-3f).seed(seed));
        learners.put("DDPG", Td3.ddpg(3, 1, nets).learningStarts(warmup).learningRate(3e-3f).seed(seed));
        for (Map.Entry<String, Learner> e : learners.entrySet()) {
            e.getValue().fit(env, steps);
            out.put(e.getKey(), Rl.evaluate(e.getValue(), env, 3, seed));
        }
        CemPlanner.Dynamics model = CemPlanner.learn(env, Nets.mlp(4, 3, Activation.RELU, seed, 64, 64), steps,
                Math.max(200, steps / 2), seed);
        CemPlanner mpc = CemPlanner.of(model, ContinuousControlComparison::pendulumReward,
                (Space.Box) env.actionSpace()).horizon(25).search(100, 10, 4).seed(seed);
        out.put("CEM-MPC (learned model)", Rl.evaluate(mpc, env, 3, seed));
        return out;
    }

    /**
     * The pendulum's reward, recomputed from an observation and action.
     *
     * @param o the observation {@code (cos, sin, speed)}
     * @param a the torque
     * @return the reward
     */
    public static double pendulumReward(float[] o, float[] a) {
        double angle = Math.atan2(o[1], o[0]);
        double u = Math.max(-2, Math.min(2, a[0]));
        return -(angle * angle + 0.1 * o[2] * o[2] + 0.001 * u * u);
    }
}
