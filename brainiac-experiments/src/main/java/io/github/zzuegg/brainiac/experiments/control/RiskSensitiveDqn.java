package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.ImplicitQuantileDqn;
import io.github.zzuegg.brainiac.experimental.rl.RainbowDqn;
import io.github.zzuegg.brainiac.experimental.rl.RiskyBridge;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.GridWorld;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Schedule;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * W6 3.5: implicit quantile networks.
 * <ul>
 *   <li>{@link #maze}: IQN against C51 and QR-DQN ({@link RainbowDqn}) on the
 *       {@link RainbowAblation#maze() RainbowAblation maze} at equal steps: greedy return.</li>
 *   <li>{@link #risk}: risk-neutral IQN against the same learner acting on CVaR at
 *       {@code alpha = 0.25} on {@link RiskyBridge}: how often the greedy agent falls off the
 *       bridge, and its mean return.</li>
 * </ul>
 *
 * <p>Paper: Dabney et al., <a href="https://arxiv.org/abs/1806.06923">arXiv:1806.06923</a>.
 */
public final class RiskSensitiveDqn {

    /**
     * The behaviour of a trained agent.
     *
     * @param fallRate falls per episode
     * @param bridgeRate share of episodes that took the bridge
     * @param meanReturn mean return
     */
    public record Behaviour(double fallRate, double bridgeRate, double meanReturn) {
    }

    private RiskSensitiveDqn() {
    }

    /**
     * @param steps environment steps per method
     * @param seeds seeds to average
     * @return method name to mean greedy return on the maze
     */
    public static Map<String, Double> maze(int steps, int seeds) {
        Map<String, Double> out = new LinkedHashMap<>();
        String[] names = {"IQN", "C51", "QR-DQN"};
        double[] sums = new double[names.length];
        for (int s = 0; s < seeds; s++) {
            long seed = 1 + s;
            int obs = 18;
            Learner[] learners = {
                ImplicitQuantileDqn.of(obs, 4, 64, seed).learningStarts(100)
                        .epsilon(Schedule.linear(1, 0.05, steps / 2)).seed(seed),
                RainbowDqn.of(() -> Nets.mlp(obs, 4 * 21, Activation.RELU, seed, 64), 4).categorical(21, -2, 2)
                        .seed(seed),
                RainbowDqn.of(() -> Nets.mlp(obs, 4 * 16, Activation.RELU, seed, 64), 4).quantile(16).seed(seed),
            };
            for (int i = 0; i < learners.length; i++) {
                GridWorld env = RainbowAblation.maze();
                learners[i].fit(env, steps);
                sums[i] += Rl.evaluate(learners[i], env, 1, seed);
            }
        }
        for (int i = 0; i < names.length; i++) {
            out.put(names[i], sums[i] / seeds);
        }
        return out;
    }

    /**
     * @param steps environment steps per agent
     * @param alpha the CVaR level of the risk-averse agent
     * @param seeds seeds to average
     * @return "risk-neutral" and "CVaR" to their greedy behaviour
     */
    public static Map<String, Behaviour> risk(int steps, float alpha, int seeds) {
        Map<String, Behaviour> out = new LinkedHashMap<>();
        double[][] sums = new double[2][3];
        for (int s = 0; s < seeds; s++) {
            long seed = 7 + s;
            for (int k = 0; k < 2; k++) {
                RiskyBridge env = RiskyBridge.standard();
                int obs = env.observationSpace().size();
                ImplicitQuantileDqn iqn = ImplicitQuantileDqn.of(obs, 2, 32, seed).learningStarts(100)
                        .quantiles(8, 8, 32).trainEvery(2).targetUpdateEvery(50)
                        .epsilon(Schedule.linear(1, 0.1, steps / 2)).seed(seed);
                iqn.fit(env, steps);
                if (k == 1) {
                    iqn.cvar(alpha);
                }
                double[] b = behaviour(iqn, 200, seed);
                for (int i = 0; i < 3; i++) {
                    sums[k][i] += b[i];
                }
            }
        }
        out.put("risk-neutral", new Behaviour(sums[0][0] / seeds, sums[0][1] / seeds, sums[0][2] / seeds));
        out.put("CVaR", new Behaviour(sums[1][0] / seeds, sums[1][1] / seeds, sums[1][2] / seeds));
        return out;
    }

    private static double[] behaviour(ImplicitQuantileDqn agent, int episodes, long seed) {
        RiskyBridge env = RiskyBridge.standard();
        SplittableRandom random = new SplittableRandom(seed + 99);
        double ret = 0;
        int bridge = 0;
        for (int e = 0; e < episodes; e++) {
            float[] o = env.reset(random.split());
            boolean first = true;
            Environment.Step st;
            do {
                float[] a = agent.act(o, false);
                if (first && Math.round(a[0]) == 0) {
                    bridge++;
                }
                first = false;
                st = env.step(a);
                ret += st.reward();
                o = st.observation();
            } while (!st.done());
        }
        return new double[] {env.falls() / (double) episodes, bridge / (double) episodes, ret / episodes};
    }
}
