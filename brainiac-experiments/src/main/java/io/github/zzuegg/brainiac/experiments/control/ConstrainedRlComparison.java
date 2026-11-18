package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.ConstrainedPolicyOptimization;
import io.github.zzuegg.brainiac.experimental.rl.CostedEnvironment;
import io.github.zzuegg.brainiac.experimental.rl.HazardGrid;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Ppo;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * W6 3.14: constrained RL on {@link HazardGrid#detour()}, where the short route crosses a
 * hazard (cost 1) and the safe detour is twice as long. With a cost limit of
 * {@code d} per episode, CPO and PPO-Lagrangian are compared with the core PPO trained on a
 * fixed reward shaping {@code r - k c}: with {@code k = 0} it ignores the hazard, and with a
 * small hand-picked {@code k} the shortcut still pays. Reported per method: the return and
 * expected cost of the final stochastic policy, averaged over seeds.
 *
 * <p>Papers: Achiam et al., <a href="https://arxiv.org/abs/1705.10528">arXiv:1705.10528</a>; Ray et
 * al., <a href="https://cdn.openai.com/safexp-short.pdf">Safety Gym</a>.
 */
public final class ConstrainedRlComparison {

    /**
     * The outcome of one method.
     *
     * @param meanReturn return of the final policy (unshaped reward)
     * @param meanCost expected cost per episode of the final policy
     */
    public record Outcome(double meanReturn, double meanCost) {
    }

    private ConstrainedRlComparison() {
    }

    /**
     * @param steps environment steps per method and seed
     * @param costLimit the allowed expected cost per episode
     * @param seeds how many seeds
     * @return method name to outcome
     */
    public static Map<String, Outcome> run(int steps, float costLimit, int seeds) {
        String[] names = {"CPO", "PPO-Lagrangian", "PPO, shaping k=0", "PPO, shaping k=0.1"};
        double[][] sums = new double[names.length][2];
        for (int s = 0; s < seeds; s++) {
            long seed = 3 + s;
            int obs = HazardGrid.detour().observationSpace().size();
            Space actions = Space.discrete(4);
            for (int m = 0; m < names.length; m++) {
                Agent agent;
                if (m < 2) {
                    ConstrainedPolicyOptimization.Method method = m == 0
                            ? ConstrainedPolicyOptimization.Method.CPO
                            : ConstrainedPolicyOptimization.Method.PPO_LAGRANGIAN;
                    ConstrainedPolicyOptimization l = ConstrainedPolicyOptimization.of(method,
                            Nets.mlp(obs, 4, seed, 32), Nets.mlp(obs, 1, seed + 1, 32), Nets.mlp(obs, 1, seed + 2, 32),
                            actions, costLimit).rolloutSteps(500).maxKl(0.05f).lagrangeRate(0.5f).seed(seed);
                    l.fit(HazardGrid.detour(), steps);
                    agent = l;
                } else {
                    float k = m == 2 ? 0f : 0.1f;
                    Ppo ppo = Ppo.of(Nets.mlp(obs, 4, seed, 32), Nets.mlp(obs, 1, seed + 1, 32), actions)
                            .rolloutSteps(500).minibatchSize(100).epochs(8).learningRate(3e-3f).entropyBonus(0.01f)
                            .seed(seed);
                    ppo.fit(shaped(HazardGrid.detour(), k), steps);
                    agent = ppo;
                }
                double[] e = evaluate(agent, HazardGrid.detour(), 100, seed);
                sums[m][0] += e[0];
                sums[m][1] += e[1];
            }
        }
        Map<String, Outcome> out = new LinkedHashMap<>();
        for (int m = 0; m < names.length; m++) {
            out.put(names[m], new Outcome(sums[m][0] / seeds, sums[m][1] / seeds));
        }
        return out;
    }

    /**
     * Runs the stochastic policy and measures return and cost.
     *
     * @param agent the agent
     * @param env the environment
     * @param episodes how many episodes
     * @param seed the seed
     * @return {@code {mean return, mean cost}}
     */
    public static double[] evaluate(Agent agent, CostedEnvironment env, int episodes, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        double ret = 0;
        double cost = 0;
        for (int e = 0; e < episodes; e++) {
            float[] o = env.reset(random.split());
            Environment.Step s;
            do {
                s = env.step(agent.act(o, true));
                ret += s.reward();
                cost += env.lastCost();
                o = s.observation();
            } while (!s.done());
        }
        return new double[] {ret / episodes, cost / episodes};
    }

    private static Environment shaped(CostedEnvironment inner, float k) {
        return new Environment() {
            @Override
            public Space observationSpace() {
                return inner.observationSpace();
            }

            @Override
            public Space actionSpace() {
                return inner.actionSpace();
            }

            @Override
            public float[] reset(SplittableRandom random) {
                return inner.reset(random);
            }

            @Override
            public Step step(float[] action) {
                Step s = inner.step(action);
                return new Step(s.observation(), s.reward() - k * inner.lastCost(), s.terminated(), s.truncated());
            }
        };
    }
}
