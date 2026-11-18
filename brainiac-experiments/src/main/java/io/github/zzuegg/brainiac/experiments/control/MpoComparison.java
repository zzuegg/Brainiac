package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.HighUtdSac;
import io.github.zzuegg.brainiac.experimental.rl.Mpo;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Pendulum;
import io.github.zzuegg.brainiac.rl.Ppo;
import io.github.zzuegg.brainiac.rl.Rl;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * W6 3.11: MPO against SAC and PPO on the pendulum swing-up, over several seeds with the
 * same environment-step budget. Metrics: the mean greedy return and its spread across
 * seeds (stability).
 *
 * <p>Paper: Abdolmaleki et al., <a href="https://arxiv.org/abs/1806.06920">arXiv:1806.06920</a>.
 */
public final class MpoComparison {

    private MpoComparison() {
    }

    /** A method's outcome. @param mean the mean return over seeds @param std its standard deviation */
    public record Outcome(double mean, double std) {
    }

    /**
     * @param steps environment steps per run
     * @param episodeLength pendulum episode length
     * @param seeds the number of seeds
     * @return method name to outcome
     */
    public static Map<String, Outcome> run(int steps, int episodeLength, int seeds) {
        Pendulum env = new Pendulum(episodeLength);
        String[] names = {"MPO", "SAC", "PPO (core)"};
        Map<String, Outcome> out = new LinkedHashMap<>();
        for (String name : names) {
            double[] r = new double[seeds];
            for (int s = 0; s < seeds; s++) {
                long seed = 1 + s;
                int warmup = Math.min(300, steps / 4);
                Learner l = switch (name) {
                    case "MPO" -> Mpo.of(3, 1, 32, 32).learningStarts(warmup).seed(seed);
                    case "SAC" -> HighUtdSac.sac(3, 1, 32, 32).learningStarts(warmup).seed(seed);
                    default -> Ppo.of(Nets.mlp(3, 1, seed, 32, 32), Nets.mlp(3, 1, seed + 1, 32, 32),
                            env.actionSpace()).rolloutSteps(200).seed(seed);
                };
                l.fit(env, steps);
                r[s] = Rl.evaluate(l, env, 3, seed);
            }
            double mean = 0;
            for (double v : r) {
                mean += v / seeds;
            }
            double var = 0;
            for (double v : r) {
                var += (v - mean) * (v - mean) / seeds;
            }
            out.put(name, new Outcome(mean, Math.sqrt(var)));
        }
        return out;
    }
}
