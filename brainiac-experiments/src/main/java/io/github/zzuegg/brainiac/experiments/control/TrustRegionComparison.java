package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.Trpo;
import io.github.zzuegg.brainiac.rl.CartPole;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Ppo;
import io.github.zzuegg.brainiac.rl.Rl;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Supplier;

/**
 * W6 3.1: TRPO against the core PPO and A2C on CartPole at the same number of environment
 * steps, over several seeds. Besides the greedy return it reports how far each policy moves
 * per update (mean KL between consecutive policies): TRPO holds it at its bound by
 * construction, PPO only approximately through clipping, and A2C not at all.
 *
 * <p>Paper: Schulman et al., <a href="https://arxiv.org/abs/1502.05477">arXiv:1502.05477</a>.
 */
public final class TrustRegionComparison {

    /**
     * The outcome of one method.
     *
     * @param meanReturn greedy return, averaged over seeds
     * @param minReturn the worst seed's greedy return
     * @param meanKl mean KL per update (NaN where not measured)
     * @param maxKl largest KL of any update (NaN where not measured)
     */
    public record Outcome(double meanReturn, double minReturn, double meanKl, double maxKl) {
    }

    private TrustRegionComparison() {
    }

    /**
     * @param steps environment steps per method and seed
     * @param seeds how many seeds
     * @return method name to outcome
     */
    public static Map<String, Outcome> run(int steps, int seeds) {
        Map<String, Outcome> out = new LinkedHashMap<>();
        Supplier<Environment> env = () -> new CartPole(200);
        int obs = 4;
        int actions = 2;
        double[][] ret = new double[3][seeds];
        double[][] kl = new double[3][seeds];
        double[] trpoMaxKl = new double[seeds];
        for (int s = 0; s < seeds; s++) {
            long seed = 11 + s;
            Trpo trpo = Trpo.of(Nets.mlp(obs, actions, seed, 32), Nets.mlp(obs, 1, seed + 1, 32),
                    env.get().actionSpace()).rolloutSteps(500).maxKl(0.02f).seed(seed);
            trpo.fit(env.get(), steps);
            ret[0][s] = Rl.evaluate(trpo, env.get(), 5, seed);
            kl[0][s] = trpo.klHistory().stream().mapToDouble(Double::doubleValue).average().orElse(0);
            trpoMaxKl[s] = trpo.klHistory().stream().mapToDouble(Double::doubleValue).max().orElse(0);

            Ppo ppo = Ppo.of(Nets.mlp(obs, actions, seed, 32), Nets.mlp(obs, 1, seed + 1, 32),
                    env.get().actionSpace()).rolloutSteps(500).minibatchSize(100).epochs(4).seed(seed);
            double[] ppoKl = {0, 0};
            ppo.fit(env.get(), steps);
            ppoKl[0] = ppo.diagnostics().getOrDefault("approxKl", Double.NaN);
            ret[1][s] = Rl.evaluate(ppo, env.get(), 5, seed);
            kl[1][s] = ppoKl[0];

            Ppo a2c = Ppo.a2c(Nets.mlp(obs, actions, seed, 32), Nets.mlp(obs, 1, seed + 1, 32),
                    env.get().actionSpace()).seed(seed);
            a2c.fit(env.get(), steps);
            ret[2][s] = Rl.evaluate(a2c, env.get(), 5, seed);
            kl[2][s] = a2c.diagnostics().getOrDefault("approxKl", Double.NaN);
        }
        String[] names = {"TRPO", "PPO (core)", "A2C (core)"};
        for (int m = 0; m < 3; m++) {
            out.put(names[m], new Outcome(mean(ret[m]), min(ret[m]), mean(kl[m]),
                    m == 0 ? max(trpoMaxKl) : Double.NaN));
        }
        return out;
    }

    static double mean(double[] x) {
        double s = 0;
        for (double v : x) {
            s += v;
        }
        return s / x.length;
    }

    static double min(double[] x) {
        double m = Double.POSITIVE_INFINITY;
        for (double v : x) {
            m = Math.min(m, v);
        }
        return m;
    }

    static double max(double[] x) {
        double m = Double.NEGATIVE_INFINITY;
        for (double v : x) {
            m = Math.max(m, v);
        }
        return m;
    }
}
