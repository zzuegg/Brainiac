package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.CemPlanner;
import io.github.zzuegg.brainiac.experimental.rl.HighUtdSac;
import io.github.zzuegg.brainiac.experimental.rl.TdMpc;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Pendulum;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * W6 3.8: TD-MPC on the pendulum swing-up with the same environment-step budget: the
 * decoder-free latent model planned with MPPI and a learned terminal value, the same model's
 * policy prior acting alone (no planning), SAC, and CEM planning with an observation-space
 * MLP model learned from random steps and the true reward. Metric: greedy return. (3.7's
 * PlaNet is measured on the same task in {@link LatentPlanningComparison}.)
 *
 * <p>Paper: Hansen et al., <a href="https://arxiv.org/abs/2310.16828">arXiv:2310.16828</a>.
 */
public final class TdMpcComparison {

    private TdMpcComparison() {
    }

    /**
     * @param steps environment steps per method
     * @param seed the seed
     * @return method name to greedy return
     */
    public static Map<String, Double> run(int steps, long seed) {
        Pendulum env = new Pendulum(100);
        Map<String, Double> out = new LinkedHashMap<>();
        int warmup = Math.min(300, steps / 4);
        TdMpc tdmpc = TdMpc.of(3, 1, 16, 64, seed).search(48, 8, 8, 3).learningStarts(warmup);
        tdmpc.fit(env, steps);
        out.put("TD-MPC (MPPI)", Rl.evaluate(tdmpc, env, 2, seed));
        tdmpc.planning(false);
        out.put("TD-MPC policy prior only", Rl.evaluate(tdmpc, env, 2, seed));
        HighUtdSac sac = HighUtdSac.sac(3, 1, 32, 32).learningStarts(warmup).seed(seed);
        sac.fit(env, steps);
        out.put("SAC", Rl.evaluate(sac, env, 2, seed));
        CemPlanner.Dynamics mlp = CemPlanner.learn(env, Nets.mlp(4, 3, Activation.RELU, seed, 64, 64), steps,
                1500, seed);
        CemPlanner cem = CemPlanner.of(mlp, ContinuousControlComparison::pendulumReward, (Space.Box) env.actionSpace())
                .horizon(10).search(80, 8, 3).seed(seed);
        out.put("MLP dynamics + CEM (true reward)", Rl.evaluate(cem, env, 2, seed));
        return out;
    }
}
