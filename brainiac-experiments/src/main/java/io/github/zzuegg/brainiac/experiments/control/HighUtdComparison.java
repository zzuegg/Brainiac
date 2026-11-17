package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.HighUtdSac;
import io.github.zzuegg.brainiac.rl.Pendulum;
import io.github.zzuegg.brainiac.rl.Rl;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * W6 3.10: many critic updates per environment step on the pendulum swing-up. SAC at UTD 1
 * and at a high UTD, against REDQ (ensemble with in-target minimum over a random subset) and
 * DroQ (two critics with dropout and layer normalization) at the same high UTD, all with the
 * same environment-step budget. Metrics: greedy return and the critics' normalized bias
 * against Monte Carlo returns.
 *
 * <p>Papers: Chen et al., <a href="https://arxiv.org/abs/2101.05982">arXiv:2101.05982</a> (REDQ);
 * Hiraoka et al., <a href="https://arxiv.org/abs/2110.02034">arXiv:2110.02034</a> (DroQ).
 */
public final class HighUtdComparison {

    private HighUtdComparison() {
    }

    /** A method's outcome. @param ret the mean greedy return @param bias the normalized Q bias */
    public record Outcome(double ret, double bias) {
    }

    /**
     * @param steps environment steps per method
     * @param episodeLength pendulum episode length
     * @param utd the high update-to-data ratio
     * @param seed the seed
     * @return method name to outcome
     */
    public static Map<String, Outcome> run(int steps, int episodeLength, int utd, long seed) {
        Pendulum env = new Pendulum(episodeLength);
        Map<String, HighUtdSac> learners = new LinkedHashMap<>();
        learners.put("SAC UTD 1", HighUtdSac.sac(3, 1, 32, 32));
        learners.put("SAC UTD " + utd, HighUtdSac.sac(3, 1, 32, 32).updateToData(utd));
        learners.put("REDQ UTD " + utd, HighUtdSac.redq(3, 1, 5, 2, 32, 32).updateToData(utd));
        learners.put("DroQ UTD " + utd, HighUtdSac.droq(3, 1, 0.01f, 32, 32).updateToData(utd));
        Map<String, Outcome> out = new LinkedHashMap<>();
        for (Map.Entry<String, HighUtdSac> e : learners.entrySet()) {
            HighUtdSac l = e.getValue().learningStarts(Math.min(300, steps / 4)).seed(seed);
            l.fit(env, steps);
            out.put(e.getKey(), new Outcome(Rl.evaluate(l, env, 3, seed), l.qBias(env, 3, seed + 7)));
        }
        return out;
    }
}
