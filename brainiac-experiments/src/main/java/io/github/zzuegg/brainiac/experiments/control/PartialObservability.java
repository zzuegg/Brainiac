package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.CueCorridor;
import io.github.zzuegg.brainiac.experimental.rl.FrameStackEnvironment;
import io.github.zzuegg.brainiac.experimental.rl.RecurrentReplayDqn;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Dqn;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Schedule;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * W6 3.4: R2D2 on the {@link CueCorridor} T-maze, where the cue is visible only on the first
 * step and the turn comes {@code length} steps later. Compared at equal environment steps:
 * R2D2 with stored recurrent state and burn-in, the same recurrent DQN replaying sequences
 * from a zero state without burn-in, and the core DQN on two stacked frames. Metric: greedy
 * success rate (0.5 is guessing).
 *
 * <p>Paper: Kapturowski et al., <a href="https://openreview.net/forum?id=r1lyTjAqYX">ICLR 2019</a>.
 */
public final class PartialObservability {

    private PartialObservability() {
    }

    /**
     * @param length corridor length
     * @param steps environment steps per method
     * @param seeds how many seeds to average
     * @return method name to greedy success rate
     */
    public static Map<String, Double> run(int length, int steps, int seeds) {
        Map<String, Double> out = new LinkedHashMap<>();
        String[] names = {"R2D2 (stored state, burn-in)", "zero state, no burn-in", "DQN, 2 stacked frames (core)"};
        double[] sums = new double[names.length];
        for (int s = 0; s < seeds; s++) {
            long seed = 21 + s;
            Schedule eps = Schedule.linear(1, 0.05, steps / 2);
            RecurrentReplayDqn stored = RecurrentReplayDqn.of(4, 3, 24, seed).burnIn(length).trainLength(length + 2)
                    .storedState(true).epsilon(eps).seed(seed);
            stored.fit(new CueCorridor(length), steps);
            sums[0] += success(stored, new CueCorridor(length), seed);
            RecurrentReplayDqn zero = RecurrentReplayDqn.of(4, 3, 24, seed).burnIn(0).trainLength(length + 2)
                    .storedState(false).epsilon(eps).seed(seed);
            zero.fit(new CueCorridor(length), steps);
            sums[1] += success(zero, new CueCorridor(length), seed);
            Dqn dqn = Dqn.of(() -> Nets.mlp(8, 3, seed, 32), 3).optimizer(Optimizer.adam(2e-3f)).learningStarts(300)
                    .epsilon(eps).discount(0.97f).seed(seed);
            dqn.fit(new FrameStackEnvironment(new CueCorridor(length), 2), steps);
            sums[2] += success(dqn, new FrameStackEnvironment(new CueCorridor(length), 2), seed);
        }
        for (int i = 0; i < names.length; i++) {
            out.put(names[i], sums[i] / seeds);
        }
        return out;
    }

    private static double success(Agent agent, Environment env, long seed) {
        SplittableRandom random = new SplittableRandom(seed + 1000);
        int wins = 0;
        int episodes = 40;
        for (int e = 0; e < episodes; e++) {
            float[] o = env.reset(random.split());
            agent.startEpisode();
            Environment.Step st;
            do {
                st = env.step(agent.act(o, false));
                o = st.observation();
            } while (!st.done());
            if (st.reward() > 0) {
                wins++;
            }
        }
        return wins / (double) episodes;
    }
}
