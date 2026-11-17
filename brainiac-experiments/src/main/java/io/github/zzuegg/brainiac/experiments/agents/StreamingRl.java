package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.experimental.rl.StreamingActorCritic;
import io.github.zzuegg.brainiac.rl.CartPole;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Rl;
import java.util.List;

/**
 * Stream AC(lambda) on cart-pole: one update per transition, nothing stored.
 */
public final class StreamingRl {

    /**
     * @param before the greedy return before training
     * @param after after training
     * @param firstEpisodes the mean training return of the first 20 episodes
     * @param lastEpisodes of the last 20
     */
    public record Result(double before, double after, double firstEpisodes, double lastEpisodes) {
    }

    private StreamingRl() {
    }

    /**
     * @param steps training steps
     * @param seed the seed
     * @return the outcome
     */
    public static Result run(long steps, long seed) {
        Environment env = new CartPole(200);
        StreamingActorCritic agent = StreamingActorCritic.of(4, 2, 32, 1f, seed);
        double before = Rl.evaluate(agent, env, 10, seed);
        agent.fit(env, steps);
        List<Double> r = agent.episodeReturns();
        return new Result(before, Rl.evaluate(agent, env, 10, seed), mean(r.subList(0, Math.min(20, r.size()))),
                mean(r.subList(Math.max(0, r.size() - 20), r.size())));
    }

    private static double mean(List<Double> xs) {
        return xs.stream().mapToDouble(Double::doubleValue).average().orElse(0);
    }
}
