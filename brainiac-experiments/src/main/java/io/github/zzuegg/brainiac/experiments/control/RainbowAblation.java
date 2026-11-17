package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.nn.NoisyDense;
import io.github.zzuegg.brainiac.experimental.rl.RainbowDqn;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.Dqn;
import io.github.zzuegg.brainiac.rl.GridWorld;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.RlResult;
import io.github.zzuegg.brainiac.rl.Schedule;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Supplier;

/**
 * The Rainbow ablation on a small maze: the core DQN against each Rainbow ingredient
 * (distributional C51 and QR-DQN, noisy-net exploration) and their combination with
 * dueling heads, n-step returns and prioritized replay. Reported per variant: the greedy
 * return after training and the mean return over all training episodes (an
 * area-under-the-curve measure of how quickly it learned).
 *
 * <p>Paper: Hessel et al., "Rainbow", <a href="https://arxiv.org/abs/1710.02298">arXiv:1710.02298</a>.
 */
public final class RainbowAblation {

    /**
     * One variant's outcome.
     *
     * @param greedyReturn the return of the greedy policy after training
     * @param meanTrainingReturn the mean return of the training episodes
     */
    public record Outcome(double greedyReturn, double meanTrainingReturn) {
    }

    private RainbowAblation() {
    }

    /** @return the maze: 3 x 6 with a wall, 40-step episodes */
    public static GridWorld maze() {
        return GridWorld.of("S.....", ".####.", "....#G").maxSteps(40);
    }

    /**
     * Trains every variant.
     *
     * @param steps environment steps each
     * @param seed the seed
     * @return variant name to outcome
     */
    public static Map<String, Outcome> run(int steps, long seed) {
        int obs = 18;
        Map<String, Learner> variants = new LinkedHashMap<>();
        variants.put("DQN (core)", Dqn.of(() -> Nets.mlp(obs, 4, Activation.RELU, seed, 64), 4).learningStarts(100)
                .epsilon(Schedule.linear(1, 0.05, steps / 2)).targetUpdateEvery(100).seed(seed));
        Supplier<Model> c51 = () -> Nets.mlp(obs, 4 * 21, Activation.RELU, seed, 64);
        variants.put("C51", RainbowDqn.of(c51, 4).categorical(21, -2, 2).seed(seed));
        variants.put("QR-DQN", RainbowDqn.of(() -> Nets.mlp(obs, 4 * 16, Activation.RELU, seed, 64), 4).quantile(16)
                .seed(seed));
        variants.put("Noisy DQN", RainbowDqn.of(() -> Model.sequential().input(obs).dense(64, Activation.RELU)
                .add(NoisyDense.spec(4, 0.5f, seed)).seed(seed).build(), 4).seed(seed));
        variants.put("Rainbow", RainbowDqn.of(() -> Model.sequential().input(obs).dense(64, Activation.RELU)
                .add(NoisyDense.spec(5 * 21, 0.5f, seed)).seed(seed).build(), 4).categorical(21, -2, 2)
                .dueling(true).nSteps(3).prioritized(0.5, Schedule.linear(0.4, 1, steps)).seed(seed));
        Map<String, Outcome> out = new LinkedHashMap<>();
        for (Map.Entry<String, Learner> e : variants.entrySet()) {
            GridWorld env = maze();
            RlResult r = e.getValue().fit(env, steps);
            double mean = r.episodes().stream().mapToDouble(x -> x.totalReward()).average().orElse(Double.NaN);
            out.put(e.getKey(), new Outcome(Rl.evaluate(e.getValue(), env, 1, seed), mean));
        }
        return out;
    }
}
