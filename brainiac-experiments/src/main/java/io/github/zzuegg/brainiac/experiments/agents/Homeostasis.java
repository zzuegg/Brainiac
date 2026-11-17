package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.experimental.rl.ForagingCorridor;
import io.github.zzuegg.brainiac.experimental.rl.HomeostaticEnvironment;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Dqn;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Schedule;
import java.util.SplittableRandom;

/**
 * A DQN agent in a {@link HomeostaticEnvironment} around a {@link ForagingCorridor}: food and
 * water lie at opposite ends, both needs drift, and the only reward is drive reduction. The
 * agent has to learn to shuttle between the two and stop eating when sated.
 */
public final class Homeostasis {

    /**
     * @param survivalRandom the mean episode length of a random agent
     * @param survivalTrained of the trained agent (greedy)
     * @param maxSteps the episode length limit
     * @param meanDeviation the trained agent's mean absolute deviation from the set points
     * @param foodShare the share of food in everything the trained agent consumed
     */
    public record Result(double survivalRandom, double survivalTrained, int maxSteps, double meanDeviation,
            double foodShare) {
    }

    private Homeostasis() {
    }

    /**
     * @param steps training steps
     * @param seed the seed
     * @return the outcome
     */
    public static Result run(long steps, long seed) {
        int maxSteps = 200;
        ForagingCorridor world = ForagingCorridor.of(7, maxSteps, 0.25f);
        HomeostaticEnvironment env = HomeostaticEnvironment.of(world, s -> world.lastIntake(),
                new float[] {0.025f, 0.025f}, 1f);
        Dqn dqn = Dqn.of(() -> Nets.mlp(9, 3, Activation.RELU, seed, 64, 64), 3).learningStarts(500)
                .epsilon(Schedule.linear(1, 0.05, steps / 2)).targetUpdateEvery(250).nSteps(3).seed(seed);
        dqn.fit(env, steps);
        float[] stats = new float[4];
        double random = survival(env, world, Agent.random(env.actionSpace(), seed), 10, seed, new float[4]);
        double trained = survival(env, world, dqn, 10, seed, stats);
        return new Result(random, trained, maxSteps, stats[0] / stats[1], stats[2] / Math.max(1e-9, stats[2] + stats[3]));
    }

    private static double survival(HomeostaticEnvironment env, ForagingCorridor world, Agent agent, int episodes,
            long seed, float[] stats) {
        SplittableRandom random = new SplittableRandom(seed);
        long total = 0;
        for (int e = 0; e < episodes; e++) {
            float[] obs = env.reset(random.split());
            boolean done = false;
            while (!done) {
                Environment.Step s = env.step(agent.act(obs, false));
                total++;
                for (float d : env.deviation()) {
                    stats[0] += Math.abs(d);
                    stats[1]++;
                }
                float[] in = world.lastIntake();
                stats[2] += in[0];
                stats[3] += in[1];
                obs = s.observation();
                done = s.done();
            }
        }
        return (double) total / episodes;
    }
}
