package io.github.zzuegg.brainiac.rl;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/** Evaluating and watching agents. */
public final class Rl {

    private Rl() {
    }

    /**
     * Plays episodes greedily (no exploration) and averages their total reward.
     *
     * @param agent the agent
     * @param env the environment
     * @param episodes how many episodes
     * @param seed the seed for the episodes
     * @return the mean total reward
     */
    public static double evaluate(Agent agent, Environment env, int episodes, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        double sum = 0;
        for (int e = 0; e < episodes; e++) {
            float[] obs = env.reset(random.split());
            Environment.Step s;
            do {
                s = env.step(agent.act(obs, false));
                sum += s.reward();
                obs = s.observation();
            } while (!s.done());
        }
        return sum / episodes;
    }

    /**
     * Plays one greedy episode and records a picture of every state.
     *
     * @param agent the agent
     * @param env the environment
     * @param seed the seed for the episode
     * @param maxFrames the most pictures to keep
     * @return {@link Environment#render()} after the reset and after every step
     */
    public static List<String> watch(Agent agent, Environment env, long seed, int maxFrames) {
        List<String> frames = new ArrayList<>();
        float[] obs = env.reset(new SplittableRandom(seed));
        frames.add(env.render());
        Environment.Step s;
        do {
            s = env.step(agent.act(obs, false));
            obs = s.observation();
            if (frames.size() < maxFrames) {
                frames.add(env.render());
            }
        } while (!s.done());
        return frames;
    }
}
