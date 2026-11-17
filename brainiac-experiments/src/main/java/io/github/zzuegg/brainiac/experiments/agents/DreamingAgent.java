package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.experimental.rl.DreamerAgent;
import io.github.zzuegg.brainiac.experimental.rl.JepaWorldModel;
import io.github.zzuegg.brainiac.rl.CartPole;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Transition;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Experiment X25, world models: a {@link DreamerAgent} learns cart-pole while training its
 * policy only in imagination, and a {@link JepaWorldModel} learns to predict the next
 * observation's embedding on random cart-pole experience.
 */
public final class DreamingAgent {

    /**
     * Dreamer's outcome.
     *
     * @param randomReturn the mean return of a random policy
     * @param before the greedy return before training
     * @param after after training
     */
    public record DreamResult(double randomReturn, double before, double after) {
    }

    /**
     * JEPA's outcome.
     *
     * @param modelError the latent prediction error with the action-conditioned predictor
     * @param noChangeError the error of predicting {@code z' = z}
     * @param latentStd the mean standard deviation of the latent dimensions
     */
    public record JepaResult(double modelError, double noChangeError, double latentStd) {
    }

    private DreamingAgent() {
    }

    /**
     * @param steps real environment steps
     * @param seed the seed
     * @return the outcome
     */
    public static DreamResult dreamer(long steps, long seed) {
        Environment env = new CartPole(200);
        DreamerAgent agent = DreamerAgent.of(4, 2, 64, seed).schedule(250, 80, 40).horizon(15);
        double random = randomReturn(env, 20, seed);
        double before = Rl.evaluate(agent, env, 10, seed);
        agent.fit(env, steps, null);
        double after = Rl.evaluate(agent, env, 10, seed);
        return new DreamResult(random, before, after);
    }

    /**
     * @param transitions random experience to collect
     * @param trainSteps training steps
     * @param seed the seed
     * @return the outcome on held-out experience
     */
    public static JepaResult jepa(int transitions, int trainSteps, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        List<Transition> train = collect(new CartPole(200), transitions, random);
        List<Transition> test = collect(new CartPole(200), 256, random);
        JepaWorldModel model = JepaWorldModel.of(4, 2, 8, 64, seed);
        double std = 0;
        for (int i = 0; i < trainSteps; i++) {
            List<Transition> batch = new ArrayList<>(64);
            for (int k = 0; k < 64; k++) {
                batch.add(train.get(random.nextInt(train.size())));
            }
            std = model.train(batch)[1];
        }
        return new JepaResult(model.error(test, true), model.error(test, false), std);
    }

    private static List<Transition> collect(Environment env, int n, SplittableRandom random) {
        List<Transition> out = new ArrayList<>(n);
        float[] obs = env.reset(random.split());
        while (out.size() < n) {
            float[] a = {random.nextInt(2)};
            Environment.Step s = env.step(a);
            out.add(new Transition(obs, a, s.reward(), s.observation(), s.terminated(), 0.99f));
            obs = s.done() ? env.reset(random.split()) : s.observation();
        }
        return out;
    }

    private static double randomReturn(Environment env, int episodes, long seed) {
        SplittableRandom random = new SplittableRandom(seed + 7);
        double total = 0;
        for (int e = 0; e < episodes; e++) {
            env.reset(random.split());
            boolean done = false;
            while (!done) {
                Environment.Step s = env.step(new float[] {random.nextInt(2)});
                total += s.reward();
                done = s.done();
            }
        }
        return total / episodes;
    }
}
