package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.experimental.rl.DiffusionWorldModel;
import io.github.zzuegg.brainiac.experimental.rl.DiscreteWorldModel;
import io.github.zzuegg.brainiac.experimental.rl.DreamerAgent;
import io.github.zzuegg.brainiac.experimental.rl.MlpWorldModel;
import io.github.zzuegg.brainiac.rl.CartPole;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A DIAMOND-style {@link DiffusionWorldModel} against the regression {@link MlpWorldModel} of
 * {@link DreamingAgent} on cart-pole (<a href="https://arxiv.org/abs/2405.12399">arXiv:2405.12399</a>):
 * open-loop rollout fidelity on held-out real action sequences, and the real-environment
 * return of a {@link DreamerAgent} that learns only in each model's imagination.
 */
public final class DiffusionDreaming {

    private DiffusionDreaming() {
    }

    /**
     * Rollout fidelity of one model.
     *
     * @param oneStep root mean squared error of the first predicted observation
     * @param tenStep after ten open-loop steps
     */
    public record Fidelity(double oneStep, double tenStep) {
    }

    /**
     * @param transitions random experience to train on
     * @param trainSteps training steps of 64 transitions
     * @param seed the seed
     * @return {MLP, diffusion} fidelity
     */
    public static Fidelity[] fidelity(int transitions, int trainSteps, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        List<Transition> train = episodes(transitions, random);
        List<Transition> test = episodes(3000, random);
        DiscreteWorldModel[] models = {MlpWorldModel.of(4, 2, 64, seed), DiffusionWorldModel.of(4, 2, 64, 10f, seed)};
        Fidelity[] out = new Fidelity[models.length];
        for (int m = 0; m < models.length; m++) {
            SplittableRandom r = new SplittableRandom(seed + 1);
            for (int i = 0; i < trainSteps; i++) {
                List<Transition> batch = new ArrayList<>(64);
                for (int k = 0; k < 64; k++) {
                    batch.add(train.get(r.nextInt(train.size())));
                }
                models[m].train(batch);
            }
            out[m] = rollouts(models[m], test);
        }
        return out;
    }

    private static Fidelity rollouts(DiscreteWorldModel model, List<Transition> test) {
        List<Integer> starts = new ArrayList<>();
        for (int i = 0; i + 10 < test.size() && starts.size() < 64; i += 7) {
            boolean ok = true;
            for (int k = 0; k < 10; k++) {
                ok &= !test.get(i + k).terminated() && sameEpisode(test.get(i + k), test.get(i + k + 1));
            }
            if (ok) {
                starts.add(i);
            }
        }
        int n = starts.size();
        float[] s = new float[n * 4];
        for (int j = 0; j < n; j++) {
            System.arraycopy(test.get(starts.get(j)).observation(), 0, s, j * 4, 4);
        }
        Tensor state = Tensor.of(Shape.of(n, 4), s);
        double one = 0;
        double ten = 0;
        for (int k = 0; k < 10; k++) {
            int[] acts = new int[n];
            for (int j = 0; j < n; j++) {
                acts[j] = Math.round(test.get(starts.get(j) + k).action()[0]);
            }
            state = model.predict(state, acts).next();
            float[] p = state.toArray();
            double err = 0;
            for (int j = 0; j < n; j++) {
                float[] truth = test.get(starts.get(j) + k).nextObservation();
                for (int c = 0; c < 4; c++) {
                    err += (p[j * 4 + c] - truth[c]) * (p[j * 4 + c] - truth[c]);
                }
            }
            err = Math.sqrt(err / (n * 4));
            if (k == 0) {
                one = err;
            }
            ten = err;
        }
        return new Fidelity(one, ten);
    }

    private static boolean sameEpisode(Transition a, Transition b) {
        float[] x = a.nextObservation();
        float[] y = b.observation();
        for (int i = 0; i < x.length; i++) {
            if (x[i] != y[i]) {
                return false;
            }
        }
        return true;
    }

    private static List<Transition> episodes(int n, SplittableRandom random) {
        Environment env = new CartPole(200);
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

    /**
     * Dreamer's real return with each world model.
     *
     * @param steps real environment steps
     * @param seed the seed
     * @return {MLP world model, diffusion world model} mean greedy return after training
     */
    public static double[] dreamer(long steps, long seed) {
        Environment env = new CartPole(200);
        DreamerAgent mlp = DreamerAgent.of(4, 2, 64, seed).schedule(250, 80, 40).horizon(15);
        mlp.fit(env, steps, null);
        DreamerAgent diffusion = DreamerAgent.of(4, 2, 64, seed).schedule(250, 80, 40).horizon(15)
                .worldModel(DiffusionWorldModel.of(4, 2, 64, 10f, seed).samplingSteps(3));
        diffusion.fit(env, steps, null);
        return new double[] {Rl.evaluate(mlp, env, 10, seed), Rl.evaluate(diffusion, env, 10, seed)};
    }
}
