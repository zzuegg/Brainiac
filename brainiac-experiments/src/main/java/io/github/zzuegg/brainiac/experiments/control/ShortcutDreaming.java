package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.ImaginationTrainer;
import io.github.zzuegg.brainiac.experimental.rl.MlpWorldModel;
import io.github.zzuegg.brainiac.experimental.rl.ShortcutWorldModel;
import io.github.zzuegg.brainiac.experimental.rl.SlipperyTrack;
import io.github.zzuegg.brainiac.rl.Agent;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * W6 3.9: a Dreamer 4 style world model trained with shortcut forcing, from offline data
 * only, on the {@link SlipperyTrack} whose pushes slip with probability 0.3 (bimodal
 * dynamics). Two measurements:
 * <ul>
 *   <li>Rollout fidelity against sampling steps: the mean distance of sampled next states to
 *       the nearest state the real system can reach, for 1, 2, 4 and 8 flow steps, for the
 *       shortcut model, the same network trained by plain flow matching, and the
 *       deterministic {@link MlpWorldModel} (which predicts the mean, between the modes).</li>
 *   <li>Real return of policies trained purely in imagination ({@link ImaginationTrainer}) in
 *       the shortcut model (2 steps) and in the MLP world model, against a random policy.</li>
 * </ul>
 *
 * <p>Paper: Hafner et al., <a href="https://arxiv.org/abs/2509.24527">arXiv:2509.24527</a>;
 * shortcut targets: Frans et al., <a href="https://arxiv.org/abs/2410.12557">arXiv:2410.12557</a>.
 */
public final class ShortcutDreaming {

    /** Sampling steps measured. */
    public static final int[] STEPS = {1, 2, 4, 8};

    private ShortcutDreaming() {
    }

    private static List<Transition> randomData(Environment env, int episodes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Transition> out = new ArrayList<>();
        for (int e = 0; e < episodes; e++) {
            float[] o = env.reset(r.split());
            Environment.Step s;
            do {
                float[] a = {r.nextInt(3)};
                s = env.step(a);
                out.add(new Transition(o, a, s.reward(), s.observation(), s.terminated(), 1));
                o = s.observation();
            } while (!s.done());
        }
        return out;
    }

    /**
     * @param episodes offline random episodes (30 steps each)
     * @param modelSteps world-model gradient steps
     * @param policySteps imagination batches for each policy
     * @param seed the seed
     * @return measurement name to value: "&lt;model&gt;, k steps" mode error and "&lt;policy&gt; return"
     */
    public static Map<String, Double> run(int episodes, int modelSteps, int policySteps, long seed) {
        SlipperyTrack env = new SlipperyTrack(0.3, 30);
        List<Transition> data = randomData(env, episodes, seed);
        List<Transition> test = new ArrayList<>();
        for (Transition t : randomData(env, 20, seed + 50)) {
            if (Math.round(t.action()[0]) != 1) {
                test.add(t);
            }
        }
        Map<String, Double> out = new LinkedHashMap<>();
        ShortcutWorldModel sc = ShortcutWorldModel.of(2, 3, 64, seed).train(data, modelSteps, 64);
        ShortcutWorldModel fm = ShortcutWorldModel.of(2, 3, 64, seed).shortcut(false).train(data, modelSteps, 64);
        for (int k : STEPS) {
            out.put("shortcut, " + k + " steps", modeError(test, (o, a) -> sc.sample(o, a, k)));
            out.put("flow matching, " + k + " steps", modeError(test, (o, a) -> fm.sample(o, a, k)));
        }
        MlpWorldModel mlp = MlpWorldModel.of(2, 3, 64, seed);
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < modelSteps; i++) {
            List<Transition> b = new ArrayList<>();
            for (int j = 0; j < 64; j++) {
                b.add(data.get(r.nextInt(data.size())));
            }
            mlp.train(b);
        }
        out.put("MLP world model", modeError(test, (o, a) -> mlp.predict(o, a).next()));

        List<float[]> starts = new ArrayList<>();
        for (Transition t : data) {
            starts.add(t.observation());
        }
        ImaginationTrainer inShortcut = ImaginationTrainer.of(2, 3, 32, seed).train(sc.sampleSteps(2), starts,
                policySteps);
        out.put("policy in shortcut model, return", Rl.evaluate(inShortcut, env, 5, seed));
        ImaginationTrainer inMlp = ImaginationTrainer.of(2, 3, 32, seed).train((o, a) -> {
            MlpWorldModel.Prediction p = mlp.predict(o, a);
            return new Tensor[] {p.next(), p.reward()};
        }, starts, policySteps);
        out.put("policy in MLP world model, return", Rl.evaluate(inMlp, env, 5, seed));
        out.put("random policy, return", Rl.evaluate(Agent.random(env.actionSpace(), seed), env, 5, seed));
        return out;
    }

    /** A batched next-state predictor. */
    private interface Predictor {
        Tensor next(Tensor obs, int[] acts);
    }

    private static double modeError(List<Transition> test, Predictor p) {
        int n = test.size();
        float[] o = new float[n * 2];
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(test.get(i).observation(), 0, o, 2 * i, 2);
            a[i] = Math.round(test.get(i).action()[0]);
        }
        float[] pred = p.next(Tensor.of(Shape.of(n, 2), o), a).to(DType.FLOAT32).toArray();
        double err = 0;
        for (int i = 0; i < n; i++) {
            double best = Double.POSITIVE_INFINITY;
            for (float[] q : SlipperyTrack.outcomes(test.get(i).observation(), a[i])) {
                double dx = pred[2 * i] - q[0];
                double dv = pred[2 * i + 1] - q[1];
                best = Math.min(best, Math.sqrt(dx * dx + dv * dv));
            }
            err += best;
        }
        return err / n;
    }
}
