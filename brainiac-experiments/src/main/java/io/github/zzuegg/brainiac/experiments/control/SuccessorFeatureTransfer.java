package io.github.zzuegg.brainiac.experiments.control;

import io.github.zzuegg.brainiac.experimental.rl.ObjectGrid;
import io.github.zzuegg.brainiac.experimental.rl.SuccessorFeatures;
import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.Schedule;
import io.github.zzuegg.brainiac.rl.Tabular;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * W6 3.13: zero-shot transfer with successor features and GPI on an {@link ObjectGrid} with
 * three object types. The agent learns one policy per training task (each rewards one type),
 * then faces new weight vectors it never trained on. Reported, averaged over the test tasks:
 * the GPI agent's zero-shot greedy return (with the true weights, and with weights inferred
 * from 2000 random transitions), against core tabular Q-learning trained from scratch on each
 * test task for a small and a large budget, and the best return of any single stored policy.
 *
 * <p>Paper: Barreto et al., <a href="https://arxiv.org/abs/1606.05312">arXiv:1606.05312</a>.
 */
public final class SuccessorFeatureTransfer {

    /** Objects of types 0, 1 and 2 around the start. */
    public static final String[] MAP = {
        "0.....1",
        ".......",
        "...#...",
        "...S...",
        ".......",
        ".......",
        "2.....1",
    };

    private SuccessorFeatureTransfer() {
    }

    /**
     * @param trainSteps steps per training task
     * @param retrainSmall the small Q-learning budget on each test task
     * @param retrainLarge the large Q-learning budget on each test task
     * @param seed the seed
     * @return method name to mean greedy return over the test tasks
     */
    public static Map<String, Double> run(int trainSteps, int retrainSmall, int retrainLarge, long seed) {
        float step = -0.05f;
        float[][] train = {{1, 0, 0, step}, {0, 1, 0, step}, {0, 0, 1, step}};
        float[][] test = {{1, -1, 0.5f, step}, {-1, 0.5f, 1, step}, {0.3f, 1, -1, step}, {-1, -1, 1, step}};
        ObjectGrid base = new ObjectGrid(MAP, train[0]).maxSteps(40);
        SuccessorFeatures sf = SuccessorFeatures.of(4, 4).seed(seed);
        for (float[] w : train) {
            sf.fit(base.withWeights(w), trainSteps);
        }
        double gpi = 0;
        double inferred = 0;
        double small = 0;
        double large = 0;
        for (int t = 0; t < test.length; t++) {
            ObjectGrid env = base.withWeights(test[t]);
            sf.task(test[t]);
            gpi += Rl.evaluate(sf, env, 1, seed);
            sf.task(SuccessorFeatures.inferWeights(randomFeatures(env, 2000, seed, null),
                    randomRewards(env, 2000, seed), 1e-3));
            inferred += Rl.evaluate(sf, env, 1, seed);
            small += retrain(env, retrainSmall, seed + t);
            large += retrain(env, retrainLarge, seed + t);
        }
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("SF + GPI, zero-shot", gpi / test.length);
        out.put("SF + GPI, inferred w", inferred / test.length);
        out.put("Q-learning, " + retrainSmall + " steps", small / test.length);
        out.put("Q-learning, " + retrainLarge + " steps", large / test.length);
        return out;
    }

    private static double retrain(ObjectGrid env, int steps, long seed) {
        Tabular q = Tabular.qLearning(4).learningRate(0.5).discount(0.95).epsilon(Schedule.constant(0.1)).seed(seed);
        q.fit(env, steps);
        return Rl.evaluate(q, env, 1, seed);
    }

    private static List<float[]> randomFeatures(ObjectGrid env, int steps, long seed, List<Float> rewards) {
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> phis = new ArrayList<>();
        env.reset(r.split());
        for (int i = 0; i < steps; i++) {
            Environment.Step s = env.step(new float[] {r.nextInt(4)});
            phis.add(env.lastFeatures());
            if (rewards != null) {
                rewards.add(s.reward());
            }
            if (s.done()) {
                env.reset(r.split());
            }
        }
        return phis;
    }

    private static List<Float> randomRewards(ObjectGrid env, int steps, long seed) {
        List<Float> rewards = new ArrayList<>();
        randomFeatures(env, steps, seed, rewards);
        return rewards;
    }
}
