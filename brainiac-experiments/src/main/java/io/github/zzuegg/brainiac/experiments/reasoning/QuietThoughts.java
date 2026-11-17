package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.QuietStar;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Quiet-STaR on Coconut's two-hop pointer chasing ({@link CoconutCurriculum}): the answer is
 * {@code f(f(x))}, and nothing in the data shows the intermediate hop. A {@link WindowLm}
 * trained with {@link QuietStar} may think one hidden token before each output token; the
 * thought is rewarded only for making the true next token more likely. It is compared with
 * the same model trained to answer directly, with no thoughts, at the same number of
 * steps, on held-out questions.
 */
public final class QuietThoughts {

    /** The start-of-thought token (unused by the task's values 0-3, '=' and the end). */
    public static final int START_THOUGHT = 8;
    /** The end-of-thought token. */
    public static final int END_THOUGHT = 9;

    /**
     * Held-out accuracy.
     *
     * @param quietStar the model that thinks before each token
     * @param noThought the same model trained to answer directly
     * @param hopThoughts the share of held-out questions whose answer-position thought is the
     *     intermediate hop {@code f(x)}
     */
    public record Result(double quietStar, double noThought, double hopThoughts) {
    }

    private QuietThoughts() {
    }

    /** {@code {train, test}} questions of six tokens: {@code f(0..3)}, a filler, {@code x}. */
    static List<List<int[]>> questions(long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        List<int[]> train = new ArrayList<>();
        List<int[]> test = new ArrayList<>();
        Set<String> seen = new HashSet<>();
        while (train.size() + test.size() < 1024) {
            int[] q = new int[6];
            for (int i = 0; i < 6; i++) {
                q[i] = i == 4 ? 0 : random.nextInt(4);
            }
            if (seen.add(Arrays.toString(q))) {
                (train.size() < 768 ? train : test).add(q);
            }
        }
        return List.of(train, test);
    }

    private static int[] target(int[] q) {
        return new int[] {CoconutCurriculum.hops(q)[1], CoconutCurriculum.END};
    }

    /**
     * @param steps optimizer steps for each model
     * @param seed the seed
     * @return the accuracies
     */
    public static Result run(int steps, long seed) {
        List<List<int[]>> data = questions(seed);
        List<int[]> train = data.get(0);
        List<int[]> test = data.get(1);
        SplittableRandom random = new SplittableRandom(seed);

        WindowLm thinker = WindowLm.of(12, 12, 64, seed);
        QuietStar qs = QuietStar.of(thinker, START_THOUGHT, END_THOUGHT).thoughts(1, 4);
        WindowLm direct = WindowLm.of(12, 12, 64, seed);
        Optimizer o1 = Optimizer.adam(3e-3f);
        Optimizer o2 = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            List<int[]> ps = new ArrayList<>();
            List<int[]> ts = new ArrayList<>();
            for (int i = 0; i < 64; i++) {
                int[] q = train.get(random.nextInt(train.size()));
                ps.add(CoconutCurriculum.prefix(q));
                ts.add(target(q));
            }
            qs.step(ps, ts, o1, random);
            direct.supervisedStep(ps, ts, o2);
        }
        int right = 0;
        int rightDirect = 0;
        int hop = 0;
        for (int[] q : test) {
            int[] p = CoconutCurriculum.prefix(q);
            right += Arrays.equals(qs.generate(p, 2, CoconutCurriculum.END), target(q)) ? 1 : 0;
            rightDirect += Arrays.equals(direct.greedy(p, 2, CoconutCurriculum.END), target(q)) ? 1 : 0;
            hop += qs.think(p)[0] == CoconutCurriculum.hops(q)[0] ? 1 : 0;
        }
        double n = test.size();
        return new Result(right / n, rightDirect / n, hop / n);
    }
}
