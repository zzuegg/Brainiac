package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.LatentThoughtLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Coconut's curriculum on multi-hop pointer chasing, a tiny relative of the paper's graph
 * reachability (ProsQA): the question lists a function {@code f} on {@code {0..3}} as four
 * tokens (and a fixed filler) and a start {@code x}; the answer is {@code f(f(x))}. The
 * language chain of thought writes the two hops. Stage {@code k} replaces the first {@code k} written hops
 * with {@code k} continuous thoughts; by the last stage the model hops entirely in latent
 * space and writes only the answer.
 */
public final class CoconutCurriculum {

    /** The token {@code =}; values are 0-3. */
    public static final int EQUALS = 10;
    /** The end token. */
    public static final int END = 11;
    private static final int HOPS = 2;

    /**
     * Held-out accuracy.
     *
     * @param latent the fully latent model (all hops as continuous thoughts)
     * @param language the same model at stage 0, writing its hops as tokens
     * @param noThought a model trained to answer directly, with neither
     */
    public record Result(double latent, double language, double noThought) {
    }

    private CoconutCurriculum() {
    }

    static int[] hops(int[] q) {
        int[] h = new int[HOPS];
        int x = q[5];
        for (int i = 0; i < HOPS; i++) {
            x = q[x];
            h[i] = x;
        }
        return h;
    }

    static int[] prefix(int[] q) {
        return new int[] {q[0], q[1], q[2], q[3], q[4], q[5], EQUALS};
    }

    /**
     * @param stepsPerStage optimizer steps per curriculum stage
     * @param seed the seed
     * @return the accuracies
     */
    public static Result run(int stepsPerStage, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        List<int[]> train = new ArrayList<>();
        List<int[]> test = new ArrayList<>();
        java.util.Set<String> seen = new java.util.HashSet<>();
        while (train.size() + test.size() < 1024) {
            int[] q = new int[6];
            for (int i = 0; i < 6; i++) {
                q[i] = i == 4 ? 0 : random.nextInt(4);
            }
            if (seen.add(java.util.Arrays.toString(q))) {
                (train.size() < 768 ? train : test).add(q);
            }
        }
        LatentThoughtLm coconut = LatentThoughtLm.of(12, 64, seed);
        Optimizer opt = Optimizer.adam(5e-3f);
        double language = 0;
        for (int stage = 0; stage <= HOPS; stage++) {
            for (int s = 0; s < stepsPerStage; s++) {
                step(coconut, opt, train, stage, random);
            }
            if (stage == 0) {
                language = accuracy(coconut, test, 0);
            }
        }
        double latent = accuracy(coconut, test, HOPS);

        LatentThoughtLm direct = LatentThoughtLm.of(12, 64, seed);
        Optimizer opt2 = Optimizer.adam(5e-3f);
        for (int s = 0; s < (HOPS + 1) * stepsPerStage; s++) {
            step(direct, opt2, train, -1, random);
        }
        return new Result(latent, language, accuracy(direct, test, 0));
    }

    /** One step at a stage; stage -1 trains a direct answer with no chain of thought at all. */
    private static void step(LatentThoughtLm lm, Optimizer opt, List<int[]> train, int stage,
            SplittableRandom random) {
        int batch = 64;
        int thoughts = Math.max(0, stage);
        int written = stage < 0 ? 0 : HOPS - stage;
        int[][] prefix = new int[batch][];
        int[][] targets = new int[batch][];
        for (int i = 0; i < batch; i++) {
            int[] q = train.get(random.nextInt(train.size()));
            int[] h = hops(q);
            prefix[i] = prefix(q);
            int[] t = new int[written + 2];
            System.arraycopy(h, HOPS - written, t, 0, written);
            t[written] = h[HOPS - 1];
            t[written + 1] = END;
            targets[i] = t;
        }
        Var loss = lm.loss(prefix, thoughts, targets);
        loss.backward();
        opt.step(lm.parameters());
    }

    private static double accuracy(LatentThoughtLm lm, List<int[]> test, int thoughts) {
        int right = 0;
        for (int[] q : test) {
            int[] out = lm.generate(prefix(q), thoughts, HOPS + 3, END);
            int answer = hops(q)[HOPS - 1];
            if (out.length >= 2 && out[out.length - 1] == END && out[out.length - 2] == answer) {
                right++;
            }
        }
        return (double) right / test.size();
    }
}
