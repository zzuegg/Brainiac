package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.experimental.data.SeedPrefix;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Open-ended generation on a toy "invent a path" task: how diverse, valid and novel are a
 * language model's outputs, and does seed-conditioning ({@link SeedPrefix}) beat temperature
 * sampling at it? A fixed random directed graph (10 nodes, 3 out-edges each) defines the valid
 * outputs, every 3-node walk {@code a b c}; a {@link WindowLm} is fine-tuned on half of them and
 * then asked, with no further input, to produce walks. An output scores when it is valid,
 * unique among the samples and, for creativity, novel (not a training walk), following
 * Nagarajan et al. (<a href="https://arxiv.org/abs/2504.15266">2504.15266</a>).
 *
 * <p>Following "Artificial Hivemind" (Jiang et al., NeurIPS 2025 best paper), homogeneity is
 * measured within one model (the share of repeated outputs) and between two models trained on
 * the same data from different initializations (the Jaccard overlap of their output sets).
 */
public final class OpenEndedDiversity {

    private static final int NODES = 10;
    private static final int END = 10;
    private static final int BOS = 11;
    private static final int SEED_FIRST = 12;
    private static final int SEED_ALPHABET = 8;
    private static final int SEED_LENGTH = 4;

    /**
     * One way of generating.
     *
     * @param method the label
     * @param valid the share of samples that are valid walks
     * @param uniqueValid the share of samples that are valid and not repeated
     * @param novelUniqueValid the share that are valid, not repeated and not training walks
     */
    public record Outcome(String method, double valid, double uniqueValid, double novelUniqueValid) {
    }

    /**
     * @param outcomes temperature sampling at several temperatures, then seed-conditioning
     * @param intraRepeat the share of a baseline model's valid samples (at temperature 1) that
     *     repeat an earlier sample
     * @param interOverlap the Jaccard overlap of two baseline models' output sets
     */
    public record Result(List<Outcome> outcomes, double intraRepeat, double interOverlap) {
    }

    private OpenEndedDiversity() {
    }

    private static boolean[][] graph(SplittableRandom r) {
        boolean[][] g = new boolean[NODES][NODES];
        for (int a = 0; a < NODES; a++) {
            int added = 0;
            while (added < 3) {
                int b = r.nextInt(NODES);
                if (b != a && !g[a][b]) {
                    g[a][b] = true;
                    added++;
                }
            }
        }
        return g;
    }

    private static boolean valid(boolean[][] g, int[] out) {
        return out.length == 4 && out[3] == END && out[0] < NODES && out[1] < NODES && out[2] < NODES
                && g[out[0]][out[1]] && g[out[1]][out[2]];
    }

    private static WindowLm train(List<int[]> prompts, List<int[]> walks, int vocab, int steps, long seed) {
        WindowLm lm = WindowLm.of(vocab, 8, 64, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            lm.supervisedStep(prompts, walks, opt);
        }
        return lm;
    }

    private static Outcome score(String method, int[][] outs, boolean[][] g, Set<String> train) {
        Set<String> seen = new HashSet<>();
        int valid = 0;
        int unique = 0;
        int novel = 0;
        for (int[] o : outs) {
            if (!valid(g, o)) {
                continue;
            }
            valid++;
            String key = Arrays.toString(o);
            if (seen.add(key)) {
                unique++;
                novel += train.contains(key) ? 0 : 1;
            }
        }
        double n = outs.length;
        return new Outcome(method, valid / n, unique / n, novel / n);
    }

    private static Set<String> validSet(int[][] outs, boolean[][] g) {
        Set<String> s = new HashSet<>();
        for (int[] o : outs) {
            if (valid(g, o)) {
                s.add(Arrays.toString(o));
            }
        }
        return s;
    }

    /**
     * @param samples outputs drawn per method
     * @param steps supervised steps
     * @param seed the seed
     * @return the measurements
     */
    public static Result run(int samples, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        boolean[][] g = graph(r);
        List<int[]> walks = new ArrayList<>();
        for (int a = 0; a < NODES; a++) {
            for (int b = 0; b < NODES; b++) {
                for (int c = 0; c < NODES; c++) {
                    if (g[a][b] && g[b][c]) {
                        walks.add(new int[] {a, b, c, END});
                    }
                }
            }
        }
        for (int i = walks.size() - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int[] t = walks.get(i);
            walks.set(i, walks.get(j));
            walks.set(j, t);
        }
        List<int[]> train = walks.subList(0, walks.size() / 2);
        Set<String> trainKeys = new HashSet<>();
        for (int[] w : train) {
            trainKeys.add(Arrays.toString(w));
        }
        SeedPrefix seeds = new SeedPrefix(SEED_FIRST, SEED_ALPHABET, SEED_LENGTH);
        int vocab = seeds.vocabularySize();
        List<int[]> plain = new ArrayList<>();
        List<int[]> seeded = new ArrayList<>();
        for (int i = 0; i < train.size(); i++) {
            plain.add(new int[] {BOS});
            seeded.add(seeds.apply(new int[] {BOS}, r));   // one fixed random seed per training walk
        }
        WindowLm base = train(plain, train, vocab, steps, seed);
        WindowLm twin = train(plain, train, vocab, steps, seed + 100);
        WindowLm conditioned = train(seeded, train, vocab, steps, seed);

        List<Outcome> outcomes = new ArrayList<>();
        int[][] bos = new int[samples][];
        Arrays.fill(bos, new int[] {BOS});
        int[][] atOne = null;
        for (double t : new double[] {0.5, 1.0, 1.5}) {
            int[][] outs = base.sample(bos, 4, END, t, new SplittableRandom(seed + 7));
            atOne = t == 1.0 ? outs : atOne;
            outcomes.add(score("temperature " + t, outs, g, trainKeys));
        }
        int[][] prompts = new int[samples][];
        SplittableRandom sr = new SplittableRandom(seed + 9);
        for (int i = 0; i < samples; i++) {
            prompts[i] = seeds.apply(new int[] {BOS}, sr);
        }
        outcomes.add(score("seed + greedy", conditioned.sample(prompts, 4, END, 0, sr), g, trainKeys));
        outcomes.add(score("seed + temperature 0.5", conditioned.sample(prompts, 4, END, 0.5, sr), g, trainKeys));

        int validOne = 0;
        for (int[] o : atOne) {
            validOne += valid(g, o) ? 1 : 0;
        }
        Set<String> a = validSet(atOne, g);
        Set<String> b = validSet(twin.sample(bos, 4, END, 1.0, new SplittableRandom(seed + 8)), g);
        Set<String> union = new HashSet<>(a);
        union.addAll(b);
        Set<String> both = new HashSet<>(a);
        both.retainAll(b);
        double repeat = validOne == 0 ? 0 : 1 - (double) a.size() / validOne;
        return new Result(outcomes, repeat, union.isEmpty() ? 0 : (double) both.size() / union.size());
    }
}
