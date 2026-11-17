package io.github.zzuegg.brainiac.experiments.data;

import io.github.zzuegg.brainiac.experiments.data.DataCuration.Grammar;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Model collapse when models train on their own output (Gerstgrasser et al. 2024, "Is Model
 * Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic
 * Data", <a href="https://arxiv.org/abs/2404.01413">arXiv:2404.01413</a>). A
 * {@link ContextLm} is trained on a small real corpus; each later generation is trained from
 * scratch on documents sampled from the previous generation. If every generation
 * <em>replaces</em> its data with the new samples, estimation errors compound: rare
 * continuations that a finite sample misses are lost for good, the model drifts away from
 * the real distribution and its output gets less diverse. If the data <em>accumulates</em>
 * (the real corpus plus every generation's samples) the error stays bounded, which is the
 * paper's result for language models, diffusion models and VAEs alike.
 */
public final class ModelCollapse {

    /** How each generation's training data is formed. */
    public enum Mode {
        /** Only the previous generation's samples. */
        REPLACE,
        /** The real data plus all samples so far. */
        ACCUMULATE
    }

    /**
     * One generation's model.
     *
     * @param generation the generation, 0 trained on real data only
     * @param testLoss its loss on held-out real documents
     * @param diversity distinct trigrams in 200 of its samples, over those in 200 real documents
     */
    public record Generation(int generation, float testLoss, double diversity) {
    }

    private ModelCollapse() {
    }

    private static double distinctTrigrams(List<int[]> docs) {
        Set<Integer> seen = new HashSet<>();
        for (int[] d : docs) {
            for (int t = 2; t < d.length; t++) {
                seen.add((d[t - 2] * 64 + d[t - 1]) * 64 + d[t]);
            }
        }
        return seen.size();
    }

    /**
     * Runs the generations.
     *
     * @param mode replace or accumulate
     * @param generations generations after the first
     * @param realDocuments the size of the real corpus, and of every generation's sample
     * @param seed the seed
     * @return one entry per generation, generation 0 first
     */
    public static Generation[] run(Mode mode, int generations, int realDocuments, long seed) {
        Grammar g = Grammar.random(16, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        List<int[]> real = g.documents(realDocuments, 16, r);
        List<int[]> test = g.documents(300, 16, new SplittableRandom(seed + 2));
        double realDiversity = distinctTrigrams(g.documents(200, 16, new SplittableRandom(seed + 3)));
        Generation[] out = new Generation[generations + 1];
        List<int[]> data = new ArrayList<>(real);
        for (int gen = 0; gen <= generations; gen++) {
            ContextLm lm = new ContextLm(16, 48, seed + 10 + gen).train(data, 400, 32, 1e-2f, seed + 20 + gen);
            double diversity = distinctTrigrams(lm.sample(200, 16, real, new SplittableRandom(seed + 30 + gen)))
                    / realDiversity;
            out[gen] = new Generation(gen, lm.loss(test), diversity);
            List<int[]> synthetic = lm.sample(realDocuments, 16, data, new SplittableRandom(seed + 40 + gen));
            if (mode == Mode.REPLACE) {
                data = synthetic;
            } else {
                data.addAll(synthetic);
            }
        }
        return out;
    }
}
