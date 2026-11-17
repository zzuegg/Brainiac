package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.experimental.llm.Eagle3Draft;
import io.github.zzuegg.brainiac.experimental.llm.EagleDraft;
import io.github.zzuegg.brainiac.experimental.llm.PagedLlama;
import io.github.zzuegg.brainiac.llm.Generator;
import io.github.zzuegg.brainiac.llm.Llama;
import io.github.zzuegg.brainiac.llm.LlamaConfig;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * EAGLE-3 against EAGLE on the {@link TreeSpeculation} setup: one random {@link Llama} target,
 * draft heads trained on the target's greedy continuations of random prompts for the same
 * steps, measured by tokens per target pass (acceptance length plus one) and wall-clock tokens
 * per second, with every method lossless. Compared: {@link EagleDraft} as a chain and as a
 * tree, {@link Eagle3Draft} (fused low/mid/high features, training-time test) as a chain and as
 * a tree (also with EAGLE's feature-regression loss added back), and its P-EAGLE mode that
 * drafts the whole chain in one draft pass.
 *
 * <p>Measured result: on this toy (a 2-layer target, about a thousand training positions)
 * EAGLE-3 does <em>not</em> beat EAGLE (about 2.2 against 2.4 tokens per pass for chains). Its
 * gains in the paper come from scaling the training data, which the feature-regression loss it
 * drops would otherwise cap; at this scale the trainable fusion overfits (freezing it, with
 * the regression loss and no training-time test, matched EAGLE in a side run) and
 * training-time test spends head capacity on later steps.
 * P-EAGLE's parallel chain keeps the sequential chain's acceptance with a quarter of the draft
 * passes.
 */
public final class Eagle3Speculation {

    /**
     * One method.
     *
     * @param tokens the generated tokens over every prompt, joined
     * @param passes target passes over every prompt
     * @param draftPasses draft-head passes over every prompt
     * @param millis wall time over every prompt
     */
    public record Outcome(int[] tokens, int passes, int draftPasses, double millis) {

        /** @return tokens per target pass */
        public double tokensPerPass() {
            return tokens.length / (double) passes;
        }

        /** @return tokens per second of wall time */
        public double tokensPerSecond() {
            return tokens.length / (millis / 1000);
        }
    }

    private Eagle3Speculation() {
    }

    /**
     * Runs every method.
     *
     * @param trainPrompts prompts whose continuations train the heads
     * @param trainSteps training steps per head
     * @param testPrompts prompts measured
     * @param maxNew tokens per prompt
     * @param seed the seed
     * @return per method its outcome, greedy decoding first
     */
    public static Map<String, Outcome> run(int trainPrompts, int trainSteps, int testPrompts, int maxNew, long seed) {
        int vocab = 32;
        Llama target = Llama.random(LlamaConfig.small(vocab, 64, 2, 4, 2), seed, Device.cpu(), DType.FLOAT32);
        SplittableRandom r = new SplittableRandom(seed);
        Generator gen = new Generator(target, null);
        int[][] prompts = new int[trainPrompts][];
        for (int i = 0; i < trainPrompts; i++) {
            prompts[i] = ServingThroughput.randomTokens(r, 4, vocab);
        }
        List<Generator.Output> cont = gen.generate(prompts, Generator.Options.greedy(40), null);
        int[][] train = new int[trainPrompts][];
        for (int i = 0; i < trainPrompts; i++) {
            train[i] = concat(prompts[i], cont.get(i).tokens());
        }
        int[][] test = new int[testPrompts][];
        for (int i = 0; i < testPrompts; i++) {
            test[i] = ServingThroughput.randomTokens(r, 4, vocab);
        }
        Map<String, Outcome> out = new LinkedHashMap<>();
        int[] greedy = new int[0];
        long g0 = System.nanoTime();
        for (int[] p : test) {
            greedy = concat(greedy, gen.generate(new int[][] {p}, Generator.Options.greedy(maxNew), null).getFirst()
                    .tokens());
        }
        out.put("greedy", new Outcome(greedy, test.length * maxNew, 0, (System.nanoTime() - g0) / 1e6));
        PagedLlama paged = PagedLlama.of(target, 16, 256);
        EagleDraft eagle = new EagleDraft(paged, seed + 2);
        eagle.train(train, trainSteps, Optimizer.adam(3e-3f));
        Eagle3Draft eagle3 = new Eagle3Draft(paged, 4, seed + 3);
        eagle3.train(train, trainSteps, Optimizer.adam(3e-3f));
        Eagle3Draft eagle3f = new Eagle3Draft(paged, 4, seed + 3).featureLoss(10f);
        eagle3f.train(train, trainSteps, Optimizer.adam(3e-3f));
        int[][] shapes = {{1, 4, 1}, {2, 4, 3}};
        for (int s = 0; s < shapes.length; s++) {
            int[] sh = shapes[s];
            String kind = s == 0 ? "chain" : "tree";
            int[] toks = new int[0];
            int passes = 0;
            long t0 = System.nanoTime();
            for (int[] p : test) {
                EagleDraft.Result e = eagle.generate(p, maxNew, sh[0], sh[1], sh[2]);
                toks = concat(toks, e.tokens());
                passes += e.passes();
            }
            out.put("eagle " + kind, new Outcome(toks, passes, passes * sh[1], (System.nanoTime() - t0) / 1e6));
            out.put("eagle-3 " + kind, collect(test, p -> eagle3.generate(p, maxNew, sh[0], sh[1], sh[2])));
            out.put("eagle-3 + feature loss " + kind, collect(test, p -> eagle3f.generate(p, maxNew, sh[0], sh[1], sh[2])));
        }
        out.put("p-eagle (parallel chain)", collect(test, p -> eagle3.generateParallel(p, maxNew, 4)));
        return out;
    }

    private static Outcome collect(int[][] prompts, java.util.function.Function<int[], Eagle3Draft.Result> run) {
        int[] toks = new int[0];
        int passes = 0;
        int drafts = 0;
        double ms = 0;
        for (int[] p : prompts) {
            Eagle3Draft.Result e = run.apply(p);
            toks = concat(toks, e.tokens());
            passes += e.passes();
            drafts += e.draftPasses();
            ms += e.millis();
        }
        return new Outcome(toks, passes, drafts, ms);
    }

    private static int[] concat(int[] a, int[] b) {
        int[] c = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, c, a.length, b.length);
        return c;
    }
}
