package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.experimental.llm.DecodingPipeline;
import io.github.zzuegg.brainiac.experimental.llm.EntropyAdaptiveSampler;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.llm.LogitsProcessors;
import io.github.zzuegg.brainiac.experimental.llm.TokenSampler;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Samplers at a high temperature on a decoder trained to count by 1 or by 2: after a
 * one-token prompt the stride is a genuine coin flip, after that every token is determined.
 * A good sampler keeps that first choice open (diversity: distinct continuations) and the rest
 * exact (accuracy: continuations that count by one stride throughout). Compared: greedy
 * decoding, plain temperature, top-p and min-p (the baselines), top-nσ, XTC and the
 * {@link EntropyAdaptiveSampler}.
 */
public final class SamplerShowdown {

    /**
     * One sampler's outcome.
     *
     * @param accuracy the share of continuations that follow one stride throughout
     * @param distinct distinct continuations over samples
     */
    public record Outcome(double accuracy, double distinct) {
    }

    private SamplerShowdown() {
    }

    /**
     * Runs every sampler.
     *
     * @param model a decoder trained with {@link CountingTasks} on strides 1 and 2
     * @param temperature the high temperature
     * @param samples continuations per sampler
     * @param length tokens per continuation
     * @param seed the seed
     * @return sampler name to outcome
     */
    public static Map<String, Outcome> run(HookedDecoder model, float temperature, int samples, int length,
            long seed) {
        Map<String, Supplier<DecodingPipeline>> samplers = new LinkedHashMap<>();
        samplers.put("greedy", () -> DecodingPipeline.of(model));
        samplers.put("temperature", () -> DecodingPipeline.of(model)
                .processor(LogitsProcessors.temperature(temperature)).sampler(TokenSampler.ancestral()));
        samplers.put("top-p 0.9", () -> DecodingPipeline.of(model).processor(LogitsProcessors.temperature(temperature))
                .processor(LogitsProcessors.topP(0.9)).sampler(TokenSampler.ancestral()));
        samplers.put("min-p 0.1", () -> DecodingPipeline.of(model).processor(LogitsProcessors.temperature(temperature))
                .processor(LogitsProcessors.minP(0.1)).sampler(TokenSampler.ancestral()));
        samplers.put("top-n-sigma 1", () -> DecodingPipeline.of(model)
                .processor(LogitsProcessors.temperature(temperature)).processor(LogitsProcessors.topNSigma(1))
                .sampler(TokenSampler.ancestral()));
        samplers.put("xtc after min-p", () -> DecodingPipeline.of(model).processor(LogitsProcessors.minP(0.1))
                .processor(LogitsProcessors.xtc(0.1, 0.5, seed)).sampler(TokenSampler.ancestral()));
        samplers.put("entropy-adaptive", () -> DecodingPipeline.of(model).sampler(new EntropyAdaptiveSampler(0.1,
                2.5, 1.5, temperature, temperature, 0.1)));
        Map<String, Outcome> out = new LinkedHashMap<>();
        int vocab = model.vocabSize();
        for (Map.Entry<String, Supplier<DecodingPipeline>> e : samplers.entrySet()) {
            DecodingPipeline pipe = e.getValue().get();
            SplittableRandom r = new SplittableRandom(seed);
            int correct = 0;
            Set<List<Integer>> seen = new HashSet<>();
            for (int s = 0; s < samples; s++) {
                int start = s % vocab;
                int[] tokens = pipe.generate(new int[] {start}, length, r);
                correct += follows(start, tokens, vocab) ? 1 : 0;
                List<Integer> key = new java.util.ArrayList<>();
                key.add(start);
                for (int t : tokens) {
                    key.add(t);
                }
                seen.add(key);
            }
            out.put(e.getKey(), new Outcome(correct / (double) samples, seen.size() / (double) samples));
        }
        return out;
    }

    /** Whether the tokens count on from {@code start} by 1 or by 2 throughout. */
    static boolean follows(int start, int[] tokens, int vocab) {
        for (int stride : new int[] {1, 2}) {
            boolean ok = true;
            for (int i = 0; i < tokens.length && ok; i++) {
                ok = tokens[i] == (start + (i + 1) * stride) % vocab;
            }
            if (ok) {
                return true;
            }
        }
        return false;
    }
}
