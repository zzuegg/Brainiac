package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.experimental.llm.KneserNeyLm;
import io.github.zzuegg.brainiac.experimental.llm.SuffixTreeDrafter;
import io.github.zzuegg.brainiac.llm.Generator;
import io.github.zzuegg.brainiac.llm.PromptLookup;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Model-free speculation on repetitive agent traces: {@link SuffixTreeDrafter} (a suffix tree
 * over earlier outputs and the request, adaptive draft length) against prompt-lookup decoding
 * (the core {@link PromptLookup} through {@code Generator.speculative}, 4 guesses per pass) and
 * plain greedy decoding. The target is an order-8 {@link KneserNeyLm} trained on synthetic
 * tool-use traces ({@code THINK w w CALL tool ARG x END OBS r r r END} per task word), so its
 * greedy outputs follow the trace templates; requests arrive one after another and the suffix
 * tree learns from every finished output. All methods must give greedy decoding's tokens.
 * Reported: tokens per target pass and target tokens verified per output token (the compute
 * spent on rejected guesses).
 */
public final class SuffixDecoding {

    /**
     * One method.
     *
     * @param tokens every request's output, joined
     * @param passes target passes
     * @param verified tokens read by the target, prompts excluded
     */
    public record Outcome(int[] tokens, int passes, long verified) {

        /** @return tokens per target pass */
        public double tokensPerPass() {
            return tokens.length / (double) passes;
        }

        /** @return target tokens read per output token */
        public double verifiedPerToken() {
            return verified / (double) tokens.length;
        }
    }

    private static final int VOCAB = 48;
    private static final int TASK = 1;
    private static final int THINK = 2;
    private static final int CALL = 3;
    private static final int ARG = 4;
    private static final int OBS = 5;
    private static final int END = 6;
    private static final int DONE = 7;

    private SuffixDecoding() {
    }

    private static int word(int x, int salt) {
        return 16 + Math.floorMod(x * 7 + salt * 13 + 3, 32);
    }

    /**
     * One agent trace for a task of three words.
     *
     * @param r the random source
     * @return {@code {prompt, trace}}
     */
    static int[][] trace(SplittableRandom r) {
        int[] prompt = {TASK, 16 + r.nextInt(32), 16 + r.nextInt(32), 16 + r.nextInt(32)};
        List<Integer> t = new ArrayList<>();
        for (int i = 1; i < 4; i++) {
            int x = prompt[i];
            t.addAll(List.of(THINK, word(x, 1), word(x, 2), CALL, 8 + x % 8, ARG, x, END, OBS, word(x, 3),
                    word(x, 4), word(x, 5), END));
        }
        t.add(DONE);
        return new int[][] {prompt, t.stream().mapToInt(Integer::intValue).toArray()};
    }

    /**
     * Runs every method.
     *
     * @param corpus traces the target is trained on
     * @param requests requests served in a row
     * @param maxNew tokens per request
     * @param seed the seed
     * @return per method its outcome, greedy first
     */
    public static Map<String, Outcome> run(int corpus, int requests, int maxNew, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[][] train = new int[corpus][];
        for (int i = 0; i < corpus; i++) {
            int[][] pt = trace(r);
            train[i] = concat(pt[0], pt[1]);
        }
        KneserNeyLm target = KneserNeyLm.train(train, 8, VOCAB, 0.75);
        int[][] prompts = new int[requests][];
        for (int i = 0; i < requests; i++) {
            prompts[i] = trace(r)[0];
        }
        Generator gen = new Generator(target, null);
        Map<String, Outcome> out = new LinkedHashMap<>();
        int[] greedy = new int[0];
        for (int[] p : prompts) {
            greedy = concat(greedy, gen.generate(new int[][] {p}, Generator.Options.greedy(maxNew), null).getFirst()
                    .tokens());
        }
        out.put("greedy", new Outcome(greedy, requests * maxNew, (long) requests * maxNew));
        int lookahead = 4;
        int[] pld = new int[0];
        int passes = 0;
        long verified = 0;
        for (int[] p : prompts) {
            Generator.Speculation s = gen.speculative(p, new PromptLookup(VOCAB), lookahead,
                    Generator.Options.greedy(maxNew), null);
            pld = concat(pld, s.output().tokens());
            passes += s.modelSteps();
            verified += (long) (s.modelSteps() - 1) * (lookahead + 1);
        }
        out.put("prompt lookup", new Outcome(pld, passes, verified));
        SuffixTreeDrafter drafter = new SuffixTreeDrafter(VOCAB, 24, 2, 0.1, 16);
        int[] suf = new int[0];
        passes = 0;
        verified = 0;
        for (int[] p : prompts) {
            SuffixTreeDrafter.Result s = drafter.generate(target, p, maxNew);
            suf = concat(suf, s.tokens());
            passes += s.passes();
            verified += s.verified();
        }
        out.put("suffix tree (adaptive)", new Outcome(suf, passes, verified));
        return out;
    }

    private static int[] concat(int[] a, int[] b) {
        int[] c = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, c, a.length, b.length);
        return c;
    }
}
