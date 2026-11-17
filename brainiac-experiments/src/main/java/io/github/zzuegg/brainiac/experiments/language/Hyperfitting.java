package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.experimental.llm.DecodingPipeline;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.llm.LogitsProcessors;
import io.github.zzuegg.brainiac.experimental.llm.TokenSampler;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Hyperfitting: fine-tuning a pretrained language model to near-zero training loss on a small
 * set of sequences. The paper found that greedy decoding from such a model stops falling into
 * repetition loops and produces more varied text (as varied as sampling, rated better by
 * humans), even though its validation perplexity is much worse; the model's predictions become
 * very sharp, but on context, not on memorized continuations.
 *
 * <p>Here a {@link HookedDecoder} is pretrained on a noisy Markov chain ("text": each token has
 * a few likely successors and a long tail), on which greedy decoding of the base model locks
 * into short cycles. It is then hyperfitted on 32 sequences. Compared on continuations of fresh
 * prompts: the base model under greedy decoding and under nucleus sampling (top-p 0.9), and the
 * hyperfitted model under greedy decoding, by repetition (the share of repeated 3-grams within
 * a continuation), diversity (distinct 2-grams over all continuations), the share of
 * continuation 8-grams copied from the fine-tuning set, and validation loss.
 *
 * <p>Carlsson et al., "The Hyperfitting Phenomenon: Sharpening and Stabilizing LLMs for
 * Open-Ended Text Generation" (ICLR 2025), <a href="https://arxiv.org/abs/2412.04318">arXiv:2412.04318</a>.
 */
public final class Hyperfitting {

    /**
     * One decoder and decoding rule.
     *
     * @param name the configuration
     * @param repetition the mean share of repeated 3-grams within a continuation
     * @param distinct2 distinct 2-grams over all 2-grams of all continuations
     * @param copied the share of continuation 8-grams that occur in the fine-tuning set
     * @param validationLoss the model's validation loss
     * @param trainLoss the model's loss on the fine-tuning set
     */
    public record Outcome(String name, double repetition, double distinct2, double copied, double validationLoss,
            double trainLoss) {
    }

    private static final int VOCAB = 24;
    private static final int LENGTH = 32;
    private static final int PROMPT = 4;

    private Hyperfitting() {
    }

    /**
     * Runs the comparison.
     *
     * @param pretrain pretraining steps
     * @param hyperfit hyperfitting epochs over the small set (one batch each)
     * @param seed the seed
     * @return base greedy, base nucleus, hyperfitted greedy
     */
    public static List<Outcome> run(int pretrain, int hyperfit, long seed) {
        int[][] chain = chain(seed);
        HookedDecoder base = HookedDecoder.create(HookedDecoder.Settings.tiny(VOCAB), seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        Optimizer opt = Optimizer.adam(0.01f);
        for (int s = 0; s < pretrain; s++) {
            base.loss(corpus(chain, r, 16), true).backward();
            opt.step(base.parameters());
            base.zeroGrad();
        }
        int[][] small = corpus(chain, new SplittableRandom(seed + 2), 32);
        HookedDecoder hyper = base.copy();
        Optimizer fine = Optimizer.adam(3e-3f);
        for (int e = 0; e < hyperfit; e++) {
            hyper.loss(small, true).backward();
            fine.step(hyper.parameters());
            hyper.zeroGrad();
        }
        int[][] validation = corpus(chain, new SplittableRandom(seed + 3), 64);
        int[][] prompts = corpus(chain, new SplittableRandom(seed + 4), 24);
        Set<List<Integer>> train8 = ngrams(small, 8, 0);
        List<Outcome> out = new ArrayList<>();
        out.add(evaluate("base greedy", DecodingPipeline.of(base), base, prompts, validation, small, train8, seed));
        out.add(evaluate("base top-p 0.9", DecodingPipeline.of(base).processor(LogitsProcessors.topP(0.9))
                .sampler(TokenSampler.ancestral()), base, prompts, validation, small, train8, seed));
        out.add(evaluate("hyperfitted greedy", DecodingPipeline.of(hyper), hyper, prompts, validation, small, train8,
                seed));
        return out;
    }

    private static Outcome evaluate(String name, DecodingPipeline pipe, HookedDecoder model, int[][] prompts,
            int[][] validation, int[][] small, Set<List<Integer>> train8, long seed) {
        SplittableRandom r = new SplittableRandom(seed + 9);
        double repetition = 0;
        Set<List<Integer>> bigrams = new HashSet<>();
        int total2 = 0;
        int copied = 0;
        int total8 = 0;
        for (int[] p : prompts) {
            int[] prompt = java.util.Arrays.copyOf(p, PROMPT);
            int[] gen = pipe.generate(prompt, LENGTH - PROMPT, r);
            List<List<Integer>> tri = list(gen, 3);
            repetition += 1 - new HashSet<>(tri).size() / (double) tri.size();
            List<List<Integer>> bi = list(gen, 2);
            bigrams.addAll(bi);
            total2 += bi.size();
            for (List<Integer> g : list(gen, 8)) {
                copied += train8.contains(g) ? 1 : 0;
                total8++;
            }
        }
        double loss = model.loss(validation, false).value().item();
        return new Outcome(name, repetition / prompts.length, bigrams.size() / (double) total2,
                copied / (double) total8, loss, model.loss(small, false).value().item());
    }

    private static List<List<Integer>> list(int[] tokens, int n) {
        List<List<Integer>> out = new ArrayList<>();
        for (int i = 0; i + n <= tokens.length; i++) {
            List<Integer> g = new ArrayList<>(n);
            for (int k = 0; k < n; k++) {
                g.add(tokens[i + k]);
            }
            out.add(g);
        }
        return out;
    }

    private static Set<List<Integer>> ngrams(int[][] sequences, int n, int from) {
        Set<List<Integer>> out = new HashSet<>();
        for (int[] s : sequences) {
            out.addAll(list(java.util.Arrays.copyOfRange(s, from, s.length), n));
        }
        return out;
    }

    /** Successor lists: three likely successors per token. */
    private static int[][] chain(long seed) {
        SplittableRandom r = new SplittableRandom(seed + 17);
        int[][] next = new int[VOCAB][3];
        for (int v = 0; v < VOCAB; v++) {
            for (int k = 0; k < 3; k++) {
                next[v][k] = r.nextInt(VOCAB);
            }
        }
        return next;
    }

    /** Markov "text": 40%, 25%, 15% for the three successors, otherwise a uniform draw. */
    private static int[][] corpus(int[][] chain, SplittableRandom r, int batch) {
        int[][] out = new int[batch][LENGTH];
        for (int[] s : out) {
            s[0] = r.nextInt(VOCAB);
            for (int i = 1; i < LENGTH; i++) {
                double u = r.nextDouble();
                int[] n = chain[s[i - 1]];
                s[i] = u < 0.4 ? n[0] : u < 0.65 ? n[1] : u < 0.8 ? n[2] : r.nextInt(VOCAB);
            }
        }
        return out;
    }
}
