package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.EagleDraft;
import io.github.zzuegg.brainiac.experimental.llm.JacobiDecoding;
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
 * Lossless speculative decoding of one {@link Llama} target, in tokens per target pass:
 * plain greedy decoding (1), Jacobi decoding, the core {@code Generator.speculative} with a
 * one-block draft model distilled on the target's outputs, and {@link EagleDraft} with a chain
 * of drafts and with a draft tree verified in one pass with tree attention. The EAGLE head and
 * the draft model train on the same greedy continuations of the target, for the same steps.
 * Medusa needs the {@code HookedDecoder} and is measured in {@link DecodingShowdown}.
 */
public final class TreeSpeculation {

    /**
     * One method.
     *
     * @param tokens the generated tokens of every prompt, joined
     * @param passes target passes over all prompts
     */
    public record Outcome(int[] tokens, int passes) {
    }

    private TreeSpeculation() {
    }

    /**
     * Runs every method.
     *
     * @param trainPrompts prompts whose greedy continuations train the drafts
     * @param trainSteps training steps for the EAGLE head and the draft model
     * @param testPrompts prompts measured
     * @param maxNew tokens per prompt
     * @param seed the seed
     * @return method name to outcome, greedy first
     */
    public static Map<String, Outcome> run(int trainPrompts, int trainSteps, int testPrompts, int maxNew, long seed) {
        int vocab = 32;
        Llama target = Llama.random(LlamaConfig.small(vocab, 64, 2, 4, 2), seed, Device.cpu(), DType.FLOAT32);
        SplittableRandom r = new SplittableRandom(seed);
        Generator gen = new Generator(target, null);
        int[][] train = continuations(gen, prompts(r, trainPrompts, vocab), 40);
        int[][] test = prompts(r, testPrompts, vocab);
        Map<String, Outcome> out = new LinkedHashMap<>();
        int[] greedy = new int[0];
        for (int[] p : test) {
            greedy = concat(greedy, gen.generate(new int[][] {p}, Generator.Options.greedy(maxNew), null).getFirst()
                    .tokens());
        }
        out.put("greedy", new Outcome(greedy, test.length * maxNew));
        int[] jac = new int[0];
        int calls = 0;
        for (int[] p : test) {
            JacobiDecoding.Result j = JacobiDecoding.generate(target, p, maxNew, 4, JacobiDecoding.Guess.LOOKAHEAD);
            jac = concat(jac, j.tokens());
            calls += j.modelCalls();
        }
        out.put("lookahead (Jacobi + n-grams)", new Outcome(jac, calls));
        Llama draft = Llama.random(LlamaConfig.small(vocab, 32, 1, 4, 2), seed + 1, Device.cpu(), DType.FLOAT32);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < trainSteps; s++) {
            Var loss = draft.loss(train);
            loss.backward();
            opt.step(draft.parameters());
            for (Var v : draft.parameters()) {
                v.zeroGrad();
            }
        }
        int[] spec = new int[0];
        int steps = 0;
        for (int[] p : test) {
            Generator.Speculation sp = gen.speculative(p, draft, 4, Generator.Options.greedy(maxNew), null);
            spec = concat(spec, sp.output().tokens());
            steps += sp.modelSteps();
        }
        out.put("core speculative (distilled draft model)", new Outcome(spec, steps));
        PagedLlama paged = PagedLlama.of(target, 16, 256);
        EagleDraft eagle = new EagleDraft(paged, seed + 2);
        eagle.train(train, trainSteps, Optimizer.adam(3e-3f));
        for (Map.Entry<String, int[]> shape : List.of(Map.entry("eagle chain", new int[] {1, 4, 1}),
                Map.entry("eagle tree", new int[] {2, 4, 3}))) {
            int[] toks = new int[0];
            int passes = 0;
            for (int[] p : test) {
                EagleDraft.Result e = eagle.generate(p, maxNew, shape.getValue()[0], shape.getValue()[1],
                        shape.getValue()[2]);
                toks = concat(toks, e.tokens());
                passes += e.passes();
            }
            out.put(shape.getKey(), new Outcome(toks, passes));
        }
        return out;
    }

    private static int[][] prompts(SplittableRandom r, int n, int vocab) {
        int[][] p = new int[n][];
        for (int i = 0; i < n; i++) {
            p[i] = ServingThroughput.randomTokens(r, 4, vocab);
        }
        return p;
    }

    private static int[][] continuations(Generator gen, int[][] prompts, int n) {
        List<Generator.Output> outs = gen.generate(prompts, Generator.Options.greedy(n), null);
        int[][] seqs = new int[prompts.length][];
        for (int i = 0; i < prompts.length; i++) {
            seqs[i] = concat(prompts[i], outs.get(i).tokens());
        }
        return seqs;
    }

    private static int[] concat(int[] a, int[] b) {
        int[] c = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, c, a.length, b.length);
        return c;
    }
}
