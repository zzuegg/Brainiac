package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.llm.SpeculativeCascade;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Quality against cost for small/large decoding schemes on a branching count: each sequence
 * counts modulo 12 with a stride {@code s} from {1, 2, 3, 5}, but every step adds {@code s}
 * (probability 0.6) or {@code 2s} (0.4), so two next tokens are right and even a perfect model
 * is unsure between them, as language models are between good continuations. A two-block
 * decoder trained well is the large model, a one-block decoder of half the width trained
 * briefly the small one. Quality is the share of generated steps that are valid ({@code s} or
 * {@code 2s}); cost is large passes plus small steps weighted by the small model's parameter
 * share. Plain speculative decoding rejects every valid small-model token that is not the large
 * model's top choice; a speculative cascade keeps it. Compared along their tolerance
 * {@code alpha}: {@link SpeculativeCascade} with the token-specific rule ({@code alpha = 0} is
 * plain greedy speculative decoding), the Diff rule, and the sequential confidence cascade.
 */
public final class CascadeDecoding {

    /**
     * One operating point.
     *
     * @param method the method
     * @param alpha its tolerance
     * @param accuracy the share of correct tokens
     * @param cost large passes plus weighted small steps, per prompt
     * @param largePasses large passes per prompt
     */
    public record Point(String method, double alpha, double accuracy, double cost, double largePasses) {
    }

    private CascadeDecoding() {
    }

    private static final int[] STRIDES = {1, 2, 3, 5};

    /**
     * Runs the sweep.
     *
     * @param prompts prompts of four tokens
     * @param maxNew tokens per prompt
     * @param alphas tolerances
     * @param seed the seed
     * @return every operating point, then the small and large models alone (alpha -1)
     */
    public static List<Point> run(int prompts, int maxNew, double[] alphas, long seed) {
        HookedDecoder large = HookedDecoder.create(HookedDecoder.Settings.tiny(12), seed);
        train(large, 250, seed + 1);
        HookedDecoder small = HookedDecoder.create(HookedDecoder.Settings.tiny(12).withLayers(1).withDim(16, 2),
                seed + 2);
        train(small, 60, seed + 3);
        double smallCost = count(small) / (double) count(large);
        int[][] seqs = branching(new SplittableRandom(seed + 4), prompts, 4 + maxNew);
        List<Point> out = new ArrayList<>();
        for (double a : alphas) {
            out.add(measure("speculative cascade (token)", a, seqs, maxNew, smallCost,
                    p -> SpeculativeCascade.speculative(small, large, p, maxNew, 3, SpeculativeCascade.Rule.TOKEN, a)));
            out.add(measure("speculative cascade (diff)", a, seqs, maxNew, smallCost,
                    p -> SpeculativeCascade.speculative(small, large, p, maxNew, 3, SpeculativeCascade.Rule.DIFF, a)));
            out.add(measure("confidence cascade", a, seqs, maxNew, smallCost,
                    p -> SpeculativeCascade.cascade(small, large, p, maxNew, a)));
        }
        out.add(measure("small alone", -1, seqs, maxNew, smallCost,
                p -> SpeculativeCascade.cascade(small, large, p, maxNew, 1)));
        out.add(measure("large alone", -1, seqs, maxNew, smallCost,
                p -> SpeculativeCascade.cascade(small, large, p, maxNew, 0)));
        return out;
    }

    private static Point measure(String name, double alpha, int[][] seqs, int maxNew, double smallCost,
            java.util.function.Function<int[], SpeculativeCascade.Result> run) {
        double correct = 0;
        double cost = 0;
        double passes = 0;
        for (int[] s : seqs) {
            int[] prompt = java.util.Arrays.copyOf(s, 4);
            int stride = s[s.length - 1];   // the generator stores the stride after the text
            SpeculativeCascade.Result r = run.apply(prompt);
            int prev = prompt[3];
            for (int i = 0; i < maxNew; i++) {
                int d = Math.floorMod(r.tokens()[i] - prev, 12);
                correct += d == stride % 12 || d == (2 * stride) % 12 ? 1 : 0;
                prev = r.tokens()[i];
            }
            cost += r.cost(smallCost);
            passes += r.largePasses();
        }
        return new Point(name, alpha, correct / (seqs.length * (double) maxNew), cost / seqs.length,
                passes / seqs.length);
    }

    /**
     * Branching counts; each row ends with one extra entry holding its stride (not a token).
     *
     * @param r the random source
     * @param batch the sequences
     * @param length the tokens per sequence
     * @return {@code [batch][length + 1]}
     */
    static int[][] branching(SplittableRandom r, int batch, int length) {
        int[][] out = new int[batch][length + 1];
        for (int[] row : out) {
            int s = STRIDES[r.nextInt(STRIDES.length)];
            row[0] = r.nextInt(12);
            for (int t = 1; t < length; t++) {
                row[t] = (row[t - 1] + (r.nextDouble() < 0.6 ? s : 2 * s)) % 12;
            }
            row[length] = s;
        }
        return out;
    }

    private static void train(HookedDecoder m, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(0.01f);
        for (int i = 0; i < steps; i++) {
            int[][] b = branching(r, 16, 12);
            int[][] text = new int[16][];
            for (int j = 0; j < 16; j++) {
                text[j] = java.util.Arrays.copyOf(b[j], 12);
            }
            m.loss(text, true).backward();
            opt.step(m.parameters());
            m.zeroGrad();
        }
    }

    private static long count(HookedDecoder m) {
        long n = 0;
        for (var p : m.parameters()) {
            n += p.shape().size();
        }
        return n;
    }
}
