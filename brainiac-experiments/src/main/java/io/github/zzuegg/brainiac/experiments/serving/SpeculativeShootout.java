package io.github.zzuegg.brainiac.experiments.serving;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.experimental.llm.SequenceDistillation;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experimental.serving.MultiDraftVerifier;
import io.github.zzuegg.brainiac.experimental.serving.SpeculativeDraftDistiller;
import io.github.zzuegg.brainiac.experiments.language.DynamicChunking;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Serving shootout, speculative rows (8.4.5 and 8.4.6), on the byte-level toy language of
 * {@link DynamicChunking} (words over {@code a-h} with a Markov chain over words, so word
 * starts are uncertain and word interiors predictable). The target is a {@link WindowLm} with a
 * six-token window; drafts have a two-token window and a quarter of the width.
 * <ul>
 * <li>{@link #distillation}: every draft is first trained on the text, then gets the same number
 *     of further steps by one recipe: more text (the data baseline), offline sequence-level KD on
 *     the target's greedy continuations ({@link SequenceDistillation}), or
 *     {@link SpeculativeDraftDistiller} (DistillSpec) with on-policy TVD or forward KL. Reported:
 *     the expected acceptance rate at T = 1 and greedy, and tokens per target pass with a
 *     four-token draft.</li>
 * <li>{@link #multiDraft}: {@link MultiDraftVerifier} (SpecTr K-SEQ) with k = 1, 2, 4, 8
 *     drafts of four tokens, tokens per target pass; and a distribution test at one position:
 *     total variation between 10^5 verified samples and the target, for K-SEQ and the naive
 *     "first draft that passes" rule.</li>
 * </ul>
 */
public final class SpeculativeShootout {

    private static final String ALPHABET = "abcdefgh ";
    private static final CharacterTokenizer TOK = new CharacterTokenizer(ALPHABET);
    private static final int PROMPT = 6;

    private SpeculativeShootout() {
    }

    /** The trained models. */
    record Models(WindowLm target, WindowLm dataDraft, List<int[]> prompts, String text) {
    }

    private static List<int[][]> windows(String text, SplittableRandom r, int count, int len) {
        List<int[][]> out = new ArrayList<>();
        int[] ids = TOK.encode(text);
        for (int i = 0; i < count; i++) {
            int s = r.nextInt(ids.length - PROMPT - len);
            out.add(new int[][] {java.util.Arrays.copyOfRange(ids, s, s + PROMPT),
                java.util.Arrays.copyOfRange(ids, s + PROMPT, s + PROMPT + len)});
        }
        return out;
    }

    private static void supervised(WindowLm lm, String text, int steps, SplittableRandom r, Optimizer opt) {
        for (int s = 0; s < steps; s++) {
            List<int[]> ps = new ArrayList<>();
            List<int[]> cs = new ArrayList<>();
            for (int[][] w : windows(text, r, 64, 2)) {
                ps.add(w[0]);
                cs.add(w[1]);
            }
            lm.supervisedStep(ps, cs, opt);
        }
    }

    static Models models(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        String text = DynamicChunking.text(r, 20000);
        WindowLm target = WindowLm.of(TOK.size(), PROMPT, 64, seed);
        supervised(target, text, 300, r, Optimizer.adam(1e-2f));
        WindowLm draft = WindowLm.of(TOK.size(), 2, 16, seed + 1);
        supervised(draft, text, 100, r, Optimizer.adam(1e-2f));
        List<int[]> prompts = new ArrayList<>();
        for (int[][] w : windows(text, new SplittableRandom(seed + 9), 40, 1)) {
            prompts.add(w[0]);
        }
        return new Models(target, draft, prompts, text);
    }

    /**
     * One draft recipe.
     *
     * @param acceptanceT1 expected acceptance per drafted token at T = 1
     * @param acceptanceGreedy argmax agreement along greedy drafts
     * @param tokensPerPass tokens per target pass with one four-token draft at T = 1
     */
    public record DraftRow(double acceptanceT1, double acceptanceGreedy, double tokensPerPass) {
    }

    /**
     * Runs the draft-distillation rows.
     *
     * @param steps further steps per recipe
     * @param seeds the seeds
     * @return recipe to row, averaged over seeds
     */
    public static Map<String, DraftRow> distillation(int steps, long[] seeds) {
        Map<String, double[]> sums = new LinkedHashMap<>();
        for (long seed : seeds) {
            Models m = models(seed);
            SplittableRandom r = new SplittableRandom(seed + 3);
            Map<String, WindowLm> drafts = new LinkedHashMap<>();
            WindowLm data = m.dataDraft().copy();
            supervised(data, m.text(), steps, r, Optimizer.adam(3e-3f));
            drafts.put("draft on data", data);
            WindowLm seqKd = m.dataDraft().copy();
            Optimizer kdOpt = Optimizer.adam(3e-3f);
            for (int s = 0; s < steps; s++) {
                List<int[]> ps = new ArrayList<>();
                for (int[][] w : windows(m.text(), r, 64, 1)) {
                    ps.add(w[0]);
                }
                SequenceDistillation.Data d = SequenceDistillation.greedy(m.target(), ps, 2, -1);
                seqKd.supervisedStep(ps, d.targets(), kdOpt);
            }
            drafts.put("SequenceDistillation (offline KD)", seqKd);
            for (SpeculativeDraftDistiller.Divergence div : List.of(SpeculativeDraftDistiller.Divergence.TVD,
                    SpeculativeDraftDistiller.Divergence.FORWARD_KL)) {
                WindowLm d = m.dataDraft().copy();
                SpeculativeDraftDistiller ds = SpeculativeDraftDistiller.of(d, m.target()).divergence(div, 0.5f)
                        .sampling(2, -1, 1.0).optimizer(Optimizer.adam(3e-3f));
                for (int s = 0; s < steps; s++) {
                    List<int[]> ps = new ArrayList<>();
                    for (int[][] w : windows(m.text(), r, 64, 1)) {
                        ps.add(w[0]);
                    }
                    ds.step(ps, null, r);
                }
                drafts.put("DistillSpec on-policy " + (div == SpeculativeDraftDistiller.Divergence.TVD ? "TVD" : "FKL"),
                        d);
            }
            for (Map.Entry<String, WindowLm> e : drafts.entrySet()) {
                WindowLm d = e.getValue();
                double a1 = SpeculativeDraftDistiller.acceptance(d, m.target(), m.prompts(), 4, 1.0,
                        new SplittableRandom(seed + 11));
                double a0 = SpeculativeDraftDistiller.acceptance(d, m.target(), m.prompts(), 4, 0,
                        new SplittableRandom(seed + 11));
                double tpp = tokensPerPass(m.target(), d, m.prompts().subList(0, 10), 1, new SplittableRandom(seed));
                double[] s = sums.computeIfAbsent(e.getKey(), k -> new double[3]);
                s[0] += a1;
                s[1] += a0;
                s[2] += tpp;
            }
        }
        Map<String, DraftRow> out = new LinkedHashMap<>();
        sums.forEach((k, s) -> out.put(k, new DraftRow(s[0] / seeds.length, s[1] / seeds.length,
                s[2] / seeds.length)));
        return out;
    }

    private static Function<int[], double[]> dist(WindowLm lm) {
        return prefix -> SpeculativeDraftDistiller.probabilities(lm.logits(prefix), 1.0);
    }

    private static double tokensPerPass(WindowLm target, WindowLm draft, List<int[]> prompts, int k,
            SplittableRandom r) {
        long tokens = 0;
        long passes = 0;
        for (int[] p : prompts) {
            int[][] g = MultiDraftVerifier.generate(dist(target), dist(draft), p, 24, k, 4, r);
            tokens += g[0].length;
            passes += g[1][0];
        }
        return tokens / (double) passes;
    }

    /**
     * Multi-draft results.
     *
     * @param tokensPerPass tokens per target pass for k = 1, 2, 4, 8 drafts
     * @param tvKSeq total variation of K-SEQ (k = 4) samples from the target, one position
     * @param tvNaive total variation of the naive rule (k = 4)
     * @param tvSampling total variation of plain target samples (the sampling-noise floor)
     */
    public record MultiDraftResult(double[] tokensPerPass, double tvKSeq, double tvNaive, double tvSampling) {
    }

    /**
     * Runs the multi-draft rows with the data-trained draft.
     *
     * @param samples samples for the distribution test
     * @param seeds the seeds
     * @return the result, averaged over seeds
     */
    public static MultiDraftResult multiDraft(int samples, long[] seeds) {
        int[] ks = {1, 2, 4, 8};
        double[] tpp = new double[ks.length];
        double tvK = 0;
        double tvN = 0;
        double tvS = 0;
        for (long seed : seeds) {
            Models m = models(seed);
            for (int i = 0; i < ks.length; i++) {
                tpp[i] += tokensPerPass(m.target(), m.dataDraft(), m.prompts().subList(0, 10), ks[i],
                        new SplittableRandom(seed + i)) / seeds.length;
            }
            // the distribution test at the most uncertain prompt: a word start
            int[] prompt = TOK.encode("ab cd ");
            double[] p = dist(m.target()).apply(prompt);
            double[] q = dist(m.dataDraft()).apply(prompt);
            SplittableRandom r = new SplittableRandom(seed + 5);
            double[] hk = new double[p.length];
            double[] hn = new double[p.length];
            double[] hs = new double[p.length];
            for (int s = 0; s < samples; s++) {
                int[] drafts = new int[4];
                for (int j = 0; j < 4; j++) {
                    drafts[j] = sample(q, r);
                }
                hk[MultiDraftVerifier.verify(p, q, drafts, r).token()]++;
                hn[MultiDraftVerifier.naive(p, q, drafts, r).token()]++;
                hs[sample(p, r)]++;
            }
            tvK += tv(hk, p, samples) / seeds.length;
            tvN += tv(hn, p, samples) / seeds.length;
            tvS += tv(hs, p, samples) / seeds.length;
        }
        return new MultiDraftResult(tpp, tvK, tvN, tvS);
    }

    private static int sample(double[] p, SplittableRandom r) {
        double u = r.nextDouble();
        double c = 0;
        for (int i = 0; i < p.length; i++) {
            c += p[i];
            if (u < c) {
                return i;
            }
        }
        return p.length - 1;
    }

    private static double tv(double[] counts, double[] p, int n) {
        double s = 0;
        for (int i = 0; i < p.length; i++) {
            s += Math.abs(counts[i] / n - p[i]);
        }
        return s / 2;
    }
}
