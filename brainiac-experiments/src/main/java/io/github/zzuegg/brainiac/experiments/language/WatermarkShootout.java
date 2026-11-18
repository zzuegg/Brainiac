package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticCorpus;
import io.github.zzuegg.brainiac.experimental.llm.GreenListWatermark;
import io.github.zzuegg.brainiac.experimental.llm.LogitMath;
import io.github.zzuegg.brainiac.experimental.llm.LogitsProcessor;
import io.github.zzuegg.brainiac.experimental.llm.TournamentWatermark;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * 10.2.17: the SynthID-Text tournament watermark ({@link TournamentWatermark}) against the
 * Kirchenbauer green list ({@link GreenListWatermark}) at {@code delta} 1 and 2 and against
 * unwatermarked text, on generations of a {@link WindowLm} trained on {@link SyntheticCorpus}
 * documents.
 *
 * <p>Detection is scored as the true-positive rate at a 1% false-positive rate, with the
 * threshold taken empirically from each detector's scores on unwatermarked texts of the same
 * length (the green list's z-test is not calibrated on short repetitive text). Quality is the
 * mean per-token log-likelihood under a separately trained judge LM and the distinct-2 share;
 * robustness is detection after random token substitutions and deletions.
 */
public final class WatermarkShootout {

    /** The generators compared. */
    public enum Scheme {
        /** No watermark. */
        NONE,
        /** Green list, {@code gamma = 0.25}, {@code delta = 1}. */
        GREEN_1,
        /** Green list, {@code gamma = 0.25}, {@code delta = 2}. */
        GREEN_2,
        /** Tournament with 10 layers and a two-token context. */
        TOURNAMENT
    }

    /**
     * A row of the comparison.
     *
     * @param tprByLength detection TPR at 1% FPR for each scored length
     * @param tprEdited TPR at 1% FPR at the full length after each edit rate
     * @param judgeLogLikelihood mean per-token log-likelihood under the judge LM
     * @param distinct2 the share of distinct bigrams over all generated bigrams
     */
    public record Row(Map<Integer, Double> tprByLength, Map<Double, Double> tprEdited, double judgeLogLikelihood,
            double distinct2) {
    }

    /**
     * The trained generator, its judge and the prompts.
     *
     * @param lm the generator
     * @param judge an LM of another seed and window that scores quality
     * @param prompts the prompt pool, two tokens each
     */
    public record Setup(WindowLm lm, WindowLm judge, List<int[]> prompts) {
    }

    private static final long KEY = 0x5EED_CAFEL;
    private static final int PROMPT = 2;

    private WatermarkShootout() {
    }

    /**
     * Trains the generator and the judge on topic streams: four documents of one topic in a row.
     *
     * @param steps training steps
     * @param seed the seed
     * @return the set-up
     */
    public static Setup setup(int steps, long seed) {
        SyntheticCorpus corpus = SyntheticCorpus.generate(4, 6, 12, 0.5, seed);
        int topics = 4;
        List<List<Integer>> byTopic = new ArrayList<>();
        for (int t = 0; t < topics; t++) {
            byTopic.add(new ArrayList<>());
        }
        for (int d = 0; d < corpus.size(); d++) {
            byTopic.get(corpus.topic(d)).add(d);
        }
        SplittableRandom r = new SplittableRandom(seed);
        List<int[]> streams = new ArrayList<>();
        for (int i = 0; i < 400; i++) {
            List<Integer> docs = byTopic.get(r.nextInt(topics));
            int[] s = new int[4 * SyntheticCorpus.DOC_LENGTH];
            for (int k = 0; k < 4; k++) {
                System.arraycopy(corpus.document(docs.get(r.nextInt(docs.size()))), 0, s,
                        k * SyntheticCorpus.DOC_LENGTH, SyntheticCorpus.DOC_LENGTH);
            }
            streams.add(s);
        }
        int vocab = corpus.vocabulary();
        WindowLm lm = train(WindowLm.of(vocab, 2, 64, seed), streams, steps, seed);
        WindowLm judge = train(WindowLm.of(vocab, 3, 64, seed + 99), streams, steps, seed + 99);
        List<int[]> prompts = new ArrayList<>();
        for (int[] s : streams) {
            prompts.add(Arrays.copyOf(s, PROMPT));
        }
        return new Setup(lm, judge, prompts);
    }

    private static WindowLm train(WindowLm lm, List<int[]> streams, int steps, long seed) {
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            List<int[]> prompts = new ArrayList<>();
            List<int[]> completions = new ArrayList<>();
            for (int i = 0; i < 16; i++) {
                int[] x = streams.get(r.nextInt(streams.size()));
                prompts.add(Arrays.copyOf(x, 1));
                completions.add(Arrays.copyOfRange(x, 1, x.length));
            }
            lm.supervisedStep(prompts, completions, opt);
        }
        return lm;
    }

    /**
     * Samples continuations in a batch, applying the temperature first and then the
     * watermark's logits processor.
     *
     * @param lm the generator
     * @param prompts the prompts
     * @param length tokens to add
     * @param temperature the sampling temperature
     * @param processor the watermark, or {@code null}
     * @param random the sampling source
     * @return prompt plus continuation for each prompt
     */
    public static int[][] generate(WindowLm lm, List<int[]> prompts, int length, double temperature,
            LogitsProcessor processor, SplittableRandom random) {
        int n = prompts.size();
        int[][] seqs = new int[n][];
        for (int i = 0; i < n; i++) {
            seqs[i] = Arrays.copyOf(prompts.get(i), prompts.get(i).length + length);
        }
        int vocab = lm.vocabSize();
        for (int t = 0; t < length; t++) {
            List<int[]> prefixes = new ArrayList<>();
            for (int[] s : seqs) {
                prefixes.add(Arrays.copyOf(s, s.length - length + t));
            }
            float[] all = Var.noGrad(() -> lm.logits(prefixes)).value().toArray();
            for (int i = 0; i < n; i++) {
                float[] row = new float[vocab];
                for (int v = 0; v < vocab; v++) {
                    row[v] = (float) (all[i * vocab + v] / temperature);
                }
                int pos = seqs[i].length - length + t;
                if (processor != null) {
                    processor.process(Arrays.copyOf(seqs[i], pos), row);
                }
                seqs[i][pos] = draw(LogitMath.softmax(row), random);
            }
        }
        return seqs;
    }

    private static int draw(double[] p, SplittableRandom random) {
        double u = random.nextDouble();
        double c = 0;
        for (int i = 0; i < p.length; i++) {
            c += p[i];
            if (u < c) {
                return i;
            }
        }
        return p.length - 1;
    }

    /**
     * Edits a text after its prompt: each token is independently substituted by a random token
     * with probability {@code rate / 2} or deleted with probability {@code rate / 2}.
     *
     * @param text prompt and continuation
     * @param rate the edit rate
     * @param vocab the vocabulary size
     * @param random the edit source
     * @return the edited text
     */
    public static int[] edit(int[] text, double rate, int vocab, SplittableRandom random) {
        int[] out = new int[text.length];
        int k = 0;
        for (int i = 0; i < text.length; i++) {
            double u = i < PROMPT ? 1 : random.nextDouble();
            if (u < rate / 2) {
                continue;
            }
            out[k++] = u < rate ? random.nextInt(vocab) : text[i];
        }
        return Arrays.copyOf(out, k);
    }

    /**
     * Runs the comparison at one temperature.
     *
     * @param setup the trained LMs
     * @param texts watermarked texts per scheme (the unwatermarked null set is twice as large)
     * @param length the generated length
     * @param lengths the prefix lengths at which detection is scored
     * @param editRates the edit rates
     * @param temperature the sampling temperature
     * @param seed the seed
     * @return one row per scheme; the {@code NONE} row's TPRs are the empirical FPRs of the
     *     tournament detector
     */
    public static Map<Scheme, Row> compare(Setup setup, int texts, int length, int[] lengths, double[] editRates,
            double temperature, long seed) {
        GreenListWatermark g1 = new GreenListWatermark(KEY, 0.25, 1f);
        GreenListWatermark g2 = new GreenListWatermark(KEY, 0.25, 2f);
        TournamentWatermark tw = new TournamentWatermark(KEY, 10, 2);
        SplittableRandom r = new SplittableRandom(seed);
        int vocab = setup.lm().vocabSize();
        int[][] nullTexts = generate(setup.lm(), prompts(setup, 2 * texts, r), length, temperature, null, r);
        Map<Scheme, Row> out = new LinkedHashMap<>();
        for (Scheme s : Scheme.values()) {
            LogitsProcessor proc = switch (s) {
                case NONE -> null;
                case GREEN_1 -> g1;
                case GREEN_2 -> g2;
                case TOURNAMENT -> tw;
            };
            ToDoubleFunction<int[]> detector = switch (s) {
                case GREEN_1 -> x -> g1.zScore(Arrays.copyOfRange(x, PROMPT - 1, x.length));
                case GREEN_2 -> x -> g2.zScore(Arrays.copyOfRange(x, PROMPT - 1, x.length));
                case NONE, TOURNAMENT -> x -> tw.weightedMeanScore(x, PROMPT);
            };
            int[][] gen = s == Scheme.NONE
                    ? generate(setup.lm(), prompts(setup, texts, r), length, temperature, null, r)
                    : generate(setup.lm(), prompts(setup, texts, r), length, temperature, proc, r);
            Map<Integer, Double> byLength = new LinkedHashMap<>();
            for (int len : lengths) {
                double thr = threshold(nullTexts, len, detector);
                byLength.put(len, tpr(gen, len, detector, thr));
            }
            Map<Double, Double> edited = new LinkedHashMap<>();
            double thr = threshold(nullTexts, length, detector);
            for (double rate : editRates) {
                SplittableRandom er = new SplittableRandom(seed + Math.round(rate * 1000));
                int[][] e = new int[gen.length][];
                for (int i = 0; i < gen.length; i++) {
                    e[i] = edit(gen[i], rate, vocab, er);
                }
                edited.put(rate, tpr(e, Integer.MAX_VALUE, detector, thr));
            }
            out.put(s, new Row(byLength, edited, judge(setup.judge(), gen), distinct2(gen)));
        }
        return out;
    }

    private static List<int[]> prompts(Setup setup, int n, SplittableRandom r) {
        List<int[]> p = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            p.add(setup.prompts().get(r.nextInt(setup.prompts().size())));
        }
        return p;
    }

    private static double threshold(int[][] nullTexts, int len,
            ToDoubleFunction<int[]> detector) {
        double[] s = new double[nullTexts.length];
        for (int i = 0; i < s.length; i++) {
            s[i] = detector.applyAsDouble(Arrays.copyOf(nullTexts[i], Math.min(nullTexts[i].length, PROMPT + len)));
        }
        Arrays.sort(s);
        return s[(int) Math.ceil(0.99 * s.length) - 1];
    }

    private static double tpr(int[][] texts, int len, ToDoubleFunction<int[]> detector,
            double threshold) {
        int hit = 0;
        for (int[] t : texts) {
            int end = (int) Math.min(t.length, (long) PROMPT + len);
            if (detector.applyAsDouble(Arrays.copyOf(t, end)) > threshold) {
                hit++;
            }
        }
        return hit / (double) texts.length;
    }

    private static double judge(WindowLm judge, int[][] texts) {
        List<int[]> prompts = new ArrayList<>();
        List<int[]> completions = new ArrayList<>();
        for (int[] t : texts) {
            prompts.add(Arrays.copyOf(t, PROMPT));
            completions.add(Arrays.copyOfRange(t, PROMPT, t.length));
        }
        return Var.noGrad(() -> judge.logProbs(prompts, completions)).value().mean().item();
    }

    private static double distinct2(int[][] texts) {
        Set<Long> seen = new HashSet<>();
        int total = 0;
        for (int[] t : texts) {
            for (int i = PROMPT + 1; i < t.length; i++) {
                seen.add(((long) t[i - 1] << 32) | t[i]);
                total++;
            }
        }
        return total == 0 ? 0 : seen.size() / (double) total;
    }

    /**
     * The long-budget run: temperatures 0.5, 0.7 and 1.0, lengths 25 to 200, edits 10–30%.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        Setup setup = setup(1500, 1);
        for (double temp : new double[] {0.5, 0.7, 1.0}) {
            Map<Scheme, Row> rows = compare(setup, 300, 200, new int[] {25, 50, 100, 200},
                    new double[] {0.1, 0.2, 0.3}, temp, 2);
            rows.forEach((s, row) -> System.out.printf("Watermark T=%.1f %-10s TPR@1%%FPR by length %s  edited %s  "
                    + "judge log-lik %.3f  distinct-2 %.3f%n", temp, s, row.tprByLength(), row.tprEdited(),
                    row.judgeLogLikelihood(), row.distinct2()));
        }
    }
}
