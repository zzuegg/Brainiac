package io.github.zzuegg.brainiac.experiments.evaluation;

import io.github.zzuegg.brainiac.experimental.eval.BudgetRankingStudy;
import io.github.zzuegg.brainiac.experimental.eval.Decontaminator;
import io.github.zzuegg.brainiac.experimental.eval.KernelDivergenceScore;
import io.github.zzuegg.brainiac.experimental.llm.ContaminationDetector;
import io.github.zzuegg.brainiac.experiments.data.ContextLm;
import io.github.zzuegg.brainiac.experiments.data.DataCuration;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Contamination audits (9.3.13) with the {@link ContextLm} toy model.
 *
 * <p>Documents come from a second-order Markov grammar over 16 "meanings"; each meaning is
 * written as one of two synonymous surface tokens chosen at random (32 surface tokens), so a
 * paraphrase of a document keeps its meaning sequence and resamples its surface tokens.
 * <ul>
 *   <li>{@link #setLevel}: a 60-document benchmark planted (three copies each) into 300
 *       pretraining documents at several rates; per rate, the Kernel Divergence Score (mean
 *       hidden activations before and after 60 fine-tuning steps on the benchmark) and the
 *       set-level means of Min-K%++ ({@link ContaminationDetector}), Min-K% and the negative
 *       loss. {@link #spearman} correlates each with the true rate;</li>
 *   <li>{@link #paraphrase}: the benchmark planted verbatim or as paraphrases: the loss
 *       inflation each causes, and the share of planted paraphrases that n-gram overlap (6-grams)
 *       and the embedding nearest-neighbour detector ({@link Decontaminator}) flag, with the
 *       detector's threshold set at the 99th percentile of fresh clean documents' similarity.</li>
 * </ul>
 */
@SuppressWarnings("exports")   // experimental.* gets exported when the lead integrates it
public final class ContaminationAudit {

    private ContaminationAudit() {
    }

    static final int MEANINGS = 16;
    static final int VOCAB = 2 * MEANINGS;
    static final int LENGTH = 24;

    private static int[] surface(int[] meanings, SplittableRandom r) {
        int[] s = new int[meanings.length];
        for (int i = 0; i < s.length; i++) {
            s[i] = meanings[i] + (r.nextBoolean() ? MEANINGS : 0);
        }
        return s;
    }

    private static int[] meanings(int[] surface) {
        int[] m = new int[surface.length];
        for (int i = 0; i < m.length; i++) {
            m[i] = surface[i] % MEANINGS;
        }
        return m;
    }

    /** The shared corpus of one seed. */
    private record Corpus(DataCuration.Grammar grammar, List<int[]> pretrain, List<int[]> benchmark,
            List<int[]> fresh) {
        static Corpus of(long seed) {
            DataCuration.Grammar g = DataCuration.Grammar.random(MEANINGS, seed);
            SplittableRandom r = new SplittableRandom(seed + 1);
            List<int[]> pre = new ArrayList<>();
            for (int[] d : g.documents(300, LENGTH, r)) {
                pre.add(surface(d, r));
            }
            List<int[]> bench = new ArrayList<>();
            for (int[] d : g.documents(60, LENGTH, r)) {
                bench.add(surface(d, r));
            }
            List<int[]> fresh = new ArrayList<>();
            for (int[] d : g.documents(200, LENGTH, r)) {
                fresh.add(surface(d, r));
            }
            return new Corpus(g, pre, bench, fresh);
        }
    }

    private static double[] meanHidden(ContextLm lm, int[] doc) {
        Tensor h = lm.net().activations(ContextLm.examples(VOCAB, List.of(doc))[0])[1];
        int n = h.shape().dim(0);
        int d = h.shape().dim(1);
        float[] v = h.toArray();
        double[] out = new double[d];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < d; k++) {
                out[k] += v[i * d + k] / n;
            }
        }
        return out;
    }

    private static double[] positional(ContextLm lm, int[] doc, double[] centre) {
        float[] v = lm.net().activations(ContextLm.examples(VOCAB, List.of(doc))[0])[1].toArray();
        double[] out = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i] - centre[i];
        }
        return out;
    }

    /**
     * Set-level scores at one contamination rate.
     *
     * @param rate the planted share of the benchmark
     * @param kds the Kernel Divergence Score
     * @param minKPlusPlus mean Min-K%++ over the benchmark
     * @param minK mean Min-K%
     * @param negativeLoss minus the mean loss
     */
    public record SetLevel(double rate, double kds, double minKPlusPlus, double minK, double negativeLoss) {
    }

    /**
     * Set-level scores against the contamination rate.
     *
     * @param rates planted shares of the benchmark
     * @param steps pretraining steps
     * @param seed the seed
     * @return one row per rate
     */
    public static List<SetLevel> setLevel(double[] rates, int steps, long seed) {
        Corpus c = Corpus.of(seed);
        List<SetLevel> out = new ArrayList<>();
        for (double rate : rates) {
            List<int[]> train = new ArrayList<>(c.pretrain());
            int planted = (int) Math.round(rate * c.benchmark().size());
            for (int copy = 0; copy < 3; copy++) {
                train.addAll(c.benchmark().subList(0, planted));
            }
            ContextLm lm = new ContextLm(VOCAB, 64, seed + 2).train(train, steps, 32, 1e-2f, seed + 3);
            double[][] before = new double[c.benchmark().size()][];
            double plus = 0;
            double mink = 0;
            double loss = 0;
            int n = c.benchmark().size();
            for (int i = 0; i < n; i++) {
                int[] d = c.benchmark().get(i);
                before[i] = meanHidden(lm, d);
                Tensor logits = lm.logits(d);
                int[] targets = Arrays.copyOfRange(d, 2, d.length);
                plus += ContaminationDetector.minKPlusPlus(logits, targets, 0.2) / n;
                mink += ContaminationDetector.minK(logits, targets, 0.2) / n;
                loss += ContaminationDetector.logLikelihood(logits, targets) / n;
            }
            lm.train(c.benchmark(), 60, 32, 1e-2f, seed + 4);
            double[][] after = new double[n][];
            for (int i = 0; i < n; i++) {
                after[i] = meanHidden(lm, c.benchmark().get(i));
            }
            out.add(new SetLevel(rate, KernelDivergenceScore.score(before, after), plus, mink, loss));
        }
        return out;
    }

    /**
     * Spearman correlations of each set-level score with the rate.
     *
     * @param rows the rows of {@link #setLevel}
     * @return {@code {KDS, Min-K%++, Min-K%, -loss}}
     */
    public static double[] spearman(List<SetLevel> rows) {
        double[] rate = rows.stream().mapToDouble(SetLevel::rate).toArray();
        return new double[] {BudgetRankingStudy.spearman(rows.stream().mapToDouble(SetLevel::kds).toArray(), rate),
            BudgetRankingStudy.spearman(rows.stream().mapToDouble(SetLevel::minKPlusPlus).toArray(), rate),
            BudgetRankingStudy.spearman(rows.stream().mapToDouble(SetLevel::minK).toArray(), rate),
            BudgetRankingStudy.spearman(rows.stream().mapToDouble(SetLevel::negativeLoss).toArray(), rate)};
    }

    /**
     * Paraphrased contamination.
     *
     * @param cleanLoss benchmark loss of the model trained without contamination
     * @param verbatimLoss with the benchmark planted verbatim
     * @param paraphraseLoss with the benchmark planted as paraphrases
     * @param ngramVerbatim share of verbatim plants flagged by 6-gram overlap
     * @param ngramParaphrase share of paraphrased plants flagged by 6-gram overlap
     * @param embeddingParaphrase share of paraphrased plants flagged by the embedding detector
     * @param embeddingFalsePositive share of clean pretraining documents it flags
     * @param ngramFalsePositive share of clean pretraining documents 6-gram overlap flags
     */
    public record Paraphrase(double cleanLoss, double verbatimLoss, double paraphraseLoss, double ngramVerbatim,
            double ngramParaphrase, double embeddingParaphrase, double embeddingFalsePositive,
            double ngramFalsePositive) {
        @Override
        public String toString() {
            return String.format("benchmark loss clean %.3f, verbatim %.3f, paraphrased %.3f; detected: 6-gram %.2f "
                    + "verbatim / %.2f paraphrased (false positives %.2f), embedding NN %.2f paraphrased (false "
                    + "positives %.2f)", cleanLoss, verbatimLoss, paraphraseLoss, ngramVerbatim, ngramParaphrase,
                    ngramFalsePositive, embeddingParaphrase, embeddingFalsePositive);
        }
    }

    /**
     * Plants the benchmark verbatim or paraphrased (three copies each) and measures loss
     * inflation and detection.
     *
     * @param steps pretraining steps
     * @param seed the seed
     * @return the outcome
     */
    public static Paraphrase paraphrase(int steps, long seed) {
        Corpus c = Corpus.of(seed);
        SplittableRandom r = new SplittableRandom(seed + 5);
        List<int[]> verbatim = new ArrayList<>();
        List<int[]> para = new ArrayList<>();
        for (int copy = 0; copy < 3; copy++) {
            for (int[] d : c.benchmark()) {
                verbatim.add(d);
                para.add(surface(meanings(d), r));
            }
        }
        ContextLm clean = new ContextLm(VOCAB, 64, seed + 2).train(c.pretrain(), steps, 32, 1e-2f, seed + 3);
        List<int[]> withVerbatim = new ArrayList<>(c.pretrain());
        withVerbatim.addAll(verbatim);
        ContextLm v = new ContextLm(VOCAB, 64, seed + 2).train(withVerbatim, steps, 32, 1e-2f, seed + 3);
        List<int[]> withPara = new ArrayList<>(c.pretrain());
        withPara.addAll(para);
        ContextLm p = new ContextLm(VOCAB, 64, seed + 2).train(withPara, steps, 32, 1e-2f, seed + 3);
        int n = c.benchmark().size();
        double[] centre = new double[(LENGTH - 2) * 64];
        for (int[] d : c.benchmark()) {
            double[] e = positional(clean, d, new double[centre.length]);
            for (int i = 0; i < centre.length; i++) {
                centre[i] += e[i] / n;
            }
        }
        Function<int[], double[]> embed = d -> positional(clean, d, centre);
        double[] freshSim = Decontaminator.nearestSimilarity(c.fresh(), c.benchmark(), embed);
        Arrays.sort(freshSim);
        double threshold = freshSim[(int) Math.floor(0.99 * (freshSim.length - 1))];
        boolean[] embPara = Decontaminator.embedding(para, c.benchmark(), embed, threshold);
        boolean[] embClean = Decontaminator.embedding(c.pretrain(), c.benchmark(), embed, threshold);
        boolean[] ngramPara = Decontaminator.ngram(para, c.benchmark(), 6);
        boolean[] ngramVerb = Decontaminator.ngram(verbatim, c.benchmark(), 6);
        boolean[] ngramClean = Decontaminator.ngram(c.pretrain(), c.benchmark(), 6);
        return new Paraphrase(clean.loss(c.benchmark()), v.loss(c.benchmark()), p.loss(c.benchmark()),
                share(ngramVerb), share(ngramPara), share(embPara), share(embClean), share(ngramClean));
    }

    private static double share(boolean[] f) {
        int n = 0;
        for (boolean b : f) {
            n += b ? 1 : 0;
        }
        return n / (double) f.length;
    }
}
