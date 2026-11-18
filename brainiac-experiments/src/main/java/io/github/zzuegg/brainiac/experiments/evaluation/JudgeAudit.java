package io.github.zzuegg.brainiac.experiments.evaluation;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.eval.BudgetRankingStudy;
import io.github.zzuegg.brainiac.experimental.eval.JudgeCorrection;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.llm.RewardModel;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * LM-judge biases and their mitigation (9.3.11) and statistical correction of judge-based pass
 * rates (9.3.12), with a toy judge whose biases are planted.
 *
 * <p>Answers are short token sequences: a source-style marker (the judge's own family or
 * another), 3-5 content tokens that are each "good" or "bad", then content-free filler that only
 * adds length. Gold quality is the share of good content tokens. The judge is a
 * {@link RewardModel} over {@code [first SEP second]} trained with a logistic loss on preference
 * labels drawn from a Bradley-Terry model with planted biases:
 * {@code logit = 3 Δquality + 0.5 Δlength + 0.8 [first] + 1.0 Δself}.
 * <ul>
 *   <li>{@link #audit}: the size of each bias (position bias, verdict change from 3 filler
 *       tokens, from switching the style marker to the judge's family), and the Spearman
 *       correlation of six systems' win rates against a baseline with their gold win rates: raw
 *       (each comparison in random order), swap-averaged, and length-controlled
 *       ({@link io.github.zzuegg.brainiac.experimental.eval.JudgeAudit#lengthControlled});</li>
 *   <li>{@link #correction}: pass/fail verdicts of the judge (against a reference answer) on
 *       10,000 outputs and a small gold-labelled calibration set: bias, interval width and
 *       coverage of the raw judge mean, human-only, Rogan–Gladen, PPI and PPI++ over
 *       replications, optionally with the calibration set drawn from a mix of systems with a
 *       different pass rate.</li>
 * </ul>
 */
public final class JudgeAudit {

    private JudgeAudit() {
    }

    static final int PAD = 0;
    static final int SEP = 1;
    static final int SELF = 2;
    static final int OTHER = 3;
    static final int GOOD = 4;
    static final int BAD = 8;
    static final int FILLER = 12;
    static final int VOCAB = 13;
    static final int SLOT = 10;

    /**
     * A system that writes answers.
     *
     * @param quality the chance that a content token is good
     * @param verbosity mean filler tokens per answer
     * @param self whether it writes in the judge's own style
     */
    public record Writer(double quality, double verbosity, boolean self) {
        int[] answer(SplittableRandom r) {
            int[] a = new int[SLOT];
            a[0] = self ? SELF : OTHER;
            int content = 3 + r.nextInt(3);
            for (int i = 0; i < content; i++) {
                a[1 + i] = r.nextDouble() < quality ? GOOD + r.nextInt(4) : BAD + r.nextInt(4);
            }
            int filler = Math.min(SLOT - 1 - content, (int) Math.floor(verbosity + r.nextDouble()));
            for (int i = 0; i < filler; i++) {
                a[1 + content + i] = FILLER;
            }
            return a;
        }
    }

    /**
     * Gold quality: the share of good content tokens.
     *
     * @param a an answer
     * @return quality in [0, 1]
     */
    public static double quality(int[] a) {
        int good = 0;
        int content = 0;
        for (int t : a) {
            if (t >= GOOD && t < FILLER) {
                content++;
                good += t < BAD ? 1 : 0;
            }
        }
        return content == 0 ? 0 : good / (double) content;
    }

    /**
     * Length in non-pad tokens after the marker.
     *
     * @param a an answer
     * @return its length
     */
    public static int length(int[] a) {
        int n = 0;
        for (int i = 1; i < a.length; i++) {
            n += a[i] != PAD ? 1 : 0;
        }
        return n;
    }

    /**
     * Appends content-free filler.
     *
     * @param a an answer
     * @param k tokens to add (as space allows)
     * @return the padded copy
     */
    public static int[] pad(int[] a, int k) {
        int[] b = a.clone();
        for (int i = 1; i < b.length && k > 0; i++) {
            if (b[i] == PAD) {
                b[i] = FILLER;
                k--;
            }
        }
        return b;
    }

    /**
     * Switches the style marker to the judge's own family.
     *
     * @param a an answer
     * @return the copy
     */
    public static int[] asSelf(int[] a) {
        int[] b = a.clone();
        b[0] = SELF;
        return b;
    }

    /** The planted preference logit of first over second. */
    static double plantedLogit(int[] first, int[] second) {
        return 3 * (quality(first) - quality(second)) + 0.5 * (length(first) - length(second)) + 0.8
                + 1.0 * ((first[0] == SELF ? 1 : 0) - (second[0] == SELF ? 1 : 0));
    }

    /** A trained toy judge. */
    public static final class ToyJudge implements io.github.zzuegg.brainiac.experimental.eval.JudgeAudit.PairJudge {
        private final RewardModel model;

        private ToyJudge(RewardModel model) {
            this.model = model;
        }

        private static int[] pair(int[] a, int[] b) {
            int[] s = new int[2 * SLOT + 1];
            System.arraycopy(a, 0, s, 0, SLOT);
            s[SLOT] = SEP;
            System.arraycopy(b, 0, s, SLOT + 1, SLOT);
            return s;
        }

        @Override
        public double prefer(int[] first, int[] second) {
            return preferAll(List.<int[][]>of(new int[][] {first, second}))[0];
        }

        /**
         * Judges many ordered pairs in one batch.
         *
         * @param pairs {@code {first, second}} pairs
         * @return the probability that each first answer is better
         */
        public double[] preferAll(List<int[][]> pairs) {
            double[] out = new double[pairs.size()];
            for (int s = 0; s < pairs.size(); s += 256) {
                int e = Math.min(pairs.size(), s + 256);
                int[][] seq = new int[e - s][];
                for (int i = s; i < e; i++) {
                    seq[i - s] = pair(pairs.get(i)[0], pairs.get(i)[1]);
                }
                float[] v = Var.noGrad(() -> model.score(seq)).value().toArray();
                for (int i = s; i < e; i++) {
                    out[i] = 1 / (1 + Math.exp(-v[i - s]));
                }
            }
            return out;
        }

        /**
         * Trains a judge on preference labels with the planted biases.
         *
         * @param steps Adam steps of 32 pairs
         * @param seed the seed
         * @return the judge
         */
        public static ToyJudge train(int steps, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            RewardModel rm = new RewardModel(HookedDecoder.create(HookedDecoder.Settings.tiny(VOCAB).withLayers(1)
                    .withDim(16, 2), seed + 1));
            Optimizer opt = Optimizer.adam(3e-3f);
            for (int s = 0; s < steps; s++) {
                int[][] seq = new int[32][];
                float[] y = new float[32];
                for (int i = 0; i < 32; i++) {
                    int[] a = randomSystem(r).answer(r);
                    int[] b = randomSystem(r).answer(r);
                    seq[i] = pair(a, b);
                    y[i] = r.nextDouble() < 1 / (1 + Math.exp(-plantedLogit(a, b))) ? 1 : 0;
                }
                Var score = rm.score(seq);
                score.softplus().sub(score.mul(Var.of(Tensor.of(Shape.of(32), y)))).mean().backward();
                opt.step(rm.parameters());
            }
            return new ToyJudge(rm);
        }
    }

    private static Writer randomSystem(SplittableRandom r) {
        return new Writer(0.15 + 0.7 * r.nextDouble(), 4 * r.nextDouble(), r.nextBoolean());
    }

    /** The six systems compared with {@link #BASELINE}: quality falls as verbosity rises. */
    public static final List<Writer> SYSTEMS = List.of(new Writer(0.35, 4, false), new Writer(0.45, 3, true),
            new Writer(0.55, 2, false), new Writer(0.6, 1, false), new Writer(0.7, 0.5, false),
            new Writer(0.8, 0, false));

    /** The reference system of the win rates. */
    public static final Writer BASELINE = new Writer(0.55, 1.5, false);

    /**
     * Bias sizes and ranking correlations.
     *
     * @param positionBias mean P(first wins) minus 1/2
     * @param inconsistency share of pairs whose verdict flips with the order
     * @param verbosityEffect swap-averaged verdict change from 3 filler tokens
     * @param selfEffect swap-averaged verdict change from the judge's own style marker
     * @param plantedVerbosity the same change under the planted preference model
     * @param spearmanRaw rank correlation of raw win rates with gold
     * @param spearmanSwap of swap-averaged win rates
     * @param spearmanLengthControlled of length-controlled swap-averaged win rates
     * @param gold the gold win rates
     * @param raw the raw win rates
     * @param lengthControlled the length-controlled win rates
     */
    public record Audit(double positionBias, double inconsistency, double verbosityEffect, double selfEffect,
            double plantedVerbosity, double spearmanRaw, double spearmanSwap, double spearmanLengthControlled,
            double[] gold, double[] raw, double[] lengthControlled) {
        @Override
        public String toString() {
            return String.format("position bias %+.3f (inconsistent %.2f), verbosity %+.3f (planted %+.3f), self %+.3f; "
                    + "Spearman with gold: raw %.2f, swap-averaged %.2f, length-controlled %.2f", positionBias,
                    inconsistency, verbosityEffect, plantedVerbosity, selfEffect, spearmanRaw, spearmanSwap,
                    spearmanLengthControlled);
        }
    }

    /**
     * Probes the judge and ranks the systems.
     *
     * @param judge the judge
     * @param comparisons comparisons per system
     * @param seed the seed
     * @return the audit
     */
    public static Audit audit(ToyJudge judge, int comparisons, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        io.github.zzuegg.brainiac.experimental.eval.JudgeAudit.PairJudge j = judge;
        List<int[][]> probes = new ArrayList<>();
        for (int i = 0; i < 300; i++) {
            Writer s = new Writer(0.5, 0.5, false);
            probes.add(new int[][] {s.answer(r), s.answer(r)});
        }
        double position = io.github.zzuegg.brainiac.experimental.eval.JudgeAudit.positionBias(j, probes);
        double inconsistent = io.github.zzuegg.brainiac.experimental.eval.JudgeAudit.inconsistency(j, probes);
        double verbosity = io.github.zzuegg.brainiac.experimental.eval.JudgeAudit.probe(j, probes, a -> pad(a, 3));
        double self = io.github.zzuegg.brainiac.experimental.eval.JudgeAudit.probe(j, probes, JudgeAudit::asSelf);
        io.github.zzuegg.brainiac.experimental.eval.JudgeAudit.PairJudge planted = (a, b) -> 1
                / (1 + Math.exp(-plantedLogit(a, b)));
        double plantedVerbosity = io.github.zzuegg.brainiac.experimental.eval.JudgeAudit.probe(planted, probes,
                a -> pad(a, 3));
        int k = SYSTEMS.size();
        int n = k * comparisons;
        List<int[][]> ordered = new ArrayList<>();
        List<int[][]> swapped = new ArrayList<>();
        int[] system = new int[n];
        boolean[] systemFirst = new boolean[n];
        double[] lengthDiff = new double[n];
        double[] gold = new double[k];
        for (int s = 0; s < k; s++) {
            for (int c = 0; c < comparisons; c++) {
                int i = s * comparisons + c;
                int[] a = SYSTEMS.get(s).answer(r);
                int[] b = BASELINE.answer(r);
                system[i] = s;
                systemFirst[i] = r.nextBoolean();
                lengthDiff[i] = length(a) - length(b);
                double qa = quality(a);
                double qb = quality(b);
                gold[s] += (qa > qb ? 1 : qa == qb ? 0.5 : 0) / comparisons;
                ordered.add(new int[][] {a, b});
                swapped.add(new int[][] {b, a});
            }
        }
        double[] ab = judge.preferAll(ordered);
        double[] ba = judge.preferAll(swapped);
        double[] raw = new double[n];
        double[] sym = new double[n];
        for (int i = 0; i < n; i++) {
            raw[i] = systemFirst[i] ? ab[i] : 1 - ba[i];
            sym[i] = 0.5 * (ab[i] + 1 - ba[i]);
        }
        double[] rawRates = io.github.zzuegg.brainiac.experimental.eval.JudgeAudit.winRates(raw, system, k);
        double[] symRates = io.github.zzuegg.brainiac.experimental.eval.JudgeAudit.winRates(sym, system, k);
        double[] lc = Arrays.copyOf(io.github.zzuegg.brainiac.experimental.eval.JudgeAudit.lengthControlled(sym,
                system, lengthDiff, k), k);
        return new Audit(position, inconsistent, verbosity, self, plantedVerbosity,
                BudgetRankingStudy.spearman(rawRates, gold), BudgetRankingStudy.spearman(symRates, gold),
                BudgetRankingStudy.spearman(lc, gold), gold, rawRates, lc);
    }

    /**
     * Judge verdicts and gold labels of a pool of outputs.
     *
     * @param judged the judge's 0/1 pass verdicts
     * @param gold the gold 0/1 labels
     */
    public record Labelled(double[] judged, double[] gold) {
        /** @return the gold pass rate */
        public double truth() {
            return Arrays.stream(gold).average().orElse(0);
        }
    }

    private static final int[] REFERENCE = {OTHER, GOOD, GOOD, BAD, BAD, PAD, PAD, PAD, PAD, PAD};

    /**
     * Pass/fail verdicts: an output passes when the judge (swap-averaged) prefers it to a fixed
     * reference answer with half its content good; gold passes when more than half is good.
     *
     * @param judge the judge
     * @param systems the systems the outputs come from, uniformly
     * @param n outputs
     * @param seed the seed
     * @return verdicts and gold labels
     */
    public static Labelled passVerdicts(ToyJudge judge, List<Writer> systems, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<int[][]> ab = new ArrayList<>();
        List<int[][]> ba = new ArrayList<>();
        double[] gold = new double[n];
        for (int i = 0; i < n; i++) {
            int[] a = systems.get(r.nextInt(systems.size())).answer(r);
            gold[i] = quality(a) > 0.5 ? 1 : 0;
            ab.add(new int[][] {a, REFERENCE});
            ba.add(new int[][] {REFERENCE, a});
        }
        double[] p1 = judge.preferAll(ab);
        double[] p2 = judge.preferAll(ba);
        double[] judged = new double[n];
        for (int i = 0; i < n; i++) {
            judged[i] = 0.5 * (p1[i] + 1 - p2[i]) > 0.5 ? 1 : 0;
        }
        return new Labelled(judged, gold);
    }

    /**
     * A judge with fixed sensitivity and specificity on outputs with a given pass rate.
     *
     * @param passRate the true pass rate
     * @param sensitivity P(judge passes | gold pass)
     * @param specificity P(judge fails | gold fail)
     * @param n outputs
     * @param seed the seed
     * @return verdicts and gold labels
     */
    public static Labelled parametric(double passRate, double sensitivity, double specificity, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] gold = new double[n];
        double[] judged = new double[n];
        for (int i = 0; i < n; i++) {
            gold[i] = r.nextDouble() < passRate ? 1 : 0;
            judged[i] = gold[i] > 0.5 ? (r.nextDouble() < sensitivity ? 1 : 0) : (r.nextDouble() < specificity ? 0 : 1);
        }
        return new Labelled(judged, gold);
    }

    /**
     * Bias, interval width and coverage of one estimator.
     *
     * @param estimator the estimator
     * @param bias mean estimate minus truth
     * @param width mean 95% interval width
     * @param coverage share of intervals covering the truth
     */
    public record Correction(String estimator, double bias, double width, double coverage) {
        @Override
        public String toString() {
            return String.format("%s: bias %+.4f, width %.4f, coverage %.2f", estimator, bias, width, coverage);
        }
    }

    /**
     * Repeatedly draws a gold-labelled calibration set and estimates the test set's pass rate.
     *
     * @param test the outputs whose pass rate is wanted (judge verdicts used; gold only as truth)
     * @param calibration the pool calibration labels are drawn from
     * @param labels calibration labels per replication
     * @param replications replications
     * @param seed the seed
     * @return one row per estimator: judge only, human only, Rogan-Gladen, PPI, PPI++
     */
    public static List<Correction> correction(Labelled test, Labelled calibration, int labels, int replications,
            long seed) {
        double truth = test.truth();
        SplittableRandom r = new SplittableRandom(seed);
        String[] names = {"judge only", "human only", "Rogan-Gladen", "PPI", "PPI++"};
        double[][] acc = new double[names.length][3];
        int pool = calibration.gold().length;
        for (int k = 0; k < replications; k++) {
            double[] fl = new double[labels];
            double[] y = new double[labels];
            for (int i = 0; i < labels; i++) {
                int j = r.nextInt(pool);
                fl[i] = calibration.judged()[j];
                y[i] = calibration.gold()[j];
            }
            JudgeCorrection.Estimate[] e = {JudgeCorrection.judgeOnly(test.judged(), 1.96),
                JudgeCorrection.humanOnly(y, 1.96), JudgeCorrection.roganGladen(test.judged(), fl, y, 1.96),
                JudgeCorrection.ppi(test.judged(), fl, y, 1.96), JudgeCorrection.ppiPlusPlus(test.judged(), fl, y, 1.96)};
            for (int m = 0; m < e.length; m++) {
                acc[m][0] += (e[m].estimate() - truth) / replications;
                acc[m][1] += e[m].width() / replications;
                acc[m][2] += (e[m].covers(truth) ? 1.0 : 0) / replications;
            }
        }
        List<Correction> out = new ArrayList<>();
        for (int m = 0; m < names.length; m++) {
            out.add(new Correction(names[m], acc[m][0], acc[m][1], acc[m][2]));
        }
        return out;
    }
}
