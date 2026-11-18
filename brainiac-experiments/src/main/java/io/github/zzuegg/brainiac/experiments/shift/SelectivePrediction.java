package io.github.zzuegg.brainiac.experiments.shift;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShiftSuites;
import io.github.zzuegg.brainiac.experimental.ood.FeatureClassifier;
import io.github.zzuegg.brainiac.experimental.ood.LearningToDefer;
import io.github.zzuegg.brainiac.experimental.ood.SelectiveClassifier;
import io.github.zzuegg.brainiac.experimental.train.ConformalClassifier;
import io.github.zzuegg.brainiac.experimental.train.PopulationEnsemble;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Wave 10 items 10.2.6-10.2.7: selective prediction and learning to defer.
 *
 * <p><b>Selective classification</b> ({@link #selective}): six shapes with heavy pixel noise
 * (0.22) and 15% training label noise; test sets in distribution and under a covariate shift
 * (a textured background). Each method gives a confidence; the risk–coverage curve is scored
 * by its area (AURC) and the risk at 80% coverage. Methods: softmax response (SR),
 * SelectiveNet (coverage 0.8), the gambler's loss (payoff 2.5), likelihood-ratio rejection
 * (Δ-MDS, Δ-KNN, each plus RLog) on the SR model, and the baselines MC dropout (10 masks),
 * a three-member deep ensemble through {@link PopulationEnsemble} (whose
 * {@code predictOrAbstain} thresholds the same uncertainty), and the conformal set size.
 *
 * <p><b>Learning to defer</b> ({@link #defer}): ten noisy digits; a simulated expert is right
 * with probability 0.95 on digits 0-4 and guesses uniformly otherwise. Rules: the softmax
 * surrogate (Mozannar and Sontag), the one-vs-all surrogate (Verma and Nalisnick), a
 * confidence threshold that ignores the expert, and the two-stage rule (classifier confidence
 * against a separately trained expert-correctness model); system accuracy at deferral
 * budgets, at the learned rules' own deferral rate, and the calibration (ECE) of the
 * expert-correctness estimates.
 */
public final class SelectivePrediction {

    /**
     * One method's risk–coverage summary.
     *
     * @param name the method
     * @param accuracy full-coverage accuracy
     * @param aurc area under the risk–coverage curve, in distribution
     * @param riskAt80 risk at 80% coverage, in distribution
     * @param shiftedAurc AURC under covariate shift
     * @param shiftedRiskAt80 risk at 80% coverage under covariate shift
     */
    public record Row(String name, double accuracy, double aurc, double riskAt80, double shiftedAurc,
            double shiftedRiskAt80) {
    }

    /**
     * One deferral rule.
     *
     * @param name the rule
     * @param ownRate the deferral rate the rule picks itself (NaN for budget-only rules)
     * @param ownAccuracy system accuracy at that rate
     * @param accuracyAt budget -> system accuracy, for the budgets asked for
     * @param expertEce ECE of the rule's expert-correctness estimate (NaN if it has none)
     */
    public record DeferRow(String name, double ownRate, double ownAccuracy, Map<Double, Double> accuracyAt,
            double expertEce) {
    }

    private static final double PIXEL_NOISE = 0.22;

    private SelectivePrediction() {
    }

    private record Data(ShiftSuites.Split train, ShiftSuites.Split validation, ShiftSuites.Split test,
            ShiftSuites.Split shifted) {
    }

    private static Data data(long seed) {
        int[] all = {0, 1, 2, 3, 4, 5};
        ShiftSuites.Split train = ShiftSuites.shapes(1500, all, ShiftSuites.Style.PLAIN, PIXEL_NOISE, seed);
        SplittableRandom r = new SplittableRandom(seed + 3);
        int[] noisy = train.y().clone();
        for (int i = 0; i < noisy.length; i++) {
            if (r.nextDouble() < 0.15) {
                noisy[i] = r.nextInt(6);
            }
        }
        ShiftSuites.Split val = ShiftSuites.shapes(500, all, ShiftSuites.Style.PLAIN, PIXEL_NOISE, seed + 1);
        ShiftSuites.Split test = ShiftSuites.shapes(600, all, ShiftSuites.Style.PLAIN, PIXEL_NOISE, seed + 2);
        ShiftSuites.Split shifted = ShiftSuites.shapes(600, all, ShiftSuites.Style.TEXTURED, PIXEL_NOISE, seed + 4);
        return new Data(new ShiftSuites.Split(train.x(), noisy), val, test, shifted);
    }

    /**
     * The selective-classification shoot-out.
     *
     * @param steps Adam steps per model
     * @param seed the seed
     * @return one row per method
     */
    public static List<Row> selective(int steps, long seed) {
        Data d = data(seed);
        List<Row> rows = new ArrayList<>();
        // softmax response
        FeatureClassifier sr = ShiftHarness.mlp(ShiftSuites.PIXELS, 6, seed);
        ShiftHarness.train(sr, d.train(), steps, 3e-3f, seed);
        rows.add(row("softmax response", d, x -> SelectiveClassifier.softmaxResponse(sr.logits(x)),
                x -> sr.predict(x)));
        // likelihood ratio on the SR model, fitted on the validation set
        Tensor valLogits = sr.logits(d.validation().x());
        SelectiveClassifier.LikelihoodRatio lr = SelectiveClassifier.LikelihoodRatio.fit(sr.features(d.validation().x()),
                d.validation().y(), valLogits.argmax(1).toIntArray(), 6, 25);
        double[] refMds = lr.deltaMahalanobis(sr.features(d.validation().x()));
        double[] refKnn = lr.deltaKnn(sr.features(d.validation().x()));
        double[] refRlog = SelectiveClassifier.relativeLogit(valLogits);
        rows.add(row("LR Delta-MDS", d, x -> lr.deltaMahalanobis(sr.features(x)), sr::predict));
        rows.add(row("LR Delta-KNN", d, x -> lr.deltaKnn(sr.features(x)), sr::predict));
        rows.add(row("LR Delta-MDS+RLog", d, x -> SelectiveClassifier.LikelihoodRatio.combine(
                SelectiveClassifier.relativeLogit(sr.logits(x)), lr.deltaMahalanobis(sr.features(x)), refRlog, refMds),
                sr::predict));
        rows.add(row("LR Delta-KNN+RLog", d, x -> SelectiveClassifier.LikelihoodRatio.combine(
                SelectiveClassifier.relativeLogit(sr.logits(x)), lr.deltaKnn(sr.features(x)), refRlog, refKnn),
                sr::predict));
        // SelectiveNet
        FeatureClassifier body = ShiftHarness.mlp(ShiftSuites.PIXELS, 6, seed);
        SelectiveClassifier.SelectiveNet sn = new SelectiveClassifier.SelectiveNet(64, 6, 0.8f, seed);
        List<Var> params = new ArrayList<>(body.body().parameters());
        params.addAll(sn.parameters());
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            ShiftSuites.Split b = ShiftHarness.batch(d.train(), 64, r);
            sn.loss(body.features(Var.of(b.x()), true), b.ids(), 32f, 0.5f).backward();
            opt.step(params);
        }
        rows.add(row("SelectiveNet", d, x -> Var.noGrad(() -> sn.selection(body.features(Var.of(x), false)).value())
                .toDoubleArray(), x -> Var.noGrad(() -> sn.logits(body.features(Var.of(x), false)).value()).argmax(1)
                .toIntArray()));
        // Deep Gamblers: pretrain with cross-entropy for a third of the steps
        FeatureClassifier gam = ShiftHarness.mlp(ShiftSuites.PIXELS, 7, seed);
        Optimizer go = Optimizer.adam(3e-3f);
        SplittableRandom gr = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            ShiftSuites.Split b = ShiftHarness.batch(d.train(), 64, gr);
            Var logits = gam.logits(Var.of(b.x()), true);
            Var loss = s < steps / 3 ? Functional.crossEntropy(logits.slice(1, 0, 6), b.ids())
                    : SelectiveClassifier.gamblerLoss(logits, b.ids(), 2.5f);
            loss.backward();
            go.step(gam.parameters());
        }
        rows.add(row("Deep Gamblers", d, x -> SelectiveClassifier.gamblerConfidence(gam.logits(x)),
                x -> SelectiveClassifier.gamblerPredict(gam.logits(x))));
        // MC dropout
        Model drop = Model.sequential().input(ShiftSuites.PIXELS).dense(64, Activation.RELU).dropout(0.2f)
                .dense(64, Activation.RELU).dropout(0.2f).dense(6).seed(seed).build();
        Optimizer dop = Optimizer.adam(3e-3f);
        SplittableRandom dr = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            ShiftSuites.Split b = ShiftHarness.batch(d.train(), 64, dr);
            Functional.crossEntropy(drop.forward(Var.of(b.x()), true), b.ids()).backward();
            dop.step(drop.parameters());
        }
        List<Function<float[][], double[][]>> masks = new ArrayList<>();
        for (int k = 0; k < 10; k++) {
            masks.add(x -> ShiftHarness.probabilities(Var.noGrad(() -> drop.forward(Var.of(Tensor.of(x)), true).value())));
        }
        rows.add(ensembleRow("MC dropout (entropy)", d, PopulationEnsemble.ofFunctions(masks)));
        // deep ensemble of three
        List<Function<float[][], double[][]>> members = new ArrayList<>();
        members.add(x -> ShiftHarness.probabilities(sr.logits(Tensor.of(x))));
        for (int k = 1; k < 3; k++) {
            FeatureClassifier m = ShiftHarness.mlp(ShiftSuites.PIXELS, 6, seed + 100L * k);
            ShiftHarness.train(m, d.train(), steps, 3e-3f, seed + 100L * k);
            members.add(x -> ShiftHarness.probabilities(m.logits(Tensor.of(x))));
        }
        rows.add(ensembleRow("deep ensemble x3 (entropy)", d, PopulationEnsemble.ofFunctions(members)));
        // conformal set size, ties broken by softmax response
        ConformalClassifier cc = ConformalClassifier.calibrate(ShiftHarness.probabilities(valLogits),
                d.validation().y(), 0.1, ConformalClassifier.Score.LAC);
        rows.add(row("conformal set size", d, x -> {
            double[][] p = ShiftHarness.probabilities(sr.logits(x));
            double[] msp = SelectiveClassifier.softmaxResponse(sr.logits(x));
            double[] out = new double[p.length];
            for (int i = 0; i < p.length; i++) {
                int size = cc.predictSet(p[i]).length;
                out[i] = -(size == 0 ? 6 : size) + 0.01 * msp[i];
            }
            return out;
        }, sr::predict));
        return rows;
    }

    private static Row row(String name, Data d, Function<Tensor, double[]> confidence, Function<Tensor, int[]> predict) {
        double[] c = confidence.apply(d.test().x());
        int[] p = predict.apply(d.test().x());
        double[] cs = confidence.apply(d.shifted().x());
        int[] ps = predict.apply(d.shifted().x());
        PopulationEnsemble.SelectiveCurve in = curve(c, p, d.test().y());
        PopulationEnsemble.SelectiveCurve sh = curve(cs, ps, d.shifted().y());
        return new Row(name, ShiftHarness.accuracy(p, d.test().y()), in.aurc(), in.riskAt(0.8), sh.aurc(),
                sh.riskAt(0.8));
    }

    private static PopulationEnsemble.SelectiveCurve curve(double[] confidence, int[] predicted, int[] labels) {
        double[] u = new double[confidence.length];
        for (int i = 0; i < u.length; i++) {
            u[i] = -confidence[i];
        }
        return PopulationEnsemble.curve(u, i -> predicted[i] == labels[i]);
    }

    private static Row ensembleRow(String name, Data d, PopulationEnsemble e) {
        float[][] test = floats(d.test().x());
        float[][] shifted = floats(d.shifted().x());
        PopulationEnsemble.SelectiveCurve in = e.selective(test, d.test().y(), PopulationEnsemble.Uncertainty.ENTROPY);
        PopulationEnsemble.SelectiveCurve sh = e.selective(shifted, d.shifted().y(),
                PopulationEnsemble.Uncertainty.ENTROPY);
        int[] abstain = e.predictOrAbstain(test, PopulationEnsemble.Uncertainty.ENTROPY, Double.POSITIVE_INFINITY);
        return new Row(name, ShiftHarness.accuracy(abstain, d.test().y()), in.aurc(), in.riskAt(0.8), sh.aurc(),
                sh.riskAt(0.8));
    }

    private static float[][] floats(Tensor x) {
        int dim = x.shape().dim(1);
        float[] v = x.toArray();
        float[][] out = new float[v.length / dim][dim];
        for (int i = 0; i < out.length; i++) {
            System.arraycopy(v, i * dim, out[i], 0, dim);
        }
        return out;
    }

    // ------------------------------------------------------------------ learning to defer

    private static int[] expert(int[] labels, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[] out = new int[labels.length];
        for (int i = 0; i < labels.length; i++) {
            out[i] = labels[i] < 5 && r.nextDouble() < 0.95 ? labels[i] : r.nextInt(10);
        }
        return out;
    }

    private static Tensor correctness(int[] expert, int[] labels) {
        float[] v = new float[labels.length];
        for (int i = 0; i < v.length; i++) {
            v[i] = expert[i] == labels[i] ? 1 : 0;
        }
        return Tensor.of(Shape.of(v.length), v);
    }

    /**
     * The learning-to-defer shoot-out.
     *
     * @param steps Adam steps per model
     * @param budgets the deferral budgets to report
     * @param seed the seed
     * @return one row per rule, then "classifier alone" and "expert alone"
     */
    public static List<DeferRow> defer(int steps, double[] budgets, long seed) {
        int[] digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        ShiftSuites.Split train = ShiftSuites.digits(2000, digits, ShiftSuites.Font.BLOCK, ShiftSuites.Style.PLAIN, 0.3,
                seed);
        ShiftSuites.Split test = ShiftSuites.digits(1000, digits, ShiftSuites.Font.BLOCK, ShiftSuites.Style.PLAIN, 0.3,
                seed + 1);
        int[] trainExpert = expert(train.y(), seed + 2);
        int[] testExpert = expert(test.y(), seed + 3);
        Tensor trainCorrect = correctness(trainExpert, train.y());
        int[] testCorrect = correctness(testExpert, test.y()).toIntArray();
        List<DeferRow> out = new ArrayList<>();
        // the two learned rules
        for (boolean ova : new boolean[] {false, true}) {
            FeatureClassifier m = ShiftHarness.mlp(ShiftSuites.PIXELS, 11, seed);
            Optimizer opt = Optimizer.adam(3e-3f);
            SplittableRandom r = new SplittableRandom(seed);
            for (int s = 0; s < steps; s++) {
                int[] rows = new int[64];
                for (int i = 0; i < rows.length; i++) {
                    rows[i] = r.nextInt(train.size());
                }
                ShiftSuites.Split b = train.rows(rows);
                Tensor mc = trainCorrect.gather(0, Tensor.of(Shape.of(64), toFloat(rows)));
                Var logits = m.logits(Var.of(b.x()), true);
                (ova ? LearningToDefer.oneVsAllSurrogate(logits, b.ids(), mc)
                        : LearningToDefer.softmaxSurrogate(logits, b.ids(), mc)).backward();
                opt.step(m.parameters());
            }
            Tensor logits = m.logits(test.x());
            int[] pred = LearningToDefer.predict(logits);
            boolean[] own = LearningToDefer.defer(logits);
            double[] margin = LearningToDefer.deferMargin(logits);
            double[] est = ova ? LearningToDefer.expertCorrectnessOneVsAll(logits)
                    : LearningToDefer.expertCorrectnessSoftmax(logits);
            out.add(new DeferRow(ova ? "one-vs-all (Verma-Nalisnick)" : "softmax (Mozannar-Sontag)", rate(own),
                    LearningToDefer.systemAccuracy(pred, testExpert, test.y(), own),
                    atBudgets(margin, pred, testExpert, test.y(), budgets), binaryEce(est, testCorrect)));
        }
        // a plain classifier with confidence thresholding
        FeatureClassifier clf = ShiftHarness.mlp(ShiftSuites.PIXELS, 10, seed);
        ShiftHarness.train(clf, train, steps, 3e-3f, seed);
        Tensor cl = clf.logits(test.x());
        int[] cp = clf.predict(test.x());
        double[] conf = SelectiveClassifier.softmaxResponse(cl);
        double[] lowConf = new double[conf.length];
        for (int i = 0; i < conf.length; i++) {
            lowConf[i] = -conf[i];
        }
        out.add(new DeferRow("confidence threshold", Double.NaN, Double.NaN,
                atBudgets(lowConf, cp, testExpert, test.y(), budgets), Double.NaN));
        // two-stage: a separate expert-correctness model
        Model em = Model.sequential().input(ShiftSuites.PIXELS).dense(64, Activation.RELU).dense(64, Activation.RELU)
                .add(Dense.spec(1, Activation.IDENTITY, Init.XAVIER)).seed(seed + 9).build();
        Optimizer eo = Optimizer.adam(3e-3f);
        SplittableRandom er = new SplittableRandom(seed + 9);
        for (int s = 0; s < steps; s++) {
            float[] rows = new float[64];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = er.nextInt(train.size());
            }
            Tensor idx = Tensor.of(Shape.of(64), rows);
            Var z = em.forward(Var.of(train.x().gather(0, idx)), true).reshape(64);
            Tensor t = trainCorrect.gather(0, idx);
            // binary cross-entropy on logits: softplus(z) - t z
            z.softplus().sub(z.mul(t)).mean().backward();
            eo.step(em.parameters());
        }
        double[] expertProb = Var.noGrad(() -> em.forward(Var.of(test.x()), false).value()).sigmoid().toDoubleArray();
        double[] twoStage = LearningToDefer.twoStage(conf, expertProb);
        boolean[] twoOwn = new boolean[twoStage.length];
        for (int i = 0; i < twoOwn.length; i++) {
            twoOwn[i] = twoStage[i] > 0;
        }
        out.add(new DeferRow("two-stage", rate(twoOwn), LearningToDefer.systemAccuracy(cp, testExpert, test.y(), twoOwn),
                atBudgets(twoStage, cp, testExpert, test.y(), budgets), binaryEce(expertProb, testCorrect)));
        boolean[] none = new boolean[test.size()];
        boolean[] all = new boolean[test.size()];
        java.util.Arrays.fill(all, true);
        out.add(new DeferRow("classifier alone", 0, LearningToDefer.systemAccuracy(cp, testExpert, test.y(), none),
                Map.of(), Double.NaN));
        out.add(new DeferRow("expert alone", 1, LearningToDefer.systemAccuracy(cp, testExpert, test.y(), all),
                Map.of(), Double.NaN));
        return out;
    }

    private static Map<Double, Double> atBudgets(double[] score, int[] pred, int[] expert, int[] labels,
            double[] budgets) {
        Map<Double, Double> m = new LinkedHashMap<>();
        for (double b : budgets) {
            m.put(b, LearningToDefer.systemAccuracy(pred, expert, labels, LearningToDefer.deferByBudget(score, b)));
        }
        return m;
    }

    private static double rate(boolean[] d) {
        int c = 0;
        for (boolean b : d) {
            c += b ? 1 : 0;
        }
        return c / (double) d.length;
    }

    /** Expected calibration error of binary probability estimates, 10 bins. */
    private static double binaryEce(double[] p, int[] outcome) {
        double[] sum = new double[10];
        double[] hit = new double[10];
        int[] cnt = new int[10];
        for (int i = 0; i < p.length; i++) {
            int b = Math.min(9, (int) (p[i] * 10));
            sum[b] += p[i];
            hit[b] += outcome[i];
            cnt[b]++;
        }
        double ece = 0;
        for (int b = 0; b < 10; b++) {
            if (cnt[b] > 0) {
                ece += Math.abs(sum[b] - hit[b]) / p.length;
            }
        }
        return ece;
    }

    private static float[] toFloat(int[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = v[i];
        }
        return f;
    }

    /**
     * The long-budget run: three seeds at 1,500 steps.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        for (long seed = 1; seed <= 3; seed++) {
            for (Row r : selective(1500, seed)) {
                System.out.printf("seed %d %-28s acc %.3f AURC %.4f risk@80 %.3f | shifted AURC %.4f risk@80 %.3f%n",
                        seed, r.name(), r.accuracy(), r.aurc(), r.riskAt80(), r.shiftedAurc(), r.shiftedRiskAt80());
            }
            for (DeferRow r : defer(1500, new double[] {0.2, 0.4, 0.6}, seed)) {
                System.out.printf("seed %d %-30s own rate %.3f acc %.3f budgets %s ECE %.3f%n", seed, r.name(),
                        r.ownRate(), r.ownAccuracy(), r.accuracyAt(), r.expertEce());
            }
        }
    }
}
