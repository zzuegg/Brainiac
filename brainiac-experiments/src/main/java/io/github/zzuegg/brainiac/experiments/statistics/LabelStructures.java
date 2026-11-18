package io.github.zzuegg.brainiac.experiments.statistics;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.calibrate.Recalibrators;
import io.github.zzuegg.brainiac.experimental.data.SparseDesigns;
import io.github.zzuegg.brainiac.experimental.ml.RandomForest;
import io.github.zzuegg.brainiac.experimental.stats.FeatureSelection;
import io.github.zzuegg.brainiac.experimental.stats.MultiLabel;
import io.github.zzuegg.brainiac.experimental.stats.OrdinalRegression;
import io.github.zzuegg.brainiac.experimental.stats.PenalizedRegression;
import io.github.zzuegg.brainiac.experimental.stats.Resampling;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Structured and skewed labels (items 12.1.10-12.1.13): ordinal heads (proportional odds, CORAL,
 * CORN) against softmax, rounded regression and independent binary heads on unequally spaced
 * grades; multi-label methods (binary relevance, chains with greedy and beam decoding, chain
 * ensembles, label powerset, ML-kNN) against independent sigmoid heads on a task with a planted
 * AND dependency; imbalance resampling (SMOTE variants, cleaning, undersampling, EasyEnsemble)
 * against class weights and post-hoc logit adjustment, scored on ranking, minority F1 and
 * calibration before and after isotonic recalibration; and feature selection (mRMR, ReliefF,
 * Boruta, stochastic gates, LassoNet) against the logistic lasso path and forest permutation
 * importance on XOR/sine features with redundant copies.
 */
public final class LabelStructures {

    private LabelStructures() {
    }

    // ------------------------------------------------------------------ ordinal

    /** Grades with unequal spacing: cut-points on a latent score, plus +/-1 label noise. */
    private static int[][] ordinalLabels(double[][] x, int classes, double noise, SplittableRandom r) {
        double[] cuts = {-1.6, -1.0, -0.7, 0.2, 1.5};
        int[] y = new int[x.length];
        for (int i = 0; i < x.length; i++) {
            double z = 0.8 * x[i][0] + 0.6 * x[i][1] - 0.5 * x[i][2] + 0.4 * Math.sin(2 * x[i][3]) + 0.3 * r.nextGaussian();
            int c = 0;
            while (c < classes - 1 && z > cuts[c]) {
                c++;
            }
            if (r.nextDouble() < noise) {
                c = Math.max(0, Math.min(classes - 1, c + (r.nextBoolean() ? 1 : -1)));
            }
            y[i] = c;
        }
        return new int[][] {y};
    }

    /**
     * Ordinal heads on one small MLP trunk each.
     *
     * @param n training rows (the test set has 1000)
     * @param steps full-batch Adam steps
     * @param seed the seed
     * @return {@code {test MAE in grades, accuracy, rank-inconsistency rate}} per method
     */
    public static Map<String, double[]> ordinal(int n, int steps, long seed) {
        int classes = 6;
        int p = 8;
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = gaussian(n, p, r);
        int[] y = ordinalLabels(x, classes, 0.1, r)[0];
        double[][] xt = gaussian(1000, p, r);
        int[] yt = ordinalLabels(xt, classes, 0, r)[0];
        Var xs = Var.of(TinyMlp.matrix(x));
        Var xts = Var.of(TinyMlp.matrix(xt));
        Map<String, double[]> out = new LinkedHashMap<>();
        for (OrdinalRegression.Kind kind : OrdinalRegression.Kind.values()) {
            TinyMlp trunk = new TinyMlp(p, 32, seed);
            OrdinalRegression head = new OrdinalRegression(kind, 32, classes, seed + 1);
            List<Var> params = trunk.parameters();
            params.addAll(head.parameters());
            train(params, () -> head.loss(head.forward(trunk.hidden(xs)), y), steps);
            double[][] cum = head.cumulative(Var.noGrad(() -> head.forward(trunk.hidden(xts))).value());
            out.put(kind.name(), ordinalScores(OrdinalRegression.decode(cum), yt, OrdinalRegression.violations(cum)));
        }
        // softmax
        {
            TinyMlp trunk = new TinyMlp(p, 32, seed);
            Var w = TinyMlp.linear(32, classes, seed + 1);
            Var b = Var.param(Tensor.zeros(Shape.of(1, classes)));
            List<Var> params = trunk.parameters();
            params.addAll(List.of(w, b));
            Tensor onehot = oneHot(y, classes);
            train(params, () -> trunk.hidden(xs).matmul(w).add(b).logSoftmax(1).mul(onehot).sum().div(-n), steps);
            double[][] probs = TinyMlp.rows(Var.noGrad(() -> trunk.hidden(xts).matmul(w).add(b).softmax(1)).value());
            int[] pred = new int[probs.length];
            double[][] cum = new double[probs.length][classes - 1];
            for (int i = 0; i < probs.length; i++) {
                for (int c = 1; c < classes; c++) {
                    if (probs[i][c] > probs[i][pred[i]]) {
                        pred[i] = c;
                    }
                }
                double tail = 0;
                for (int c = classes - 1; c >= 1; c--) {
                    tail += probs[i][c];
                    cum[i][c - 1] = tail;
                }
            }
            out.put("softmax", ordinalScores(pred, yt, OrdinalRegression.violations(cum)));
        }
        // MSE regression with rounding
        {
            TinyMlp trunk = new TinyMlp(p, 32, seed);
            Var w = TinyMlp.linear(32, 1, seed + 1);
            Var b = Var.param(Tensor.zeros(Shape.of(1, 1)));
            List<Var> params = trunk.parameters();
            params.addAll(List.of(w, b));
            float[] target = new float[n];
            for (int i = 0; i < n; i++) {
                target[i] = y[i];
            }
            Tensor tt = Tensor.of(Shape.of(n, 1), target);
            train(params, () -> trunk.hidden(xs).matmul(w).add(b).sub(tt).square().mean(), steps);
            float[] pr = Var.noGrad(() -> trunk.hidden(xts).matmul(w).add(b)).value().toArray();
            int[] pred = new int[pr.length];
            for (int i = 0; i < pr.length; i++) {
                pred[i] = Math.max(0, Math.min(classes - 1, Math.round(pr[i])));
            }
            out.put("MSE + rounding", ordinalScores(pred, yt, 0));
        }
        // independent binary heads (unshared, unconditional)
        {
            TinyMlp trunk = new TinyMlp(p, 32, seed);
            Var w = TinyMlp.linear(32, classes - 1, seed + 1);
            Var b = Var.param(Tensor.zeros(Shape.of(1, classes - 1)));
            List<Var> params = trunk.parameters();
            params.addAll(List.of(w, b));
            train(params, () -> OrdinalRegression.independentTasksLoss(trunk.hidden(xs).matmul(w).add(b), y), steps);
            double[][] z = TinyMlp.rows(Var.noGrad(() -> trunk.hidden(xts).matmul(w).add(b).sigmoid()).value());
            out.put("independent binary", ordinalScores(OrdinalRegression.decode(z), yt, OrdinalRegression.violations(z)));
        }
        return out;
    }

    private static double[] ordinalScores(int[] pred, int[] y, double violations) {
        double mae = 0;
        int acc = 0;
        for (int i = 0; i < y.length; i++) {
            mae += Math.abs(pred[i] - y[i]);
            if (pred[i] == y[i]) {
                acc++;
            }
        }
        return new double[] {mae / y.length, (double) acc / y.length, violations};
    }

    private static void train(List<Var> params, java.util.function.Supplier<Var> loss, int steps) {
        Optimizer opt = Optimizer.adam(0.01f);
        for (int s = 0; s < steps; s++) {
            loss.get().backward();
            opt.step(params);
        }
    }

    private static Tensor oneHot(int[] y, int classes) {
        float[] v = new float[y.length * classes];
        for (int i = 0; i < y.length; i++) {
            v[i * classes + y[i]] = 1;
        }
        return Tensor.of(Shape.of(y.length, classes), v);
    }

    private static double[][] gaussian(int n, int p, SplittableRandom r) {
        double[][] x = new double[n][p];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                x[i][j] = r.nextGaussian();
            }
        }
        return x;
    }

    // ------------------------------------------------------------------ multi-label

    private static int[][] multiLabels(double[][] x, SplittableRandom r) {
        int[][] y = new int[x.length][4];
        for (int i = 0; i < x.length; i++) {
            y[i][0] = x[i][0] + x[i][1] + 0.4 * r.nextGaussian() > 0 ? 1 : 0;
            y[i][1] = x[i][2] - x[i][0] + 0.4 * r.nextGaussian() > 0 ? 1 : 0;
            int and = y[i][0] & y[i][1];
            y[i][2] = r.nextDouble() < 0.05 ? 1 - and : and;
            y[i][3] = x[i][3] + 0.5 * x[i][4] + 0.4 * r.nextGaussian() > 0 ? 1 : 0;
        }
        return y;
    }

    /**
     * Multi-label methods on a tabular task where label 2 is (noisily) label 0 AND label 1.
     *
     * @param n training rows (the test set has 1000)
     * @param seed the seed
     * @return {@code {Hamming loss, subset accuracy, micro-F1}} per method
     */
    public static Map<String, double[]> multiLabel(int n, long seed) {
        int p = 8;
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = gaussian(n, p, r);
        int[][] y = multiLabels(x, r);
        double[][] xt = gaussian(1000, p, r);
        int[][] yt = multiLabels(xt, r);
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("binary relevance", MultiLabel.scores(
                MultiLabel.fit(MultiLabel.Method.BINARY_RELEVANCE, x, y, 1, seed).predict(xt), yt));
        out.put("classifier chain", MultiLabel.scores(
                MultiLabel.fit(MultiLabel.Method.CLASSIFIER_CHAIN, x, y, 1, seed).predict(xt), yt));
        out.put("chain beam 4", MultiLabel.scores(
                MultiLabel.fit(MultiLabel.Method.CLASSIFIER_CHAIN, x, y, 1, seed).beamWidth(4).predict(xt), yt));
        out.put("chain ensemble x5", MultiLabel.scores(
                MultiLabel.fit(MultiLabel.Method.ENSEMBLE_CHAINS, x, y, 5, seed).predict(xt), yt));
        out.put("label powerset", MultiLabel.scores(
                MultiLabel.fit(MultiLabel.Method.LABEL_POWERSET, x, y, 1, seed).predict(xt), yt));
        out.put("ML-kNN k=10", MultiLabel.scores(MultiLabel.fit(MultiLabel.Method.ML_KNN, x, y, 10, seed).predict(xt),
                yt));
        // independent sigmoid heads on one MLP
        TinyMlp trunk = new TinyMlp(p, 32, seed);
        Var w = TinyMlp.linear(32, 4, seed + 1);
        Var b = Var.param(Tensor.zeros(Shape.of(1, 4)));
        List<Var> params = trunk.parameters();
        params.addAll(List.of(w, b));
        float[] tv = new float[n * 4];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < 4; j++) {
                tv[i * 4 + j] = y[i][j];
            }
        }
        Tensor tt = Tensor.of(Shape.of(n, 4), tv);
        Var xs = Var.of(TinyMlp.matrix(x));
        train(params, () -> {
            Var z = trunk.hidden(xs).matmul(w).add(b);
            return z.softplus().sub(z.mul(tt)).mean();
        }, 300);
        double[][] pr = TinyMlp.rows(Var.noGrad(() -> trunk.hidden(Var.of(TinyMlp.matrix(xt))).matmul(w).add(b))
                .value());
        int[][] pred = new int[pr.length][4];
        for (int i = 0; i < pr.length; i++) {
            for (int j = 0; j < 4; j++) {
                pred[i][j] = pr[i][j] > 0 ? 1 : 0;
            }
        }
        out.put("MLP sigmoid heads", MultiLabel.scores(pred, yt));
        return out;
    }

    // ------------------------------------------------------------------ imbalance

    private static double[][] imbalanced(int n, double prevalence, SplittableRandom r, int[] y) {
        int p = 5;
        double[][] x = new double[n][p];
        for (int i = 0; i < n; i++) {
            y[i] = r.nextDouble() < prevalence ? 1 : 0;
            for (int j = 0; j < p; j++) {
                x[i][j] = r.nextGaussian() + (y[i] == 1 && j < 3 ? 1.2 : 0);
            }
            if (y[i] == 1) {
                x[i][3] += 0.8 * x[i][0] * x[i][1];
            }
        }
        return x;
    }

    /**
     * Imbalance methods with a logistic model.
     *
     * @param ratio majority-to-minority ratio
     * @param n training rows (validation and test sets are the same size)
     * @param seed the seed
     * @return {@code {AUROC, minority F1, balanced accuracy, ECE, ECE after isotonic}} per method
     */
    public static Map<String, double[]> imbalance(int ratio, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double prev = 1.0 / (1 + ratio);
        int[] y = new int[n];
        double[][] x = imbalanced(n, prev, r, y);
        int[] yv = new int[n];
        double[][] xv = imbalanced(n, prev, r, yv);
        int[] yt = new int[n];
        double[][] xt = imbalanced(n, prev, r, yt);
        double truePrior = mean(y);
        Map<String, double[]> out = new LinkedHashMap<>();
        double[] w1 = new double[n];
        Arrays.fill(w1, 1);
        double[] plain = logit(x, y, w1);
        out.put("plain", imbalanceScores(plain, xv, yv, xt, yt, null));
        double[] wc = new double[n];
        for (int i = 0; i < n; i++) {
            wc[i] = y[i] == 1 ? (1 - truePrior) / truePrior : 1;
        }
        out.put("class weights", imbalanceScores(logit(x, y, wc), xv, yv, xt, yt, null));
        double shift = Math.log(truePrior / (1 - truePrior));
        out.put("logit adjustment (post-hoc)", imbalanceScores(plain, xv, yv, xt, yt, z -> 1 / (1 + Math.exp(-(z - shift)))));
        for (Resampling.Method m : Resampling.Method.values()) {
            Resampling.Sample s = Resampling.resample(m, x, y, 5, seed);
            double[] beta = logit(s.x(), s.y(), ones(s.y().length));
            out.put(m.name(), imbalanceScores(beta, xv, yv, xt, yt, null));
            if (m == Resampling.Method.RANDOM_UNDER || m == Resampling.Method.SMOTE) {
                double ps = s.positiveRate();
                out.put(m.name() + " + prior shift", imbalanceScores(beta, xv, yv, xt, yt,
                        z -> Resampling.priorShift(1 / (1 + Math.exp(-z)), ps, truePrior)));
            }
        }
        // EasyEnsemble: average the members' probabilities
        List<double[]> members = new ArrayList<>();
        for (Resampling.Sample s : Resampling.easyEnsemble(x, y, 5, seed)) {
            members.add(logit(s.x(), s.y(), ones(s.y().length)));
        }
        out.put("EasyEnsemble x5", imbalanceScores(row -> {
            double pr = 0;
            for (double[] b : members) {
                pr += sigmoid(dot(b, row)) / members.size();
            }
            return pr;
        }, xv, yv, xt, yt));
        return out;
    }

    private static double[] imbalanceScores(double[] beta, double[][] xv, int[] yv, double[][] xt, int[] yt,
            java.util.function.DoubleUnaryOperator link) {
        return imbalanceScores(row -> link == null ? sigmoid(dot(beta, row)) : link.applyAsDouble(dot(beta, row)), xv,
                yv, xt, yt);
    }

    private static double[] imbalanceScores(java.util.function.ToDoubleFunction<double[]> prob, double[][] xv,
            int[] yv, double[][] xt, int[] yt) {
        double[] pt = new double[xt.length];
        double[] pv = new double[xv.length];
        for (int i = 0; i < xt.length; i++) {
            pt[i] = prob.applyAsDouble(xt[i]);
        }
        for (int i = 0; i < xv.length; i++) {
            pv[i] = prob.applyAsDouble(xv[i]);
        }
        int tp = 0;
        int fp = 0;
        int fn = 0;
        int tn = 0;
        for (int i = 0; i < yt.length; i++) {
            boolean pos = pt[i] > 0.5;
            if (pos && yt[i] == 1) {
                tp++;
            } else if (pos) {
                fp++;
            } else if (yt[i] == 1) {
                fn++;
            } else {
                tn++;
            }
        }
        double f1 = tp == 0 ? 0 : 2.0 * tp / (2.0 * tp + fp + fn);
        double bacc = 0.5 * ((double) tp / Math.max(1, tp + fn) + (double) tn / Math.max(1, tn + fp));
        double[] yvd = new double[yv.length];
        for (int i = 0; i < yv.length; i++) {
            yvd[i] = yv[i];
        }
        double[][] iso = Recalibrators.isotonic(pv, yvd);
        double[] recal = new double[pt.length];
        for (int i = 0; i < pt.length; i++) {
            recal[i] = stepLookup(iso, pt[i]);
        }
        return new double[] {Scores.auc(pt, yt), f1, bacc, Scores.ece(pt, yt, 15), Scores.ece(recal, yt, 15)};
    }

    private static double stepLookup(double[][] iso, double s) {
        double[] xs = iso[0];
        double[] vs = iso[1];
        if (s <= xs[0]) {
            return vs[0];
        }
        for (int k = 1; k < xs.length; k++) {
            if (s < xs[k]) {
                double t = (s - xs[k - 1]) / (xs[k] - xs[k - 1]);
                return vs[k - 1] + t * (vs[k] - vs[k - 1]);
            }
        }
        return vs[vs.length - 1];
    }

    /** Weighted logistic regression by IRLS; the intercept is the last coefficient. */
    private static double[] logit(double[][] x, int[] y, double[] w) {
        int n = x.length;
        int p = x[0].length;
        double[] b = new double[p + 1];
        for (int it = 0; it < 50; it++) {
            double[][] g = new double[p + 1][p + 1];
            double[] grad = new double[p + 1];
            for (int i = 0; i < n; i++) {
                double[] z = Arrays.copyOf(x[i], p + 1);
                z[p] = 1;
                double pr = sigmoid(dot(b, x[i]));
                double h = w[i] * Math.max(pr * (1 - pr), 1e-6);
                for (int a = 0; a <= p; a++) {
                    grad[a] += w[i] * (y[i] - pr) * z[a];
                    for (int c = 0; c <= p; c++) {
                        g[a][c] += h * z[a] * z[c];
                    }
                }
            }
            for (int a = 0; a <= p; a++) {
                g[a][a] += 1e-4;
            }
            double[] step = Recalibrators.solve(g, grad);
            double change = 0;
            for (int a = 0; a <= p; a++) {
                b[a] += step[a];
                change = Math.max(change, Math.abs(step[a]));
            }
            if (change < 1e-8) {
                break;
            }
        }
        return b;
    }

    private static double dot(double[] b, double[] row) {
        double s = b[row.length];
        for (int j = 0; j < row.length; j++) {
            s += b[j] * row[j];
        }
        return s;
    }

    private static double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    private static double[] ones(int n) {
        double[] a = new double[n];
        Arrays.fill(a, 1);
        return a;
    }

    private static double mean(int[] y) {
        double s = 0;
        for (int v : y) {
            s += v;
        }
        return s / y.length;
    }

    // ------------------------------------------------------------------ feature selection

    /**
     * Feature selection on the XOR/sine task with redundant copies.
     *
     * @param n rows
     * @param p features
     * @param seed the seed
     * @return {@code {recall of the 5 relevant features in the top 5, in the top 8, redundant
     *     copies in the top 8, forest accuracy on the top 5, ms}} per method
     */
    public static Map<String, double[]> featureSelection(int n, int p, long seed) {
        SparseDesigns.Selection task = SparseDesigns.nonlinear(n, p, 3, seed);
        SparseDesigns.Selection test = SparseDesigns.nonlinear(1000, p, 3, seed + 1);
        Map<String, double[]> out = new LinkedHashMap<>();
        for (FeatureSelection.Method m : FeatureSelection.Method.values()) {
            long t0 = System.nanoTime();
            double[] s = FeatureSelection.scores(m, task.x(), task.y(), seed);
            long t1 = System.nanoTime();
            out.put(m.name(), selectionScores(s, task, test, (t1 - t0) / 1e6, seed));
        }
        long t0 = System.nanoTime();
        double[] yd = new double[n];
        for (int i = 0; i < n; i++) {
            yd[i] = task.y()[i];
        }
        PenalizedRegression lasso = PenalizedRegression.of(PenalizedRegression.Penalty.LASSO)
                .family(PenalizedRegression.Family.LOGISTIC);
        PenalizedRegression.Path path = lasso.path(task.x(), yd, 30, 0.01);
        double[] entry = new double[p];
        for (PenalizedRegression.Fit f : path.fits()) {
            for (int j = 0; j < p; j++) {
                if (entry[j] == 0 && f.beta()[j] != 0) {
                    entry[j] = f.lambda();
                }
            }
        }
        long t1 = System.nanoTime();
        out.put("lasso path (entry lambda)", selectionScores(entry, task, test, (t1 - t0) / 1e6, seed));
        t0 = System.nanoTime();
        double[] perm = FeatureSelection.permutationImportance(task.x(), task.y(), 100, seed);
        t1 = System.nanoTime();
        out.put("RF permutation importance", selectionScores(perm, task, test, (t1 - t0) / 1e6, seed));
        return out;
    }

    private static double[] selectionScores(double[] scores, SparseDesigns.Selection task,
            SparseDesigns.Selection test, double ms, long seed) {
        int[] top5 = FeatureSelection.top(scores, 5);
        int[] top8 = FeatureSelection.top(scores, 8);
        double r5 = overlap(top5, task.relevant()) / 5.0;
        double r8 = overlap(top8, task.relevant()) / 5.0;
        double red = overlap(top8, task.redundant());
        double[][] xs = columns(task.x(), top5);
        double[][] xts = columns(test.x(), top5);
        RandomForest f = RandomForest.fit(xs, task.y(), 2, new RandomForest.Settings(RandomForest.Mode.RANDOM_FOREST,
                100, 12, 2, 0, seed));
        int[] pred = f.predict(xts);
        int c = 0;
        for (int i = 0; i < pred.length; i++) {
            if (pred[i] == test.y()[i]) {
                c++;
            }
        }
        return new double[] {r5, r8, red, (double) c / pred.length, ms};
    }

    private static int overlap(int[] chosen, int[] set) {
        int c = 0;
        for (int a : chosen) {
            for (int b : set) {
                if (a == b) {
                    c++;
                }
            }
        }
        return c;
    }

    private static double[][] columns(double[][] x, int[] cols) {
        double[][] o = new double[x.length][cols.length];
        for (int i = 0; i < x.length; i++) {
            for (int k = 0; k < cols.length; k++) {
                o[i][k] = x[i][cols[k]];
            }
        }
        return o;
    }
}
