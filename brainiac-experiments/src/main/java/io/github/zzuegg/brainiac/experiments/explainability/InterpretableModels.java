package io.github.zzuegg.brainiac.experiments.explainability;

import io.github.zzuegg.brainiac.experimental.boost.ExplainableBoosting;
import io.github.zzuegg.brainiac.experimental.data.PlantedFunctions;
import io.github.zzuegg.brainiac.experimental.explain.OptimalTrees;
import io.github.zzuegg.brainiac.experimental.explain.RashomonSet;
import io.github.zzuegg.brainiac.experimental.explain.RuleFit;
import io.github.zzuegg.brainiac.experimental.explain.RuleLists;
import io.github.zzuegg.brainiac.experimental.explain.ScoringSystems;
import io.github.zzuegg.brainiac.experimental.explain.TreeView;
import io.github.zzuegg.brainiac.experimental.stats.PenalizedRegression;
import io.github.zzuegg.brainiac.train.GradientBoostedTrees;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Wave 13 items 13.1.10 to 13.1.13: models that are their own explanation, each against a
 * black-box or greedy baseline.
 *
 * <ul>
 *   <li>{@link #rules}: {@link RuleLists} (CORELS) and {@link RuleFit} against a greedy tree,
 *       core {@link GradientBoostedTrees}, {@link ExplainableBoosting} and an L1 logistic
 *       regression, on {@link PlantedFunctions#ruleList} (a planted three-rule list on 12 binary
 *       features) at label noise 0, 5 and 10%: accuracy against the noisy labels, agreement with
 *       the noise-free planted list, model size, time and whether CORELS certified optimality.</li>
 *   <li>{@link #optimalTrees} and {@link #thresholds}: the DL8.5/GOSDT dynamic program
 *       ({@link OptimalTrees#solve}) against greedy top-down splitting on {@link PlantedFunctions#xorTree}; and threshold guessing
 *       (thresholds taken from a boosted ensemble) against coarse and fine quantile thresholds on
 *       the continuous {@link PlantedFunctions#rule} data.</li>
 *   <li>{@link #rashomon}: the Rashomon set ({@link RashomonSet#enumerate}) against brute force on
 *       {@link PlantedFunctions#redundant} data, its ambiguity and discrepancy against the
 *       multiplicity that bootstrap retraining reveals, and the spread of model reliance.</li>
 *   <li>{@link #scoring}: RiskSLIM and FasterRisk ({@link ScoringSystems}) against rounding a
 *       logistic regression, the continuous logistic regression and EBM, on
 *       {@link PlantedFunctions#score}: test log loss, AUROC, expected calibration error and
 *       time.</li>
 * </ul>
 */
public final class InterpretableModels {

    private InterpretableModels() {
    }

    // ------------------------------------------------------------------ rules (13.1.10)

    /**
     * One rule-model result.
     *
     * @param method the model
     * @param noise the label noise
     * @param accuracy test accuracy against the noisy labels
     * @param truth test agreement with the noise-free planted list
     * @param size rules (CORELS, RuleFit), leaves (tree), non-zero terms or features
     * @param millis fit time
     * @param certified whether CORELS proved optimality
     */
    public record RuleRow(String method, double noise, double accuracy, double truth, int size, double millis,
            boolean certified) {
    }

    /**
     * Fits every rule model and baseline on one noise level.
     *
     * @param noise the label-flip rate
     * @param n training rows (the test set has as many)
     * @param seed the seed
     * @return one row per model
     */
    public static List<RuleRow> rules(double noise, int n, long seed) {
        int p = 12;
        PlantedFunctions.Table train = PlantedFunctions.ruleList(n, p, noise, seed);
        PlantedFunctions.Table test = PlantedFunctions.ruleList(n, p, noise, seed + 1000);
        double[] truth = Arrays.stream(test.x()).mapToDouble(PlantedFunctions::ruleListTruth).toArray();
        List<RuleRow> rows = new ArrayList<>();

        long t0 = System.nanoTime();
        List<RuleLists.Antecedent> ante = RuleLists.mine(train.x(), 0.05, true);
        RuleLists.RuleList list = RuleLists.fit(train.x(), train.y(), ante, 0.01, 4, 4_000_000);
        rows.add(row("CORELS", noise, x -> list.predict(x), test, truth, list.rules().size(), t0, list.certified()));

        t0 = System.nanoTime();
        RuleFit rf = RuleFit.fit(train.x(), train.y(), true, 15, 3, 10, seed);
        rows.add(row("RuleFit", noise, x -> rf.predict(x) > 0.5 ? 1 : 0, test, truth, rf.activeRules(), t0, false));

        t0 = System.nanoTime();
        OptimalTrees.Tree greedy = new OptimalTrees(train.x(), train.y(), 0).greedy(3);
        rows.add(row("greedy tree", noise, greedy::predict, test, truth, greedy.leaves(), t0, false));

        t0 = System.nanoTime();
        GradientBoostedTrees gbt = GradientBoostedTrees.fit(train.x(), train.y(), GradientBoostedTrees.Objective.BINARY,
                GradientBoostedTrees.Settings.defaults().withTrees(100).withDepth(3));
        rows.add(row("GBT", noise, x -> gbt.predict(x) > 0.5 ? 1 : 0, test, truth, 100, t0, false));

        t0 = System.nanoTime();
        ExplainableBoosting ebm = ExplainableBoosting.fit(train.x(), train.y(),
                GradientBoostedTrees.Objective.BINARY.loss(), ExplainableBoosting.Settings.defaults());
        rows.add(row("EBM", noise, x -> ebm.predict(x) > 0.5 ? 1 : 0, test, truth, p + ebm.pairs().length, t0, false));

        t0 = System.nanoTime();
        PenalizedRegression.Fit l1 = PenalizedRegression.of(PenalizedRegression.Penalty.LASSO)
                .family(PenalizedRegression.Family.LOGISTIC).fit(train.x(), train.y(), 0.01);
        rows.add(row("L1 logistic", noise, x -> l1.predict(new double[][] {x})[0] > 0 ? 1 : 0, test, truth,
                l1.nonZero(), t0, false));
        return rows;
    }

    private static RuleRow row(String method, double noise, ToDoubleFunction<double[]> classify,
            PlantedFunctions.Table test, double[] truth, int size, long t0, boolean certified) {
        double millis = (System.nanoTime() - t0) / 1e6;
        int right = 0;
        int agree = 0;
        for (int i = 0; i < test.rows(); i++) {
            double c = classify.applyAsDouble(test.x()[i]);
            right += c == test.y()[i] ? 1 : 0;
            agree += c == truth[i] ? 1 : 0;
        }
        return new RuleRow(method, noise, right / (double) test.rows(), agree / (double) test.rows(), size, millis,
                certified);
    }

    // ------------------------------------------------------------------ trees (13.1.11)

    /**
     * One tree result.
     *
     * @param method the method
     * @param accuracy test accuracy against the noisy labels
     * @param truth test agreement with the planted function
     * @param objective the training objective (misclassification plus the per-leaf penalty)
     * @param leaves the leaves
     * @param binary the binary features the search saw
     * @param subproblems the dynamic program's subproblems (0 for greedy)
     * @param millis the time, threshold selection included
     */
    public record TreeRow(String method, double accuracy, double truth, double objective, int leaves, int binary,
            long subproblems, double millis) {
    }

    /**
     * Optimal against greedy depth-3 trees on the XOR-rooted tree with decoy features.
     *
     * @param n training rows (the test set has as many)
     * @param seed the seed
     * @return the rows: DL8.5 then greedy
     */
    public static List<TreeRow> optimalTrees(int n, long seed) {
        PlantedFunctions.Table train = PlantedFunctions.xorTree(n, 10, 0.05, 0.15, seed);
        PlantedFunctions.Table test = PlantedFunctions.xorTree(n, 10, 0.05, 0.15, seed + 1000);
        double[] truth = Arrays.stream(test.x()).mapToDouble(PlantedFunctions::xorTreeTruth).toArray();
        OptimalTrees problem = new OptimalTrees(train.x(), train.y(), 0.005);
        long t0 = System.nanoTime();
        OptimalTrees.Result opt = problem.solve(3);
        double optMillis = (System.nanoTime() - t0) / 1e6;
        t0 = System.nanoTime();
        OptimalTrees.Tree greedy = problem.greedy(3);
        double greedyMillis = (System.nanoTime() - t0) / 1e6;
        return List.of(
                treeRow("DL8.5 depth 3", opt.tree(), problem, test.x(), test.y(), truth, 10, opt.subproblems(),
                        optMillis),
                treeRow("greedy depth 3", greedy, problem, test.x(), test.y(), truth, 10, 0, greedyMillis));
    }

    /**
     * Threshold guessing against quantile thresholds, each followed by an optimal depth-2 tree, on
     * continuous data with the planted rule {@code x1 > 0.5 and x3 < 0}.
     *
     * @param n training rows (the test set has as many)
     * @param seed the seed
     * @return the rows: coarse quantiles (3 per feature), fine quantiles (24), guessed thresholds
     */
    public static List<TreeRow> thresholds(int n, long seed) {
        int p = 6;
        PlantedFunctions.Table train = PlantedFunctions.rule(n, p, 0.05, seed);
        PlantedFunctions.Table test = PlantedFunctions.rule(n, p, 0.05, seed + 1000);
        double[] truth = Arrays.stream(test.x()).mapToDouble(x -> x[1] > 0.5 && x[3] < 0 ? 1 : 0).toArray();
        List<TreeRow> rows = new ArrayList<>();
        for (String method : new String[] {"quantiles 3", "quantiles 24", "guessed (GBT)"}) {
            long t0 = System.nanoTime();
            double[][] th = switch (method) {
                case "quantiles 3" -> OptimalTrees.quantileThresholds(train.x(), 3);
                case "quantiles 24" -> OptimalTrees.quantileThresholds(train.x(), 24);
                default -> OptimalTrees.guessedThresholds(TreeView.boost(train.x(), train.y(), 10, 2, 0.1, seed));
            };
            double[][] bx = OptimalTrees.binarise(train.x(), th);
            OptimalTrees problem = new OptimalTrees(bx, train.y(), 0.005);
            OptimalTrees.Result r = problem.solve(2);
            double millis = (System.nanoTime() - t0) / 1e6;
            rows.add(treeRow(method, r.tree(), problem, OptimalTrees.binarise(test.x(), th), test.y(), truth,
                    bx[0].length, r.subproblems(), millis));
        }
        return rows;
    }

    private static TreeRow treeRow(String method, OptimalTrees.Tree t, OptimalTrees problem, double[][] x, double[] y,
            double[] truth, int binary, long subproblems, double millis) {
        int right = 0;
        int agree = 0;
        for (int i = 0; i < x.length; i++) {
            int c = t.predict(x[i]);
            right += c == y[i] ? 1 : 0;
            agree += c == truth[i] ? 1 : 0;
        }
        return new TreeRow(method, right / (double) x.length, agree / (double) x.length, problem.objective(t),
                t.leaves(), binary, subproblems, millis);
    }

    // ------------------------------------------------------------------ Rashomon (13.1.12)

    /**
     * The Rashomon-set report.
     *
     * @param setSize members found by the shared-list enumeration
     * @param bruteSize members found by brute force (at the brute-force depth)
     * @param identical whether both lists hold the same objectives
     * @param enumerateMillis the enumeration time
     * @param bruteMillis the brute-force time
     * @param deepSize members at the deeper depth (enumeration only)
     * @param ambiguity the set's ambiguity on the test rows against the optimal tree
     * @param discrepancy the set's discrepancy
     * @param retrainAmbiguity the ambiguity of 20 optimal trees retrained on bootstrap samples
     * @param retrainDiscrepancy their discrepancy
     * @param minReliance per feature, the least model reliance in the set
     * @param maxReliance per feature, the most
     * @param copyShare the share of members using a noisy copy (feature 2 or 3)
     */
    public record RashomonReport(int setSize, int bruteSize, boolean identical, double enumerateMillis,
            double bruteMillis, int deepSize, double ambiguity, double discrepancy, double retrainAmbiguity,
            double retrainDiscrepancy, double[] minReliance, double[] maxReliance, double copyShare) {
    }

    /**
     * Enumerates the Rashomon set of sparse trees on redundant data.
     *
     * @param n training rows
     * @param epsilon the objective tolerance
     * @param seed the seed
     * @return the report
     */
    public static RashomonReport rashomon(int n, double epsilon, long seed) {
        int p = 6;
        PlantedFunctions.Table train = PlantedFunctions.redundant(n, p, 0.05, 0.03, seed);
        PlantedFunctions.Table test = PlantedFunctions.redundant(n, p, 0.05, 0.03, seed + 1000);
        double lambda = 0.01;
        OptimalTrees problem = new OptimalTrees(train.x(), train.y(), lambda);
        OptimalTrees.Tree best = problem.solve(3).tree();
        RashomonSet set = new RashomonSet(problem, 100_000);
        long t0 = System.nanoTime();
        List<RashomonSet.Member> members = set.enumerate(2, epsilon);
        double enumMillis = (System.nanoTime() - t0) / 1e6;
        t0 = System.nanoTime();
        List<RashomonSet.Member> brute = new RashomonSet(problem, 100_000).bruteForce(2, epsilon);
        double bruteMillis = (System.nanoTime() - t0) / 1e6;
        boolean identical = members.size() == brute.size();
        for (int i = 0; identical && i < members.size(); i++) {
            identical = Math.abs(members.get(i).objective() - brute.get(i).objective()) < 1e-9;
        }
        List<RashomonSet.Member> deep = set.enumerate(3, epsilon);
        List<OptimalTrees.Tree> models = deep.stream().map(RashomonSet.Member::tree).toList();
        int[] reference = Arrays.stream(test.x()).mapToInt(best::predict).toArray();
        double amb = RashomonSet.ambiguity(models, reference, test.x());
        double disc = RashomonSet.discrepancy(models, reference, test.x());

        SplittableRandom r = new SplittableRandom(seed);
        List<OptimalTrees.Tree> retrained = new ArrayList<>();
        for (int b = 0; b < 20; b++) {
            double[][] bx = new double[n][];
            double[] by = new double[n];
            for (int i = 0; i < n; i++) {
                int k = r.nextInt(n);
                bx[i] = train.x()[k];
                by[i] = train.y()[k];
            }
            retrained.add(new OptimalTrees(bx, by, lambda).solve(3).tree());
        }
        double rAmb = RashomonSet.ambiguity(retrained, reference, test.x());
        double rDisc = RashomonSet.discrepancy(retrained, reference, test.x());

        double[][] imp = RashomonSet.importance(models, test.x(), test.y(), seed);
        double[] lo = new double[p];
        double[] hi = new double[p];
        Arrays.fill(lo, Double.POSITIVE_INFINITY);
        Arrays.fill(hi, Double.NEGATIVE_INFINITY);
        int copies = 0;
        for (int m = 0; m < models.size(); m++) {
            for (int j = 0; j < p; j++) {
                lo[j] = Math.min(lo[j], imp[m][j]);
                hi[j] = Math.max(hi[j], imp[m][j]);
            }
            int[] used = models.get(m).features();
            if (Arrays.stream(used).anyMatch(f -> f == 2 || f == 3)) {
                copies++;
            }
        }
        return new RashomonReport(members.size(), brute.size(), identical, enumMillis, bruteMillis, deep.size(), amb,
                disc, rAmb, rDisc, lo, hi, copies / (double) Math.max(1, models.size()));
    }

    // ------------------------------------------------------------------ scoring (13.1.13)

    /**
     * One risk-score result.
     *
     * @param method the method
     * @param k the sparsity budget
     * @param trainLoss the training log loss (what the score methods minimise)
     * @param logLoss the test log loss
     * @param auroc the test AUROC
     * @param ece the test expected calibration error (10 bins)
     * @param size the non-zero points or coefficients
     * @param millis fit time
     * @param certified whether RiskSLIM closed its gap
     * @param points the integer points (empty for continuous models)
     */
    public record ScoreRow(String method, int k, double trainLoss, double logLoss, double auroc, double ece, int size,
            double millis, boolean certified, int[] points) {
    }

    /**
     * Fits the risk scores.
     *
     * @param n training rows (the test set has as many)
     * @param p binary features
     * @param k the sparsity budget
     * @param nonlinear whether the planted log-odds has an interaction no score fits
     * @param scale the factor on the planted log-odds (not 1: the best coefficients are not integers)
     * @param seed the seed
     * @return one row per method
     */
    public static List<ScoreRow> scoring(int n, int p, int k, boolean nonlinear, double scale, long seed) {
        int limit = 5;
        PlantedFunctions.Table train = PlantedFunctions.score(n, p, nonlinear, scale, seed);
        PlantedFunctions.Table test = PlantedFunctions.score(n, p, nonlinear, scale, seed + 1000);
        List<ScoreRow> rows = new ArrayList<>();

        long t0 = System.nanoTime();
        ScoringSystems.Score slim = ScoringSystems.riskSlim(train.x(), train.y(), k, limit, 400);
        rows.add(scoreRow("RiskSLIM", k, slim::risk, train, test, slim.size(), ms(t0), slim.certified(),
                slim.points()));

        t0 = System.nanoTime();
        ScoringSystems.Score fr = ScoringSystems.fasterRisk(train.x(), train.y(), k, limit, 10);
        rows.add(scoreRow("FasterRisk", k, fr::risk, train, test, fr.size(), ms(t0), false, fr.points()));

        t0 = System.nanoTime();
        PenalizedRegression.Fit lr = PenalizedRegression.of(PenalizedRegression.Penalty.LASSO)
                .family(PenalizedRegression.Family.LOGISTIC).fit(train.x(), train.y(), 1e-4);
        double lrMillis = ms(t0);
        double[] w = lr.beta().clone();
        Integer[] order = new Integer[p];
        for (int j = 0; j < p; j++) {
            order[j] = j;
        }
        Arrays.sort(order, Comparator.comparingDouble(j -> -Math.abs(w[j])));
        for (int j = k; j < p; j++) {
            w[order[j]] = 0;
        }
        ScoringSystems.Score naive = ScoringSystems.naiveRounding(train.x(), train.y(), w, lr.intercept(), limit);
        rows.add(scoreRow("rounded logistic", k, naive::risk, train, test, naive.size(), lrMillis, false,
                naive.points()));

        rows.add(scoreRow("logistic", k, x -> 1 / (1 + Math.exp(-lr.predict(new double[][] {x})[0])), train, test,
                lr.nonZero(), lrMillis, false, new int[0]));

        t0 = System.nanoTime();
        ExplainableBoosting ebm = ExplainableBoosting.fit(train.x(), train.y(),
                GradientBoostedTrees.Objective.BINARY.loss(), ExplainableBoosting.Settings.defaults());
        rows.add(scoreRow("EBM", k, ebm::predict, train, test, p + ebm.pairs().length, ms(t0), false, new int[0]));
        return rows;
    }

    private static ScoreRow scoreRow(String method, int k, ToDoubleFunction<double[]> risk, PlantedFunctions.Table train,
            PlantedFunctions.Table test, int size, double millis, boolean certified, int[] points) {
        double[] pr = Arrays.stream(test.x()).mapToDouble(risk).toArray();
        double[] pt = Arrays.stream(train.x()).mapToDouble(risk).toArray();
        return new ScoreRow(method, k, logLoss(pt, train.y()), logLoss(pr, test.y()), Models.auroc(pr, test.y()),
                ece(pr, test.y()), size, millis, certified, points);
    }

    private static double logLoss(double[] pr, double[] y) {
        double ll = 0;
        for (int i = 0; i < pr.length; i++) {
            double q = Math.min(1 - 1e-9, Math.max(1e-9, pr[i]));
            ll -= y[i] > 0.5 ? Math.log(q) : Math.log(1 - q);
        }
        return ll / pr.length;
    }

    private static double ms(long t0) {
        return (System.nanoTime() - t0) / 1e6;
    }

    /**
     * The expected calibration error over 10 equal-width bins.
     *
     * @param p predicted probabilities
     * @param y 0/1 labels
     * @return the error
     */
    static double ece(double[] p, double[] y) {
        int bins = 10;
        double[] sp = new double[bins];
        double[] sy = new double[bins];
        for (int i = 0; i < p.length; i++) {
            int b = Math.min(bins - 1, (int) (p[i] * bins));
            sp[b] += p[i];
            sy[b] += y[i];
        }
        double e = 0;
        for (int b = 0; b < bins; b++) {
            e += Math.abs(sp[b] - sy[b]) / p.length;
        }
        return e;
    }
}
