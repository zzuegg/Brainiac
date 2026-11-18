package io.github.zzuegg.brainiac.experiments.explainability;

import io.github.zzuegg.brainiac.experimental.data.PlantedFunctions;
import io.github.zzuegg.brainiac.experimental.explain.Anchors;
import io.github.zzuegg.brainiac.experimental.explain.Recourse;
import io.github.zzuegg.brainiac.experimental.nn.KnnIndex;
import io.github.zzuegg.brainiac.experimental.stats.PenalizedRegression;
import io.github.zzuegg.brainiac.experimental.train.ModelAgnosticExplainer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.GradientBoostedTrees;

import java.util.Arrays;
import java.util.EnumMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Wave 13 items 13.1.5 and 13.1.6: local rule explanations and recourse.
 *
 * <p>{@link #anchors}: on {@link PlantedFunctions#rule} ({@code x1 > 0.5 and x3 < 0}, 8
 * features, 5% label noise), a GBT and an MLP are explained by {@link Anchors} with the KL-LUCB
 * and the uniform sampler, and by the rules that LIME's and KernelSHAP's top-2 features make
 * (the bins of {@code x} on those features, the comparison of the paper). Scores: held-out
 * precision (the share of held-out rows satisfying the rule that get the same prediction), held-out
 * coverage, the share of rules naming only planted features, and model calls.
 *
 * <p>{@link #recourse}: on {@link PlantedFunctions#credit} (age and tenure immutable, income
 * and education increase-only, debt decrease-only), denied rows of a logistic regression and of
 * an MLP get recourse from {@link Recourse}'s integer program, robust IP, DiCE and FACE, against
 * {@link ModelAgnosticExplainer#counterfactual} (Wachter) and the nearest accepted training row
 * (via {@link KnnIndex}). Scores: validity, actionability, cost, local density, DiCE's diversity,
 * and validity after the model is refitted on a fresh sample.
 */
public final class LocalRules {

    private LocalRules() {
    }

    /** The rule explainers. */
    public enum RuleMethod {
        /** Anchors with KL-LUCB. */
        ANCHOR_KL_LUCB,
        /** Anchors with uniform sampling. */
        ANCHOR_UNIFORM,
        /** LIME's top-2 features as a rule. */
        LIME_TOP2,
        /** KernelSHAP's top-2 features as a rule. */
        KERNEL_SHAP_TOP2
    }

    /**
     * Scores of a rule explainer.
     *
     * @param precision mean held-out precision
     * @param coverage mean held-out coverage
     * @param planted share of rules naming only (some of) the planted features 1 and 3
     * @param calls mean model calls per explanation
     */
    public record RuleScores(double precision, double coverage, double planted, double calls) {
    }

    /**
     * The anchors table.
     *
     * @param trees GBT (else MLP)
     * @param rows rows explained
     * @param seed the seed
     * @return scores per method
     */
    public static Map<RuleMethod, RuleScores> anchors(boolean trees, int rows, long seed) {
        PlantedFunctions.Table all = PlantedFunctions.rule(3000, 8, 0.05, seed);
        PlantedFunctions.Table train = all.slice(0, 2000);
        PlantedFunctions.Table held = all.slice(2000, 3000);
        Function<double[][], double[]> prob;
        if (trees) {
            GradientBoostedTrees g = GradientBoostedTrees.fit(train.x(), train.y(), GradientBoostedTrees.Objective.BINARY,
                    GradientBoostedTrees.Settings.defaults().withDepth(3));
            prob = g::predict;
        } else {
            prob = Models.probability(Models.mlpClassifier(train.x(), train.y(), 2, 32, 1500, seed));
        }
        Function<double[][], int[]> cls = x -> Arrays.stream(prob.apply(x)).mapToInt(v -> v > 0.5 ? 1 : 0).toArray();
        int[] heldPred = cls.apply(held.x());
        Map<RuleMethod, double[]> sums = new EnumMap<>(RuleMethod.class);
        for (RuleMethod m : RuleMethod.values()) {
            sums.put(m, new double[5]);
        }
        Anchors klLucb = new Anchors(cls, train.x(), 4, 0.95, 0.1, 2, Anchors.Sampler.KL_LUCB);
        Anchors uniform = new Anchors(cls, train.x(), 4, 0.95, 0.1, 2, Anchors.Sampler.UNIFORM);
        ModelAgnosticExplainer black = new ModelAgnosticExplainer(prob);
        double[] scale = new double[8];
        Arrays.fill(scale, 1);
        double[][] background = Arrays.copyOf(train.x(), 20);
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < rows; i++) {
            double[] x = held.x()[i];
            int target = heldPred[i];
            for (RuleMethod m : RuleMethod.values()) {
                Anchors.Anchor a = switch (m) {
                    case ANCHOR_KL_LUCB -> klLucb.explain(x, 4, seed + i);
                    case ANCHOR_UNIFORM -> uniform.explain(x, 4, seed + i);
                    case LIME_TOP2 -> klLucb.ruleOf(x, top(black.lime(x, scale, 500, 0.75 * Math.sqrt(8), r), 2));
                    case KERNEL_SHAP_TOP2 -> klLucb.ruleOf(x, top(black.kernelShap(x, background, 254, r), 2));
                };
                double[] s = sums.get(m);
                int cover = 0;
                int agree = 0;
                for (int k = 0; k < held.rows(); k++) {
                    if (k != i && a.holds(held.x()[k])) {
                        cover++;
                        if (heldPred[k] == target) {
                            agree++;
                        }
                    }
                }
                if (cover > 0) {
                    s[0] += agree / (double) cover;
                    s[4]++;
                }
                s[1] += cover / (double) (held.rows() - 1);
                int[] f = a.features();
                boolean onlyPlanted = f.length > 0 && Arrays.stream(f).allMatch(v -> v == 1 || v == 3);
                s[2] += onlyPlanted ? 1 : 0;
                s[3] += m == RuleMethod.LIME_TOP2 ? 500 : m == RuleMethod.KERNEL_SHAP_TOP2 ? 254 * 20 : a.calls();
            }
        }
        Map<RuleMethod, RuleScores> out = new EnumMap<>(RuleMethod.class);
        sums.forEach((m, s) -> out.put(m, new RuleScores(s[0] / Math.max(1, s[4]), s[1] / rows, s[2] / rows,
                s[3] / rows)));
        return out;
    }

    private static int[] top(double[] attribution, int k) {
        Integer[] idx = new Integer[attribution.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(Math.abs(attribution[b]), Math.abs(attribution[a])));
        int[] out = new int[k];
        for (int i = 0; i < k; i++) {
            out[i] = idx[i];
        }
        return out;
    }

    /** The recourse methods. */
    public enum RecourseMethod {
        /** Integer-programming recourse (linear model only). */
        LINEAR_IP,
        /** Robust integer-programming recourse, epsilon 0.15 (linear model only). */
        ROBUST_IP,
        /** DiCE, k = 4. */
        DICE,
        /** FACE on 800 training rows. */
        FACE,
        /** Wachter's counterfactual (existing {@code ModelAgnosticExplainer.counterfactual}). */
        WACHTER,
        /** The nearest accepted training row. */
        NEAREST_UNLIKE
    }

    /**
     * Scores of a recourse method.
     *
     * @param validity share of rows whose (first) counterfactual is accepted
     * @param actionable share of counterfactuals that change no immutable feature and respect directions
     * @param cost mean cost of the mutable changes (cost-weighted L1 in standard deviations)
     * @param density mean k-NN density proxy at the counterfactual (larger is denser)
     * @param diversity mean pairwise L1 distance within DiCE's set (0 for single answers)
     * @param featuresCovered mean number of distinct features changed across the set
     * @param validityAfterUpdate share still accepted after the model is refitted on fresh data
     */
    public record RecourseScores(double validity, double actionable, double cost, double density, double diversity,
            double featuresCovered, double validityAfterUpdate) {
    }

    private static Recourse.Constraints constraints() {
        Recourse.Mutability[] m = new Recourse.Mutability[6];
        for (int j = 0; j < 6; j++) {
            m[j] = Recourse.Mutability.valueOf(PlantedFunctions.CREDIT_ACTIONS[j].name());
        }
        double[] scale = new double[6];
        Arrays.fill(scale, 1);
        return new Recourse.Constraints(m, PlantedFunctions.CREDIT_COSTS.clone(), scale);
    }

    private static double mutableCost(Recourse.Constraints c, double[] x, double[] cf) {
        double s = 0;
        for (int j = 0; j < x.length; j++) {
            if (c.mutability()[j] != Recourse.Mutability.IMMUTABLE) {
                s += c.cost()[j] * Math.abs(cf[j] - x[j]);
            }
        }
        return s;
    }

    /**
     * The recourse table.
     *
     * @param linear explain a logistic regression (else an MLP; the IP rows are then skipped)
     * @param rows denied rows explained
     * @param seed the seed
     * @return scores per method
     */
    public static Map<RecourseMethod, RecourseScores> recourse(boolean linear, int rows, long seed) {
        PlantedFunctions.Table train = PlantedFunctions.credit(2000, seed);
        PlantedFunctions.Table fresh = PlantedFunctions.credit(400, seed + 7);
        PlantedFunctions.Table test = PlantedFunctions.credit(400, seed + 3);
        double[] w = null;
        double b = 0;
        Function<double[][], double[]> logit;
        Function<double[][], double[]> updatedLogit;
        if (linear) {
            PenalizedRegression lr = PenalizedRegression.of(PenalizedRegression.Penalty.LASSO)
                    .family(PenalizedRegression.Family.LOGISTIC);
            PenalizedRegression.Fit fit = lr.fit(train.x(), train.y(), 1e-4);
            PenalizedRegression.Fit upd = lr.fit(fresh.x(), fresh.y(), 1e-4);
            w = fit.beta();
            b = fit.intercept();
            logit = fit::predict;
            updatedLogit = upd::predict;
        } else {
            Model m = Models.mlpClassifier(train.x(), train.y(), 2, 32, 1500, seed);
            Model u = Models.mlpClassifier(fresh.x(), fresh.y(), 2, 32, 1500, seed + 1);
            logit = logOdds(m);
            updatedLogit = logOdds(u);
        }
        Function<double[][], double[]> prob = rowsIn -> Arrays.stream(logit.apply(rowsIn))
                .map(v -> 1 / (1 + Math.exp(-v))).toArray();
        Recourse.Constraints c = constraints();
        double[] testLogit = logit.apply(test.x());
        double[][] accepted = Arrays.stream(Arrays.copyOf(train.x(), 800))
                .filter(x -> logit.apply(new double[][] {x})[0] > 0).toArray(double[][]::new);
        KnnIndex index = KnnIndex.create(6, 0, KnnIndex.Metric.L2);
        for (double[] a : accepted) {
            index.add(floats(a), new float[0]);
        }
        Map<RecourseMethod, double[]> sums = new EnumMap<>(RecourseMethod.class);
        int done = 0;
        for (int i = 0; i < test.rows() && done < rows; i++) {
            if (testLogit[i] >= 0) {
                continue;
            }
            done++;
            double[] x = test.x()[i];
            for (RecourseMethod m : RecourseMethod.values()) {
                if (!linear && (m == RecourseMethod.LINEAR_IP || m == RecourseMethod.ROBUST_IP)) {
                    continue;
                }
                double[][] set = switch (m) {
                    case LINEAR_IP -> single(Recourse.linear(w, b, x, c, 0.05, 12, 0.25, 0));
                    case ROBUST_IP -> single(Recourse.linear(w, b, x, c, 0.05, 12, 0.25, 0.15));
                    case DICE -> Recourse.dice(prob, 0.5, x, c, 4, 200, 1.0, seed + i);
                    case FACE -> single(Recourse.face(logit, 0, x, c, Arrays.copyOf(train.x(), 800), 15));
                    case WACHTER -> single(new ModelAgnosticExplainer(logit).counterfactual(x, c.scale(), 0, 0.05, 40,
                            0.01).changed());
                    case NEAREST_UNLIKE -> single(accepted[index.search(floats(x), 1)[0].row()]);
                };
                double[] s = sums.computeIfAbsent(m, k -> new double[8]);
                if (set == null) {
                    s[7]++;
                    continue;
                }
                double[] first = set[0];
                double[] now = logit.apply(set);
                double[] later = updatedLogit.apply(set);
                s[0] += now[0] >= 0 ? 1 : 0;
                boolean act = c.actionable(x, first);
                s[1] += act ? 1 : 0;
                s[2] += mutableCost(c, x, first);
                s[3] += Recourse.localDensity(first, Arrays.copyOf(train.x(), 800), c.scale(), 5);
                double div = 0;
                int pairs = 0;
                Set<Integer> feats = new HashSet<>();
                for (int a = 0; a < set.length; a++) {
                    for (int j = 0; j < 6; j++) {
                        if (Math.abs(set[a][j] - x[j]) > 1e-6) {
                            feats.add(j);
                        }
                    }
                    for (int bb = a + 1; bb < set.length; bb++) {
                        for (int j = 0; j < 6; j++) {
                            div += Math.abs(set[a][j] - set[bb][j]);
                        }
                        pairs++;
                    }
                }
                s[4] += pairs > 0 ? div / pairs : 0;
                s[5] += feats.size();
                s[6] += later[0] >= 0 ? 1 : 0;
            }
        }
        Map<RecourseMethod, RecourseScores> out = new EnumMap<>(RecourseMethod.class);
        int n = done;
        sums.forEach((m, s) -> {
            double found = n - s[7];
            out.put(m, new RecourseScores(s[0] / n, s[1] / Math.max(1, found), s[2] / Math.max(1, found),
                    s[3] / Math.max(1, found), s[4] / Math.max(1, found), s[5] / Math.max(1, found), s[6] / n));
        });
        return out;
    }

    private static double[][] single(double[] row) {
        return row == null ? null : new double[][] {row};
    }

    private static float[] floats(double[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = (float) v[i];
        }
        return f;
    }

    private static Function<double[][], double[]> logOdds(Model m) {
        Function<double[][], double[]> p = Models.probability(m);
        return rows -> Arrays.stream(p.apply(rows)).map(v -> {
            double q = Math.min(Math.max(v, 1e-9), 1 - 1e-9);
            return Math.log(q / (1 - q));
        }).toArray();
    }
}
