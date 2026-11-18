package io.github.zzuegg.brainiac.experimental.explain;

import io.github.zzuegg.brainiac.experimental.stats.PenalizedRegression;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * RuleFit (Friedman and Popescu, "Predictive learning via rule ensembles", Annals of Applied
 * Statistics 2008): every root-to-node path of a boosted tree ensemble becomes a binary rule (a
 * conjunction of threshold conditions), and the lasso picks a sparse linear model over the rules
 * and the raw features, giving a short list of weighted rules. The rules come from
 * {@link TreeView#boost}; the lasso is {@link PenalizedRegression} (logistic for 0/1 labels),
 * with the penalty chosen on its regularisation path as the smallest that keeps at most
 * {@code maxTerms} non-zero terms.
 */
public final class RuleFit {

    /**
     * A condition {@code x_feature > threshold} (or {@code <=}).
     *
     * @param feature the feature
     * @param greater whether the condition is {@code >}
     * @param threshold the threshold
     */
    public record Condition(int feature, boolean greater, double threshold) {

        boolean holds(double[] x) {
            return greater ? x[feature] > threshold : x[feature] <= threshold;
        }
    }

    /**
     * A rule, the conjunction of its conditions.
     *
     * @param conditions the conditions
     */
    public record Rule(List<Condition> conditions) {

        /**
         * @param x a row
         * @return 1 if every condition holds, else 0
         */
        public double value(double[] x) {
            for (Condition c : conditions) {
                if (!c.holds(x)) {
                    return 0;
                }
            }
            return 1;
        }
    }

    private final List<Rule> rules;
    private final int features;
    private final PenalizedRegression.Fit fit;
    private final boolean classification;

    private RuleFit(List<Rule> rules, int features, PenalizedRegression.Fit fit, boolean classification) {
        this.rules = rules;
        this.features = features;
        this.fit = fit;
        this.classification = classification;
    }

    /**
     * Extracts every distinct root-to-node rule of the ensemble.
     *
     * @param ensemble the trees
     * @return the rules
     */
    public static List<Rule> extract(TreeView ensemble) {
        List<Rule> out = new ArrayList<>();
        Set<List<Condition>> seen = new HashSet<>();
        for (TreeView.Tree t : ensemble.trees()) {
            walk(t, 0, new ArrayList<>(), out, seen);
        }
        return out;
    }

    private static void walk(TreeView.Tree t, int node, List<Condition> path, List<Rule> out,
            Set<List<Condition>> seen) {
        if (!path.isEmpty() && seen.add(List.copyOf(path))) {
            out.add(new Rule(List.copyOf(path)));
        }
        int f = t.feature()[node];
        if (f < 0) {
            return;
        }
        path.add(new Condition(f, false, t.threshold()[node]));
        walk(t, t.left()[node], path, out, seen);
        path.set(path.size() - 1, new Condition(f, true, t.threshold()[node]));
        walk(t, t.right()[node], path, out, seen);
        path.remove(path.size() - 1);
    }

    private double[] design(double[] x) {
        double[] row = new double[rules.size() + features];
        for (int k = 0; k < rules.size(); k++) {
            row[k] = rules.get(k).value(x);
        }
        System.arraycopy(x, 0, row, rules.size(), features);
        return row;
    }

    /**
     * Fits RuleFit.
     *
     * @param x the rows
     * @param y the targets (0/1 for classification)
     * @param classification whether to use the logistic lasso
     * @param trees the number of boosted trees generating rules
     * @param depth their depth
     * @param maxTerms the most non-zero rules and features
     * @param seed the boosting seed
     * @return the fitted model
     */
    public static RuleFit fit(double[][] x, double[] y, boolean classification, int trees, int depth, int maxTerms,
            long seed) {
        TreeView ensemble = TreeView.boost(x, y, trees, depth, 0.1, seed);
        List<Rule> rules = extract(ensemble);
        RuleFit shell = new RuleFit(rules, x[0].length, null, classification);
        double[][] z = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            z[i] = shell.design(x[i]);
        }
        PenalizedRegression lasso = PenalizedRegression.of(PenalizedRegression.Penalty.LASSO)
                .family(classification ? PenalizedRegression.Family.LOGISTIC : PenalizedRegression.Family.GAUSSIAN);
        PenalizedRegression.Path path = lasso.path(z, y, 40, 0.005);
        PenalizedRegression.Fit chosen = path.fits().get(0);
        for (PenalizedRegression.Fit f : path.fits()) {
            if (f.nonZero() <= maxTerms) {
                chosen = f;
            }
        }
        return new RuleFit(rules, x[0].length, chosen, classification);
    }

    /**
     * @param x a row
     * @return the probability of class 1 (classification) or the value
     */
    public double predict(double[] x) {
        double eta = fit.predict(new double[][] {design(x)})[0];
        return classification ? 1 / (1 + Math.exp(-eta)) : eta;
    }

    /** @return the number of candidate rules extracted */
    public int candidateRules() {
        return rules.size();
    }

    /** @return the number of rules with a non-zero weight */
    public int activeRules() {
        int c = 0;
        for (int k = 0; k < rules.size(); k++) {
            if (fit.beta()[k] != 0) {
                c++;
            }
        }
        return c;
    }

    /** @return the total number of conditions in the active rules plus the active linear terms */
    public int activeConditions() {
        int c = 0;
        for (int k = 0; k < rules.size(); k++) {
            if (fit.beta()[k] != 0) {
                c += rules.get(k).conditions().size();
            }
        }
        for (int j = 0; j < features; j++) {
            if (fit.beta()[rules.size() + j] != 0) {
                c++;
            }
        }
        return c;
    }
}
