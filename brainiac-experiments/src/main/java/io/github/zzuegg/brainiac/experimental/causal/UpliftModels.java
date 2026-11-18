package io.github.zzuegg.brainiac.experimental.causal;

import io.github.zzuegg.brainiac.experimental.ml.RegressionForest;

import java.util.SplittableRandom;

/**
 * Uplift models for randomised campaigns: estimators of
 * {@code uplift(x) = P(y = 1 | t = 1, x) - P(y = 1 | t = 0, x)}, the change the treatment causes,
 * rather than the response probability (Radcliffe 2007; Jaśkowski and Jaroszewicz 2012; Gutierrez
 * and Gérardy 2017, <a href="https://proceedings.mlr.press/v67/gutierrez17a.html">PMLR v67</a>).
 * Targeting by response picks "sure things" who would buy anyway; targeting by uplift picks the
 * "persuadables" and avoids the "do-not-disturbs".
 * <ul>
 *   <li>{@link #classTransformation}: the transformed outcome {@code z = y (t - p) / (p (1 - p))}
 *       has {@code E[z | x] = uplift(x)} under randomisation with rate {@code p} (for {@code p = 1/2}
 *       it is the class-variable trick {@code P(z' = 1 | x) = (1 + uplift(x)) / 2}), so any regressor
 *       fits uplift directly;</li>
 *   <li>{@link #twoModel}: separate response models per arm, differenced;</li>
 *   <li>{@link #forest}: uplift trees whose splits maximise the divergence between the treated and
 *       control outcome distributions in the children: Euclidean {@link Criterion#DELTA_DELTA_P}
 *       (the difference of the children's uplifts) or {@link Criterion#KL} (the gain in
 *       Kullback-Leibler divergence), bagged into a forest.</li>
 * </ul>
 */
public final class UpliftModels {

    /** The uplift-tree split criterion. */
    public enum Criterion {
        /** {@code |uplift_left - uplift_right|} (Hansotia and Rukstales). */
        DELTA_DELTA_P,
        /** Weighted KL divergence of the children minus the parent's (Rzepakowski and Jaroszewicz). */
        KL
    }

    private UpliftModels() {
    }

    /**
     * The class-transformation (transformed-outcome) model.
     *
     * @param x covariates
     * @param t treatments
     * @param y 0/1 outcomes
     * @param model the regressor fitted to the transformed outcome
     * @return the uplift function
     */
    public static Regressor.Fitted classTransformation(double[][] x, int[] t, int[] y, Regressor model) {
        double p = 0;
        for (int v : t) {
            p += v;
        }
        p /= t.length;
        double[] z = new double[t.length];
        for (int i = 0; i < z.length; i++) {
            z[i] = y[i] * (t[i] - p) / (p * (1 - p));
        }
        return model.fit(x, z, null);
    }

    /**
     * The two-model approach.
     *
     * @param x covariates
     * @param t treatments
     * @param y 0/1 outcomes
     * @param model the per-arm response regressor
     * @return the uplift function
     */
    public static Regressor.Fitted twoModel(double[][] x, int[] t, int[] y, Regressor model) {
        return new MetaLearners(model, model, model).fit(MetaLearners.Kind.T, x, t, CrossFitting.asDouble(y));
    }

    /**
     * A bagged uplift forest.
     *
     * @param x covariates
     * @param t treatments
     * @param y 0/1 outcomes
     * @param criterion the split criterion
     * @param trees trees
     * @param maxDepth the depth limit
     * @param minArm the fewest treated and the fewest control units in each child
     * @param seed the seed
     * @return the uplift function
     */
    public static Regressor.Fitted forest(double[][] x, int[] t, int[] y, Criterion criterion, int trees,
            int maxDepth, int minArm, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        RegressionForest.Tree[] grown = new RegressionForest.Tree[trees];
        UpliftScore score = new UpliftScore(t, y, criterion, minArm);
        for (int b = 0; b < trees; b++) {
            int[] rows = RegressionForest.subsample(x.length, 1, false, random)[0];
            grown[b] = RegressionForest.grow(x, rows, maxDepth, 2 * minArm, 0, score, random,
                    r -> leafUplift(r, t, y));
        }
        return row -> {
            double s = 0;
            for (RegressionForest.Tree tree : grown) {
                s += tree.predict(row);
            }
            return s / grown.length;
        };
    }

    private static double leafUplift(int[] rows, int[] t, int[] y) {
        double nt = 0;
        double yt = 0;
        double nc = 0;
        double yc = 0;
        for (int r : rows) {
            if (t[r] == 1) {
                nt++;
                yt += y[r];
            } else {
                nc++;
                yc += y[r];
            }
        }
        return (yt + 0.5) / (nt + 1) - (yc + 0.5) / (nc + 1);
    }

    /** Divergence-based uplift split score with per-arm prefix counts. */
    private static final class UpliftScore implements RegressionForest.SplitScore {
        private final int[] t;
        private final int[] y;
        private final Criterion criterion;
        private final int minArm;
        private int[] cached;
        private double[][] prefix = new double[4][0];
        private final double[] total = new double[4];

        UpliftScore(int[] t, int[] y, Criterion criterion, int minArm) {
            this.t = t;
            this.y = y;
            this.criterion = criterion;
            this.minArm = minArm;
        }

        @Override
        public void node(int[] rows) {
            java.util.Arrays.fill(total, 0);
            for (int r : rows) {
                total[t[r] == 1 ? 0 : 2]++;
                total[t[r] == 1 ? 1 : 3] += y[r];
            }
            cached = null;
        }

        @Override
        public double gain(int[] rows, int cut) {
            if (rows != cached) {
                prefix = new double[4][rows.length + 1];
                for (int i = 0; i < rows.length; i++) {
                    int r = rows[i];
                    for (int k = 0; k < 4; k++) {
                        prefix[k][i + 1] = prefix[k][i];
                    }
                    prefix[t[r] == 1 ? 0 : 2][i + 1]++;
                    prefix[t[r] == 1 ? 1 : 3][i + 1] += y[r];
                }
                cached = rows;
            }
            double ntL = prefix[0][cut];
            double ytL = prefix[1][cut];
            double ncL = prefix[2][cut];
            double ycL = prefix[3][cut];
            double ntR = total[0] - ntL;
            double ytR = total[1] - ytL;
            double ncR = total[2] - ncL;
            double ycR = total[3] - ycL;
            if (ntL < minArm || ncL < minArm || ntR < minArm || ncR < minArm) {
                return Double.NaN;
            }
            double ptL = (ytL + 0.5) / (ntL + 1);
            double pcL = (ycL + 0.5) / (ncL + 1);
            double ptR = (ytR + 0.5) / (ntR + 1);
            double pcR = (ycR + 0.5) / (ncR + 1);
            if (criterion == Criterion.DELTA_DELTA_P) {
                return Math.abs((ptL - pcL) - (ptR - pcR));
            }
            double n = total[0] + total[2];
            double pt = (total[1] + 0.5) / (total[0] + 1);
            double pc = (total[3] + 0.5) / (total[2] + 1);
            double left = (ntL + ncL) / n;
            return left * kl(ptL, pcL) + (1 - left) * kl(ptR, pcR) - kl(pt, pc);
        }

        private static double kl(double p, double q) {
            return p * Math.log(p / q) + (1 - p) * Math.log((1 - p) / (1 - q));
        }
    }
}
