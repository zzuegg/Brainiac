package io.github.zzuegg.brainiac.experimental.logic;

import io.github.zzuegg.brainiac.data.LeastSquares;

import java.util.ArrayList;
import java.util.List;

/**
 * Fuzzy inference systems (Mamdani &amp; Assilian, 1975; Takagi &amp; Sugeno, IEEE SMC 1985). Each
 * rule has one membership function per input; a rule fires with the product of its memberships
 * (the product t-norm).
 *
 * <ul>
 *   <li>{@link Type#TAKAGI_SUGENO}: first-order rules {@code y_r = p_r . x + q_r}, combined by the
 *       firing strengths normalised to sum to one. The consequents are linear in the parameters
 *       given the premises, so {@link #fitConsequents} solves them by least squares (core
 *       {@code LeastSquares.ols}, ridge by row augmentation).</li>
 *   <li>{@link Type#MAMDANI}: rules with a Gaussian output set; the clipped output sets are
 *       aggregated by max and defuzzified by the centroid on a grid. {@link #fitConsequents}
 *       places each output set at the firing-weighted mean of the targets.</li>
 *   <li>Membership families: {@link Membership#GAUSSIAN}, generalised {@link Membership#BELL}
 *       ({@code 1 / (1 + |(x - c) / a|^(2b))}, {@code b = 2}) and {@link Membership#TRIANGULAR}.</li>
 *   <li>Rule bases: a grid partition ({@link #grid}, {@code m^d} rules) or subtractive clustering
 *       (Chiu, 1994; {@link #clustering}), whose rule count follows the data instead of the
 *       input dimension.</li>
 * </ul>
 * {@link Anfis} trains the premises of a Takagi–Sugeno system.
 */
public final class FuzzyInference {

    /** The rule consequent form. */
    public enum Type {
        /** Fuzzy output sets, centroid defuzzification. */
        MAMDANI,
        /** First-order linear consequents. */
        TAKAGI_SUGENO
    }

    /** Membership function families. */
    public enum Membership {
        /** {@code exp(-((x - c) / s)^2 / 2)}. */
        GAUSSIAN,
        /** {@code 1 / (1 + ((x - c) / a)^4)}. */
        BELL,
        /** {@code max(0, 1 - |x - c| / a)}. */
        TRIANGULAR
    }

    private static final double BELL_B = 2;

    final Type type;
    final Membership membership;
    final int inputs;
    final double[][] center;
    final double[][] width;
    final double[][] consequent;
    private double outLow;
    private double outHigh;

    private FuzzyInference(Type type, Membership membership, double[][] center, double[][] width) {
        this.type = type;
        this.membership = membership;
        this.inputs = center[0].length;
        this.center = center;
        this.width = width;
        this.consequent = new double[center.length][type == Type.TAKAGI_SUGENO ? inputs + 1 : 2];
    }

    /** @return the number of rules */
    public int rules() {
        return center.length;
    }

    /** @return the type */
    public Type type() {
        return type;
    }

    /** @return the membership family */
    public Membership membership() {
        return membership;
    }

    /** @return the number of adjustable parameters (premises plus consequents) */
    public int parameters() {
        return center.length * (2 * inputs + consequent[0].length);
    }

    /**
     * One membership value.
     *
     * @param m the family
     * @param x the input
     * @param c the centre
     * @param w the width
     * @return the membership degree
     */
    public static double membership(Membership m, double x, double c, double w) {
        double z = (x - c) / w;
        return switch (m) {
            case GAUSSIAN -> Math.exp(-0.5 * z * z);
            case BELL -> 1 / (1 + Math.pow(z * z, BELL_B));
            case TRIANGULAR -> Math.max(0, 1 - Math.abs(z));
        };
    }

    /**
     * Rule firing strengths.
     *
     * @param x the input
     * @return per rule the product of memberships
     */
    public double[] firing(double[] x) {
        double[] w = new double[center.length];
        for (int r = 0; r < w.length; r++) {
            double p = 1;
            for (int i = 0; i < inputs; i++) {
                p *= membership(membership, x[i], center[r][i], width[r][i]);
            }
            w[r] = p;
        }
        return w;
    }

    private double[] normalised(double[] x) {
        double[] w = firing(x);
        double s = 0;
        for (double v : w) {
            s += v;
        }
        for (int r = 0; r < w.length; r++) {
            w[r] = s > 1e-300 ? w[r] / s : 1.0 / w.length;
        }
        return w;
    }

    /**
     * The system's output.
     *
     * @param x the input
     * @return the crisp output
     */
    public double predict(double[] x) {
        if (type == Type.TAKAGI_SUGENO) {
            double[] w = normalised(x);
            double y = 0;
            for (int r = 0; r < w.length; r++) {
                double f = consequent[r][inputs];
                for (int i = 0; i < inputs; i++) {
                    f += consequent[r][i] * x[i];
                }
                y += w[r] * f;
            }
            return y;
        }
        double[] w = firing(x);
        int grid = 101;
        double num = 0;
        double den = 0;
        for (int k = 0; k < grid; k++) {
            double y = outLow + (outHigh - outLow) * k / (grid - 1);
            double mu = 0;
            for (int r = 0; r < w.length; r++) {
                mu = Math.max(mu, Math.min(w[r], membership(Membership.GAUSSIAN, y, consequent[r][0],
                    consequent[r][1])));
            }
            num += mu * y;
            den += mu;
        }
        return den > 0 ? num / den : (outLow + outHigh) / 2;
    }

    /**
     * Fits the consequents with the premises fixed: least squares for Takagi–Sugeno, weighted
     * target means for Mamdani.
     *
     * @param x inputs
     * @param y targets
     * @param ridge the ridge penalty (Takagi–Sugeno)
     */
    public void fitConsequents(double[][] x, double[] y, double ridge) {
        int n = x.length;
        int rules = center.length;
        if (type == Type.MAMDANI) {
            outLow = Double.POSITIVE_INFINITY;
            outHigh = Double.NEGATIVE_INFINITY;
            for (double v : y) {
                outLow = Math.min(outLow, v);
                outHigh = Math.max(outHigh, v);
            }
            double[] num = new double[rules];
            double[] den = new double[rules];
            for (int i = 0; i < n; i++) {
                double[] w = normalised(x[i]);
                for (int r = 0; r < rules; r++) {
                    num[r] += w[r] * y[i];
                    den[r] += w[r];
                }
            }
            for (int r = 0; r < rules; r++) {
                consequent[r][0] = den[r] > 0 ? num[r] / den[r] : (outLow + outHigh) / 2;
                consequent[r][1] = Math.max(1e-6, (outHigh - outLow) / (2.0 * Math.sqrt(rules)));
            }
            return;
        }
        int k = rules * (inputs + 1);
        double[][] design = new double[n + k][k];
        double[] target = new double[n + k];
        for (int i = 0; i < n; i++) {
            double[] w = normalised(x[i]);
            for (int r = 0; r < rules; r++) {
                for (int j = 0; j < inputs; j++) {
                    design[i][r * (inputs + 1) + j] = w[r] * x[i][j];
                }
                design[i][r * (inputs + 1) + inputs] = w[r];
            }
            target[i] = y[i];
        }
        double s = Math.sqrt(Math.max(ridge, 1e-10));
        for (int j = 0; j < k; j++) {
            design[n + j][j] = s;
        }
        double[] c = LeastSquares.ols(design, target, false).coefficients();
        for (int r = 0; r < rules; r++) {
            System.arraycopy(c, r * (inputs + 1), consequent[r], 0, inputs + 1);
        }
    }

    /**
     * A grid partition: {@code perInput} evenly spaced membership functions per input, one rule
     * per combination.
     *
     * @param type the consequent form
     * @param membership the membership family
     * @param low per input the lower bound
     * @param high per input the upper bound
     * @param perInput membership functions per input
     * @return the system (consequents zero until fitted)
     */
    public static FuzzyInference grid(Type type, Membership membership, double[] low, double[] high, int perInput) {
        int d = low.length;
        int rules = (int) Math.pow(perInput, d);
        double[][] c = new double[rules][d];
        double[][] w = new double[rules][d];
        for (int r = 0; r < rules; r++) {
            int rem = r;
            for (int i = 0; i < d; i++) {
                int k = rem % perInput;
                rem /= perInput;
                double spacing = (high[i] - low[i]) / Math.max(1, perInput - 1);
                c[r][i] = perInput == 1 ? (low[i] + high[i]) / 2 : low[i] + k * spacing;
                w[r][i] = membership == Membership.TRIANGULAR ? spacing : spacing / 2;
            }
        }
        return new FuzzyInference(type, membership, c, w);
    }

    /**
     * Subtractive clustering (Chiu, 1994) of the inputs: every point's potential is a sum of
     * Gaussians over the others; the highest becomes a cluster centre and reduces the potentials
     * around it, until the best remaining potential falls below {@code 0.15} of the first. Each
     * centre becomes a rule with Gaussian memberships of width {@code radius * range / sqrt(8)}.
     *
     * @param type the consequent form
     * @param membership the membership family
     * @param x inputs
     * @param radius the cluster radius as a share of each input's range
     * @return the system (consequents zero until fitted)
     */
    public static FuzzyInference clustering(Type type, Membership membership, double[][] x, double radius) {
        int n = x.length;
        int d = x[0].length;
        double[] lo = new double[d];
        double[] hi = new double[d];
        java.util.Arrays.fill(lo, Double.POSITIVE_INFINITY);
        java.util.Arrays.fill(hi, Double.NEGATIVE_INFINITY);
        for (double[] row : x) {
            for (int i = 0; i < d; i++) {
                lo[i] = Math.min(lo[i], row[i]);
                hi[i] = Math.max(hi[i], row[i]);
            }
        }
        double[][] u = new double[n][d];
        for (int p = 0; p < n; p++) {
            for (int i = 0; i < d; i++) {
                u[p][i] = (x[p][i] - lo[i]) / Math.max(1e-12, hi[i] - lo[i]);
            }
        }
        double alpha = 4 / (radius * radius);
        double beta = 4 / (1.5 * radius * 1.5 * radius);
        double[] pot = new double[n];
        for (int p = 0; p < n; p++) {
            for (int q = 0; q < n; q++) {
                pot[p] += Math.exp(-alpha * dist2(u[p], u[q]));
            }
        }
        List<double[]> centres = new ArrayList<>();
        double first = -1;
        while (centres.size() < n) {
            int best = 0;
            for (int p = 1; p < n; p++) {
                if (pot[p] > pot[best]) {
                    best = p;
                }
            }
            if (first < 0) {
                first = pot[best];
            } else if (pot[best] < 0.15 * first) {
                break;
            }
            double pb = pot[best];
            double[] cu = u[best].clone();
            centres.add(cu);
            for (int p = 0; p < n; p++) {
                pot[p] -= pb * Math.exp(-beta * dist2(u[p], cu));
            }
        }
        double[][] c = new double[centres.size()][d];
        double[][] w = new double[centres.size()][d];
        for (int r = 0; r < c.length; r++) {
            for (int i = 0; i < d; i++) {
                double range = Math.max(1e-12, hi[i] - lo[i]);
                c[r][i] = lo[i] + centres.get(r)[i] * range;
                w[r][i] = radius * range / Math.sqrt(8) * (membership == Membership.TRIANGULAR ? 2 : 1);
            }
        }
        return new FuzzyInference(type, membership, c, w);
    }

    private static double dist2(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            double t = a[i] - b[i];
            s += t * t;
        }
        return s;
    }
}
