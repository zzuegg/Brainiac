package io.github.zzuegg.brainiac.experimental.survival;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Conformal lower predictive bounds (LPBs) on survival times: a bound {@code L(x)} with
 * {@code P(T >= L(X)) >= 1 - alpha} in finite samples, around any survival model's conditional
 * quantile {@code q_alpha(x)}.
 * <ul>
 *   <li>{@link Method#TYPE_I_LPB} (Candès, Lei and Ren, JRSS-B 2023,
 *       <a href="https://arxiv.org/abs/2103.09763">arXiv:2103.09763</a>): when every unit's censoring
 *       time {@code C} is known, the units with {@code C >= c0} reveal {@code min(T, c0)}. They differ
 *       from the population only by a covariate shift with likelihood ratio
 *       {@code 1 / P(C >= c0 | x)}, so weighted split conformal calibration of the score
 *       {@code q(x) - min(T, c0)} gives an LPB on {@code min(T, c0)}, which also bounds {@code T}.</li>
 *   <li>{@link Method#GENERAL_CENSORING_LPB}: only {@code min(T, C)} and the event flag are
 *       observed. The events of the calibration split are reweighted by inverse probability of
 *       censoring weights {@code 1 / G(T_i | x_i)} from a censoring model, which restores the
 *       distribution of {@code T} (under conditionally independent censoring), and calibrated as
 *       above. This follows the general-censoring setting of Davidov et al. (ICLR 2025) through
 *       IPCW calibration; it is valid when the censoring model is right.</li>
 *   <li>{@link Method#DOUBLY_ROBUST} (Sesia and Svetnik, ICML 2025,
 *       <a href="https://arxiv.org/abs/2412.09729">arXiv:2412.09729</a>): the censoring times of
 *       units with an observed event are imputed by sampling the censoring model conditional on
 *       {@code C > T}; the imputed type-I data go through {@link Method#TYPE_I_LPB}, and the result is
 *       capped by the model quantile {@code q(x)}. Coverage holds asymptotically if either the
 *       censoring model or the survival model is right.</li>
 *   <li>{@link Method#NAIVE}: split conformal on {@code q(x) - min(T, C)}, treating censoring
 *       times as event times: valid but conservative.</li>
 *   <li>{@link Method#UNCALIBRATED}: the model quantile itself.</li>
 * </ul>
 * Models are plugged in as {@link ConditionalSurvival} curves; {@link #cox} wraps
 * {@link SurvivalModels.Cox}.
 */
public final class ConformalSurvival {

    /** The calibration method. */
    public enum Method {
        /** The model's quantile, uncalibrated. */
        UNCALIBRATED,
        /** Split conformal treating censoring times as event times. */
        NAIVE,
        /** Weighted conformal with known censoring times (type-I censoring). */
        TYPE_I_LPB,
        /** IPCW-weighted conformal for general right censoring. */
        GENERAL_CENSORING_LPB,
        /** Imputed censoring times, then type-I calibration, capped by the model quantile. */
        DOUBLY_ROBUST
    }

    /** A conditional survival curve {@code S(t | x)}. */
    @FunctionalInterface
    public interface ConditionalSurvival {
        /**
         * @param x covariates
         * @param t a time
         * @return {@code P(T > t | x)}
         */
        double survival(double[] x, double t);

        /**
         * The conditional {@code tau}-quantile by bisection: the smallest {@code t} with
         * {@code S(t | x) <= 1 - tau}, or {@code horizon} if the curve never gets there.
         *
         * @param x covariates
         * @param tau the level
         * @param horizon the largest time searched
         * @return the quantile
         */
        default double quantile(double[] x, double tau, double horizon) {
            if (survival(x, horizon) > 1 - tau) {
                return horizon;
            }
            double lo = 0;
            double hi = horizon;
            for (int it = 0; it < 50; it++) {
                double mid = 0.5 * (lo + hi);
                if (survival(x, mid) > 1 - tau) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return hi;
        }
    }

    /**
     * Wraps a Cox model.
     *
     * @param cox the fitted model
     * @return its survival curve
     */
    public static ConditionalSurvival cox(SurvivalModels.Cox cox) {
        return cox::survival;
    }

    /**
     * The calibration data.
     *
     * @param x covariates
     * @param time observed times {@code min(T, C)}
     * @param event 1 if the event was observed, 0 if censored
     * @param censoring the censoring times when known for every unit (type-I), else null
     */
    public record Calibration(double[][] x, double[] time, int[] event, double[] censoring) {
    }

    private final Method method;
    private final ConditionalSurvival model;
    private final ConditionalSurvival censoringModel;
    private final double alpha;
    private final double horizon;
    private final double c0;
    private final double[] scores;
    private final double[] weights;

    private ConformalSurvival(Method method, ConditionalSurvival model, ConditionalSurvival censoringModel,
            double alpha, double horizon, double c0, double[] scores, double[] weights) {
        this.method = method;
        this.model = model;
        this.censoringModel = censoringModel;
        this.alpha = alpha;
        this.horizon = horizon;
        this.c0 = c0;
        this.scores = scores;
        this.weights = weights;
    }

    /**
     * Calibrates.
     *
     * @param method the method
     * @param model the survival model (fitted on a separate split)
     * @param censoringModel the censoring model {@code P(C > c | x)}; unused by UNCALIBRATED and NAIVE
     * @param cal the calibration split
     * @param alpha the miscoverage, such as 0.1
     * @param c0 the cutoff of the type-I and doubly robust methods (such as the median censoring time)
     * @param horizon the largest time any quantile search goes to
     * @param seed the imputation seed
     * @return the calibrated bound
     */
    public static ConformalSurvival fit(Method method, ConditionalSurvival model, ConditionalSurvival censoringModel,
            Calibration cal, double alpha, double c0, double horizon, long seed) {
        int n = cal.time().length;
        double[] q = new double[n];
        for (int i = 0; i < n; i++) {
            q[i] = model.quantile(cal.x()[i], alpha, horizon);
        }
        switch (method) {
            case UNCALIBRATED -> {
                return new ConformalSurvival(method, model, censoringModel, alpha, horizon, c0, new double[0],
                        new double[0]);
            }
            case NAIVE -> {
                double[] s = new double[n];
                double[] w = new double[n];
                for (int i = 0; i < n; i++) {
                    s[i] = q[i] - cal.time()[i];
                    w[i] = 1;
                }
                return new ConformalSurvival(method, model, censoringModel, alpha, horizon, c0, s, w);
            }
            case GENERAL_CENSORING_LPB -> {
                int k = 0;
                for (int e : cal.event()) {
                    k += e != 0 ? 1 : 0;
                }
                double[] s = new double[k];
                double[] w = new double[k];
                int j = 0;
                for (int i = 0; i < n; i++) {
                    if (cal.event()[i] != 0) {
                        s[j] = q[i] - cal.time()[i];
                        w[j++] = 1 / Math.max(0.05, censoringModel.survival(cal.x()[i], cal.time()[i] - 1e-9));
                    }
                }
                return new ConformalSurvival(method, model, censoringModel, alpha, horizon, c0, s, w);
            }
            case TYPE_I_LPB, DOUBLY_ROBUST -> {
                double[] c = new double[n];
                if (method == Method.TYPE_I_LPB) {
                    if (cal.censoring() == null) {
                        throw new IllegalArgumentException("TYPE_I_LPB needs every unit's censoring time");
                    }
                    c = cal.censoring();
                } else {
                    SplittableRandom r = new SplittableRandom(seed);
                    for (int i = 0; i < n; i++) {
                        c[i] = cal.event()[i] == 0 ? cal.time()[i]
                                : sampleAfter(censoringModel, cal.x()[i], cal.time()[i], horizon, r);
                    }
                }
                int k = 0;
                for (int i = 0; i < n; i++) {
                    k += c[i] >= c0 ? 1 : 0;
                }
                double[] s = new double[k];
                double[] w = new double[k];
                int j = 0;
                for (int i = 0; i < n; i++) {
                    if (c[i] >= c0) {
                        s[j] = q[i] - Math.min(cal.time()[i], c0);
                        w[j++] = 1 / Math.max(0.05, censoringModel.survival(cal.x()[i], c0));
                    }
                }
                return new ConformalSurvival(method, model, censoringModel, alpha, horizon, c0, s, w);
            }
            default -> throw new IllegalArgumentException(method.toString());
        }
    }

    /** Inverse-transform sampling of {@code C | C > after, x} from a censoring curve. */
    private static double sampleAfter(ConditionalSurvival g, double[] x, double after, double horizon,
            SplittableRandom r) {
        double sAfter = g.survival(x, after);
        if (sAfter <= 1e-12) {
            return after;
        }
        double target = sAfter * r.nextDouble();
        if (g.survival(x, horizon) > target) {
            return Double.POSITIVE_INFINITY;
        }
        double lo = after;
        double hi = horizon;
        for (int it = 0; it < 50; it++) {
            double mid = 0.5 * (lo + hi);
            if (g.survival(x, mid) > target) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return hi;
    }

    /**
     * The lower predictive bound of a test point.
     *
     * @param x covariates
     * @return {@code L(x)}, at least 0
     */
    public double lowerBound(double[] x) {
        double q = model.quantile(x, alpha, horizon);
        if (method == Method.UNCALIBRATED) {
            return q;
        }
        double wTest = switch (method) {
            case NAIVE -> 1;
            case GENERAL_CENSORING_LPB -> median(weights);
            default -> 1 / Math.max(0.05, censoringModel.survival(x, c0));
        };
        double eta = weightedQuantile(scores, weights, wTest, 1 - alpha);
        double bound = q - eta;
        if (method == Method.TYPE_I_LPB || method == Method.DOUBLY_ROBUST) {
            bound = Math.min(bound, c0);
        }
        if (method == Method.DOUBLY_ROBUST) {
            bound = Math.min(bound, q);
        }
        return Math.max(0, bound);
    }

    private static double median(double[] v) {
        double[] s = v.clone();
        Arrays.sort(s);
        return s.length == 0 ? 1 : s[s.length / 2];
    }

    /**
     * The weighted conformal quantile: the smallest score {@code s} with
     * {@code sum_{i: s_i <= s} p_i >= level}, where the test point carries weight {@code wTest} at
     * {@code +infinity}.
     */
    static double weightedQuantile(double[] scores, double[] weights, double wTest, double level) {
        int n = scores.length;
        Integer[] order = new Integer[n];
        double total = wTest;
        for (int i = 0; i < n; i++) {
            order[i] = i;
            total += weights[i];
        }
        Arrays.sort(order, (a, b) -> Double.compare(scores[a], scores[b]));
        double cum = 0;
        for (int k = 0; k < n; k++) {
            cum += weights[order[k]] / total;
            if (cum >= level) {
                return scores[order[k]];
            }
        }
        return Double.POSITIVE_INFINITY;
    }

    /**
     * Lower bounds of many points.
     *
     * @param x covariates
     * @return the bounds
     */
    public double[] lowerBounds(double[][] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = lowerBound(x[i]);
        }
        return out;
    }

    /** @return the method */
    public Method method() {
        return method;
    }
}
