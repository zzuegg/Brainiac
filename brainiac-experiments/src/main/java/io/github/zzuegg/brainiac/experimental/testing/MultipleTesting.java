package io.github.zzuegg.brainiac.experimental.testing;

import java.util.Arrays;

/**
 * Batch multiple-testing procedures. Family-wise error control: Bonferroni and Holm's
 * step-down. False discovery rate control: Benjamini-Hochberg (JRSS-B 1995; valid under
 * independence and PRDS), Benjamini-Yekutieli (Annals 2001; any dependence, at the price of a
 * {@code sum 1/k ~ log m} factor), Storey's adaptive BH (JRSS-B 2002), which estimates the null
 * share {@code pi0} from the p-values above {@code 1/2} for more power, and e-BH (Wang and
 * Ramdas, JRSS-B 2022, <a href="https://arxiv.org/abs/2009.02824">arXiv:2009.02824</a>), the
 * e-value counterpart of BH that controls the FDR under arbitrary dependence.
 */
public final class MultipleTesting {

    /** The procedure. */
    public enum Method {
        /** Reject {@code p <= alpha / m}. */
        BONFERRONI,
        /** Holm's step-down. */
        HOLM,
        /** Benjamini-Hochberg step-up. */
        BH,
        /** Benjamini-Yekutieli step-up. */
        BY,
        /** Storey's adaptive BH with {@code lambda = 1/2}. */
        STOREY_BH,
        /** e-BH on e-values. */
        E_BH,
        /** No correction: reject {@code p <= alpha}. */
        UNCORRECTED
    }

    private MultipleTesting() {
    }

    /**
     * Applies a p-value procedure ({@link Method#E_BH} expects e-values instead).
     *
     * @param method the procedure
     * @param values p-values, or e-values for e-BH
     * @param alpha the level
     * @return the rejections
     */
    public static boolean[] reject(Method method, double[] values, double alpha) {
        int m = values.length;
        boolean[] out = new boolean[m];
        switch (method) {
            case UNCORRECTED -> {
                for (int i = 0; i < m; i++) {
                    out[i] = values[i] <= alpha;
                }
            }
            case BONFERRONI -> {
                for (int i = 0; i < m; i++) {
                    out[i] = values[i] <= alpha / m;
                }
            }
            case HOLM -> {
                Integer[] o = order(values, false);
                for (int k = 0; k < m; k++) {
                    if (values[o[k]] > alpha / (m - k)) {
                        break;
                    }
                    out[o[k]] = true;
                }
            }
            case BH -> stepUp(values, alpha, out);
            case BY -> {
                double c = 0;
                for (int k = 1; k <= m; k++) {
                    c += 1.0 / k;
                }
                stepUp(values, alpha / c, out);
            }
            case STOREY_BH -> stepUp(values, alpha / storeyPi0(values, 0.5), out);
            case E_BH -> {
                Integer[] o = order(values, true);
                int kStar = 0;
                for (int k = 1; k <= m; k++) {
                    if (values[o[k - 1]] >= m / (alpha * k)) {
                        kStar = k;
                    }
                }
                for (int k = 0; k < kStar; k++) {
                    out[o[k]] = true;
                }
            }
            default -> throw new IllegalArgumentException(String.valueOf(method));
        }
        return out;
    }

    /**
     * BH adjusted rejections at a given level with an explicit null-share estimate.
     *
     * @param p the p-values
     * @param alpha the level
     * @param pi0 the null share used to inflate the level
     * @return the rejections
     */
    public static boolean[] benjaminiHochberg(double[] p, double alpha, double pi0) {
        boolean[] out = new boolean[p.length];
        stepUp(p, alpha / Math.max(pi0, 1e-9), out);
        return out;
    }

    /**
     * Storey's estimate {@code (1 + #{p > lambda}) / (m (1 - lambda))}, capped at 1.
     *
     * @param p the p-values
     * @param lambda the tuning level
     * @return the null-share estimate
     */
    public static double storeyPi0(double[] p, double lambda) {
        int above = 0;
        for (double v : p) {
            if (v > lambda) {
                above++;
            }
        }
        return Math.min(1, (1.0 + above) / (p.length * (1 - lambda)));
    }

    private static void stepUp(double[] p, double alpha, boolean[] out) {
        int m = p.length;
        Integer[] o = order(p, false);
        int kStar = 0;
        for (int k = 1; k <= m; k++) {
            if (p[o[k - 1]] <= alpha * k / m) {
                kStar = k;
            }
        }
        for (int k = 0; k < kStar; k++) {
            out[o[k]] = true;
        }
    }

    private static Integer[] order(double[] v, boolean descending) {
        Integer[] o = new Integer[v.length];
        for (int i = 0; i < o.length; i++) {
            o[i] = i;
        }
        Arrays.sort(o, (a, b) -> descending ? Double.compare(v[b], v[a]) : Double.compare(v[a], v[b]));
        return o;
    }

    /**
     * The false discovery proportion and true-positive share of a rejection set.
     *
     * @param rejected the rejections
     * @param isNull whether each hypothesis is null
     * @return {@code {FDP, power}}
     */
    public static double[] fdpAndPower(boolean[] rejected, boolean[] isNull) {
        int r = 0;
        int v = 0;
        int alt = 0;
        int tp = 0;
        for (int i = 0; i < rejected.length; i++) {
            if (!isNull[i]) {
                alt++;
            }
            if (rejected[i]) {
                r++;
                if (isNull[i]) {
                    v++;
                } else {
                    tp++;
                }
            }
        }
        return new double[] {r == 0 ? 0 : (double) v / r, alt == 0 ? 0 : (double) tp / alt};
    }
}
