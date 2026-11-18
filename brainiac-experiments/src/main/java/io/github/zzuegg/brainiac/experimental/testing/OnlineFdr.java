package io.github.zzuegg.brainiac.experimental.testing;

/**
 * Online false-discovery-rate control: hypotheses arrive one at a time and each must be
 * accepted or rejected before the next is seen, with {@code FDR <= alpha} at every time.
 *
 * <ul>
 *   <li>{@link Method#LOND} (Javanmard and Montanari 2018): level {@code alpha gamma_t (R + 1)}.</li>
 *   <li>{@link Method#LORD_PLUS_PLUS} (Ramdas et al., NeurIPS 2017): alpha-wealth that every
 *       rejection refills, spent along the sequence {@code gamma}.</li>
 *   <li>{@link Method#SAFFRON} (Ramdas et al., ICML 2018,
 *       <a href="https://arxiv.org/abs/1802.09098">arXiv:1802.09098</a>): adapts to the null
 *       share by only charging wealth for "candidates" ({@code p <= lambda}).</li>
 *   <li>{@link Method#E_LORD}/{@link Method#E_SAFFRON} (e-GAI, Zhang, Liu and Ren, ICML 2025,
 *       <a href="https://arxiv.org/abs/2506.01452">arXiv:2506.01452</a>): on e-values, reject when
 *       {@code e_t >= 1 / alpha_t} with {@code alpha_t = omega_t (alpha - sum_j alpha_j /
 *       (R_{j-1} + 1)) (R_{t-1} + 1)} (e-SAFFRON charges only tests with
 *       {@code e_j < 1/lambda} and starts from {@code alpha (1 - lambda)}), where the invested
 *       fraction {@code omega_t} grows after acceptances and shrinks after rejections. Valid
 *       under arbitrary dependence.</li>
 * </ul>
 */
public final class OnlineFdr {

    /** The procedure. */
    public enum Method {
        /** Significance levels based on the number of discoveries. */
        LOND,
        /** LORD++. */
        LORD_PLUS_PLUS,
        /** SAFFRON with {@code lambda = 1/2}. */
        SAFFRON,
        /** e-LORD on e-values. */
        E_LORD,
        /** e-SAFFRON on e-values with {@code lambda = 0.1}. */
        E_SAFFRON
    }

    private OnlineFdr() {
    }

    /**
     * Runs a procedure over a stream.
     *
     * @param method the procedure
     * @param values p-values in arrival order (e-values for the e-methods)
     * @param alpha the target FDR
     * @return the rejections
     */
    public static boolean[] run(Method method, double[] values, double alpha) {
        return switch (method) {
            case LOND -> lond(values, alpha);
            case LORD_PLUS_PLUS -> lord(values, alpha);
            case SAFFRON -> saffron(values, alpha, 0.5);
            case E_LORD -> eGai(values, alpha, 0, 0.05, 0.5, 0.5);
            case E_SAFFRON -> eGai(values, alpha, 0.1, 0.05, 0.5, 0.5);
        };
    }

    /**
     * The LORD spending sequence {@code gamma_j ~ log(max(j, 2)) / (j exp(sqrt(log j)))},
     * normalised to sum to 1 over the horizon.
     *
     * @param m the horizon
     * @return {@code gamma[1..m]} (index 0 unused, 0)
     */
    public static double[] lordGamma(int m) {
        double[] g = new double[m + 1];
        double s = 0;
        for (int j = 1; j <= m; j++) {
            g[j] = Math.log(Math.max(j, 2)) / (j * Math.exp(Math.sqrt(Math.log(j))));
            s += g[j];
        }
        for (int j = 1; j <= m; j++) {
            g[j] /= s;
        }
        return g;
    }

    private static double[] powerGamma(int m, double exponent) {
        double[] g = new double[m + 1];
        double s = 0;
        for (int j = 1; j <= m; j++) {
            g[j] = Math.pow(j, -exponent);
            s += g[j];
        }
        for (int j = 1; j <= m; j++) {
            g[j] /= s;
        }
        return g;
    }

    private static boolean[] lond(double[] p, double alpha) {
        int m = p.length;
        double[] g = lordGamma(m);
        boolean[] out = new boolean[m];
        int r = 0;
        for (int t = 1; t <= m; t++) {
            if (p[t - 1] <= alpha * g[t] * (r + 1)) {
                out[t - 1] = true;
                r++;
            }
        }
        return out;
    }

    private static boolean[] lord(double[] p, double alpha) {
        int m = p.length;
        double[] g = lordGamma(m);
        boolean[] out = new boolean[m];
        double w0 = alpha / 2;
        int[] tau = new int[m];
        int r = 0;
        for (int t = 1; t <= m; t++) {
            double a = g[t] * w0;
            for (int j = 0; j < r; j++) {
                a += (j == 0 ? alpha - w0 : alpha) * g[t - tau[j]];
            }
            if (p[t - 1] <= a) {
                out[t - 1] = true;
                tau[r++] = t;
            }
        }
        return out;
    }

    private static boolean[] saffron(double[] p, double alpha, double lambda) {
        int m = p.length;
        double[] g = powerGamma(m, 1.6);
        boolean[] out = new boolean[m];
        double w0 = alpha / 2;
        int[] tau = new int[m];
        int r = 0;
        boolean[] candidate = new boolean[m + 1];
        int[] candidatesBefore = new int[m + 2]; // prefix counts of candidates among 1..t
        for (int t = 1; t <= m; t++) {
            int c0 = candidatesBefore[t - 1];
            double a = w0 * g[Math.max(1, t - c0)];
            for (int j = 0; j < r; j++) {
                int cj = candidatesBefore[t - 1] - candidatesBefore[tau[j]];
                a += (j == 0 ? alpha - w0 : alpha) * g[Math.max(1, t - tau[j] - cj)];
            }
            a = Math.min(lambda, (1 - lambda) * a);
            candidate[t] = p[t - 1] <= lambda;
            candidatesBefore[t] = candidatesBefore[t - 1] + (candidate[t] ? 1 : 0);
            if (p[t - 1] <= a) {
                out[t - 1] = true;
                tau[r++] = t;
            }
        }
        return out;
    }

    /**
     * The e-GAI rule (e-LORD when {@code lambda = 0}, e-SAFFRON otherwise).
     *
     * @param e e-values
     * @param alpha the target FDR
     * @param lambda the candidate threshold (tests with {@code e < 1/lambda} are charged)
     * @param omega1 the initial invested fraction
     * @param phi the growth base after acceptances
     * @param psi the decay base after rejections
     * @return the rejections
     */
    public static boolean[] eGai(double[] e, double alpha, double lambda, double omega1, double phi, double psi) {
        int m = e.length;
        boolean[] out = new boolean[m];
        double omega = omega1;
        double spent = 0;
        int r = 0;
        double budget = lambda > 0 ? alpha * (1 - lambda) : alpha;
        for (int t = 1; t <= m; t++) {
            double at = omega * Math.max(0, budget - spent) * (r + 1);
            boolean reject = at > 0 && e[t - 1] >= 1 / at;
            boolean charged = lambda <= 0 || e[t - 1] < 1 / lambda;
            if (charged) {
                spent += at / (r + 1);
            }
            if (reject) {
                out[t - 1] = true;
                r++;
            }
            omega = omega + omega1 * (reject ? -Math.pow(psi, r) : Math.pow(phi, t - r));
            omega = Math.min(0.99, Math.max(1e-6, omega));
        }
        return out;
    }
}
