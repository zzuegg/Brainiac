package io.github.zzuegg.brainiac.experimental.sensing;

import io.github.zzuegg.brainiac.tensor.Matrices;

/**
 * Adaptive FIR filters {@code y(n) = w^T u(n)}, {@code u(n) = [x(n), ..., x(n-N+1)]}, updated from
 * the a-priori error {@code e(n) = d(n) - w^T u(n)} (Haykin, Adaptive Filter Theory):
 * <ul>
 *   <li>{@link Method#LMS} (Widrow &amp; Hoff 1960): {@code w += mu e u};</li>
 *   <li>{@link Method#NLMS}: {@code w += mu e u / (delta + |u|^2)};</li>
 *   <li>{@link Method#APA} (Ozeki &amp; Umeda 1984): the last {@code K} regressors
 *       {@code U}, {@code w += mu U (U^T U + delta I)^-1 e_K};</li>
 *   <li>{@link Method#RLS}: exponentially weighted least squares with forgetting
 *       {@code lambda}, {@code O(N^2)} per sample;</li>
 *   <li>{@link Method#KALMAN_FIR}: the filter as the state of a random walk with variance
 *       {@code q}, observed through {@code d = u^T w + v}; the observation variance is tracked from
 *       the error power (as in Enzner &amp; Vary's frequency-domain Kalman echo canceller, here in
 *       the time domain), so the gain shrinks during double-talk.</li>
 * </ul>
 * {@link #geigel} is the Geigel double-talk detector: talk is declared when
 * {@code |mic(n)| >= threshold max_k |far(n - k)|} over a window, held for a while.
 */
public final class AdaptiveFilters {

    /** The update rules. */
    public enum Method {
        /** Least mean squares. */
        LMS,
        /** Normalised LMS. */
        NLMS,
        /** Affine projection. */
        APA,
        /** Recursive least squares. */
        RLS,
        /** Kalman filter on the taps. */
        KALMAN_FIR
    }

    /**
     * Step sizes and regularisation.
     *
     * @param mu the LMS/NLMS/APA step
     * @param projections APA's {@code K}
     * @param delta the regularisation (NLMS, APA) or RLS's initial {@code P = I / delta}
     * @param lambda RLS's forgetting factor
     * @param q the Kalman random-walk variance per tap
     * @param r0 the Kalman filter's initial observation variance
     */
    public record Settings(double mu, int projections, double delta, double lambda, double q, double r0) {
    }

    /** One filter's state. */
    public static final class Filter {

        private final Method method;
        private final Settings settings;
        private final int taps;
        private final double[] w;
        private final double[] u;
        private final double[][] p;
        private final double[][] history;
        private final double[] dHistory;
        private double obsVariance;
        private int seen;

        /**
         * Creates a zero filter.
         *
         * @param method the rule
         * @param taps {@code N}
         * @param settings the settings
         */
        public Filter(Method method, int taps, Settings settings) {
            this.method = method;
            this.settings = settings;
            this.taps = taps;
            this.w = new double[taps];
            this.u = new double[taps];
            boolean matrix = method == Method.RLS || method == Method.KALMAN_FIR;
            this.p = matrix ? new double[taps][taps] : null;
            if (matrix) {
                double init = method == Method.RLS ? 1 / settings.delta() : 1.0 / taps;
                for (int i = 0; i < taps; i++) {
                    p[i][i] = init;
                }
            }
            int k = method == Method.APA ? settings.projections() : 0;
            this.history = new double[k][];
            this.dHistory = new double[k];
            this.obsVariance = settings.r0();
        }

        /** @return a copy of the taps */
        public double[] weights() {
            return w.clone();
        }

        /**
         * Feeds one sample: shifts {@code x} into the regressor, returns the a-priori error and
         * adapts unless frozen.
         *
         * @param x the new input sample
         * @param d the desired sample
         * @param adapt whether to update the taps
         * @return {@code e = d - w^T u}
         */
        public double step(double x, double d, boolean adapt) {
            System.arraycopy(u, 0, u, 1, taps - 1);
            u[0] = x;
            seen++;
            double e = d - dot(w, u);
            if (!adapt) {
                return e;
            }
            switch (method) {
                case LMS -> axpy(w, settings.mu() * e, u);
                case NLMS -> axpy(w, settings.mu() * e / (settings.delta() + dot(u, u)), u);
                case APA -> apa(d);
                case RLS -> rls(e);
                case KALMAN_FIR -> kalman(e);
                default -> throw new IllegalStateException();
            }
            return e;
        }

        private void apa(double d) {
            int k = history.length;
            for (int i = k - 1; i > 0; i--) {
                history[i] = history[i - 1];
                dHistory[i] = dHistory[i - 1];
            }
            history[0] = u.clone();
            dHistory[0] = d;
            int used = Math.min(k, seen);
            double[][] g = new double[used][used];
            double[] err = new double[used];
            for (int i = 0; i < used; i++) {
                err[i] = dHistory[i] - dot(w, history[i]);
                for (int j = 0; j < used; j++) {
                    g[i][j] = dot(history[i], history[j]) + (i == j ? settings.delta() : 0);
                }
            }
            double[][] inv = Matrices.inverse(g);
            for (int i = 0; i < used; i++) {
                double c = 0;
                for (int j = 0; j < used; j++) {
                    c += inv[i][j] * err[j];
                }
                axpy(w, settings.mu() * c, history[i]);
            }
        }

        private double[] pu() {
            double[] pu = new double[taps];
            for (int i = 0; i < taps; i++) {
                pu[i] = dot(p[i], u);
            }
            return pu;
        }

        private void rls(double e) {
            double[] pu = pu();
            double den = settings.lambda() + dot(u, pu);
            double il = 1 / settings.lambda();
            for (int i = 0; i < taps; i++) {
                double ki = pu[i] / den;
                w[i] += ki * e;
                for (int j = 0; j < taps; j++) {
                    p[i][j] = (p[i][j] - ki * pu[j]) * il;
                }
            }
        }

        private void kalman(double e) {
            double[] pu = pu();
            double upu = dot(u, pu);
            obsVariance = 0.99 * obsVariance + 0.01 * Math.max(e * e - upu, 0);
            double den = upu + Math.max(obsVariance, 1e-6);
            for (int i = 0; i < taps; i++) {
                double ki = pu[i] / den;
                w[i] += ki * e;
                for (int j = 0; j < taps; j++) {
                    p[i][j] -= ki * pu[j];
                }
                p[i][i] += settings.q();
            }
        }
    }

    private AdaptiveFilters() {
    }

    /**
     * Runs a filter over a record.
     *
     * @param method the rule
     * @param settings the settings
     * @param taps {@code N}
     * @param input the input
     * @param desired the desired signal
     * @param freeze where adaptation is frozen (double-talk), or {@code null}
     * @param truth the true taps over time, for the misalignment trace, or {@code null}
     * @return the a-priori errors and, with {@code truth}, the misalignment
     *     {@code |w - w*|^2 / |w*|^2} after each sample
     */
    public static double[][] run(Method method, Settings settings, int taps, double[] input, double[] desired,
            boolean[] freeze, double[][] truth) {
        Filter f = new Filter(method, taps, settings);
        int n = input.length;
        double[] err = new double[n];
        double[] mis = truth == null ? null : new double[n];
        for (int i = 0; i < n; i++) {
            err[i] = f.step(input[i], desired[i], freeze == null || !freeze[i]);
            if (mis != null) {
                mis[i] = misalignment(f.w, truth[i]);
            }
        }
        return new double[][] {err, mis};
    }

    /**
     * The normalised misalignment.
     *
     * @param w the estimate
     * @param truth the true taps
     * @return {@code |w - w*|^2 / |w*|^2}
     */
    public static double misalignment(double[] w, double[] truth) {
        double num = 0;
        double den = 0;
        for (int i = 0; i < truth.length; i++) {
            double d = (i < w.length ? w[i] : 0) - truth[i];
            num += d * d;
            den += truth[i] * truth[i];
        }
        return num / den;
    }

    /**
     * The Geigel double-talk detector.
     *
     * @param far the far-end signal
     * @param mic the microphone signal
     * @param window the far-end window (at least the echo path length)
     * @param threshold the ratio, typically 0.5 (6 dB loss)
     * @param hold samples to hold a detection
     * @return true where double-talk is declared
     */
    public static boolean[] geigel(double[] far, double[] mic, int window, double threshold, int hold) {
        int n = far.length;
        boolean[] talk = new boolean[n];
        int left = 0;
        java.util.ArrayDeque<Integer> maxima = new java.util.ArrayDeque<>();
        for (int i = 0; i < n; i++) {
            while (!maxima.isEmpty() && Math.abs(far[maxima.peekLast()]) <= Math.abs(far[i])) {
                maxima.pollLast();
            }
            maxima.addLast(i);
            if (maxima.peekFirst() <= i - window) {
                maxima.pollFirst();
            }
            double peak = Math.abs(far[maxima.peekFirst()]);
            if (Math.abs(mic[i]) >= threshold * peak && peak > 0) {
                left = hold;
            }
            talk[i] = left > 0;
            if (left > 0) {
                left--;
            }
        }
        return talk;
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static void axpy(double[] y, double a, double[] x) {
        for (int i = 0; i < y.length; i++) {
            y[i] += a * x[i];
        }
    }
}
