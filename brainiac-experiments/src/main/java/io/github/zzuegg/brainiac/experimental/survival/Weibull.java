package io.github.zzuegg.brainiac.experimental.survival;

/**
 * The Weibull distribution in the rate parametrisation used across the repo:
 * {@code S(t) = exp(-lambda t^k)}, hazard {@code h(t) = lambda k t^(k-1)}. A shared helper (wave
 * 11, item 11.3.17); the private copies in {@code SurvivalData}, {@code OpenMax} and
 * {@code PointProcessShootout} are left as they are.
 *
 * @param lambda the rate {@code lambda > 0}
 * @param k the shape {@code k > 0}
 */
public record Weibull(double lambda, double k) {

    /**
     * Validates the parameters.
     *
     * @param lambda the rate
     * @param k the shape
     */
    public Weibull {
        if (!(lambda > 0) || !(k > 0)) {
            throw new IllegalArgumentException("lambda and k must be positive: " + lambda + ", " + k);
        }
    }

    /**
     * @param t a time, {@code t > 0}
     * @return the density
     */
    public double density(double t) {
        return hazard(t) * survival(t);
    }

    /**
     * @param t a time
     * @return {@code log f(t)}
     */
    public double logDensity(double t) {
        return Math.log(lambda) + Math.log(k) + (k - 1) * Math.log(t) - lambda * Math.pow(t, k);
    }

    /**
     * @param t a time
     * @return {@code S(t)}
     */
    public double survival(double t) {
        return Math.exp(-lambda * Math.pow(t, k));
    }

    /**
     * @param t a time
     * @return {@code h(t)}
     */
    public double hazard(double t) {
        return lambda * k * Math.pow(t, k - 1);
    }

    /**
     * @param p a probability in {@code (0, 1)}
     * @return the {@code p}-quantile {@code (-log(1 - p) / lambda)^(1/k)}
     */
    public double quantile(double p) {
        return Math.pow(-Math.log1p(-p) / lambda, 1 / k);
    }

    /** @return the median */
    public double median() {
        return quantile(0.5);
    }

    /**
     * The right-censored log-likelihood {@code sum event log h(t) + log S(t)}.
     *
     * @param time observed times
     * @param event 1 (or any non-zero cause) for an event, 0 for censored
     * @return the log-likelihood
     */
    public double logLikelihood(double[] time, int[] event) {
        double ll = 0;
        for (int i = 0; i < time.length; i++) {
            ll += -lambda * Math.pow(time[i], k);
            if (event[i] != 0) {
                ll += Math.log(lambda) + Math.log(k) + (k - 1) * Math.log(time[i]);
            }
        }
        return ll;
    }

    /**
     * The censored maximum-likelihood fit: for a fixed shape the rate is
     * {@code events / sum t^k}, and the profile likelihood is maximised over {@code log k} by
     * golden-section search on {@code [-4, 4]}.
     *
     * @param time observed times
     * @param event event indicators
     * @return the fit
     */
    public static Weibull fit(double[] time, int[] event) {
        int events = 0;
        for (int e : event) {
            events += e != 0 ? 1 : 0;
        }
        if (events == 0) {
            throw new IllegalArgumentException("no events");
        }
        final int d = events;
        java.util.function.DoubleUnaryOperator profile = logK -> {
            double k = Math.exp(logK);
            double s = 0;
            for (double t : time) {
                s += Math.pow(t, k);
            }
            return new Weibull(d / s, k).logLikelihood(time, event);
        };
        double lo = -4;
        double hi = 4;
        double g = (Math.sqrt(5) - 1) / 2;
        double a = hi - g * (hi - lo);
        double b = lo + g * (hi - lo);
        double fa = profile.applyAsDouble(a);
        double fb = profile.applyAsDouble(b);
        for (int it = 0; it < 80; it++) {
            if (fa < fb) {
                lo = a;
                a = b;
                fa = fb;
                b = lo + g * (hi - lo);
                fb = profile.applyAsDouble(b);
            } else {
                hi = b;
                b = a;
                fb = fa;
                a = hi - g * (hi - lo);
                fa = profile.applyAsDouble(a);
            }
        }
        double k = Math.exp((lo + hi) / 2);
        double s = 0;
        for (double t : time) {
            s += Math.pow(t, k);
        }
        return new Weibull(d / s, k);
    }
}
