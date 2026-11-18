package io.github.zzuegg.brainiac.experimental.eval;

/**
 * Unbiased pass-rate estimates from an imperfect judge plus a small human-labelled calibration
 * set. The judge scores {@code N} unlabelled outputs ({@code f}); on {@code n} other outputs both
 * the judge ({@code fl}) and a human ({@code y}) give a 0/1 verdict.
 * <ul>
 *   <li>{@link #judgeOnly}: the raw judge mean, biased whenever sensitivity or specificity is
 *       below 1;</li>
 *   <li>{@link #humanOnly}: the mean of the gold labels, unbiased but wide;</li>
 *   <li>{@link #roganGladen}: estimates sensitivity and specificity on the labelled set and
 *       inverts the misclassification, {@code θ = (p̂ + spec - 1) / (sens + spec - 1)}, with a
 *       delta-method interval; valid under a shift of the pass rate between calibration and
 *       test sets, because sensitivity and specificity are conditional on the truth;</li>
 *   <li>{@link #ppi}: prediction-powered inference (Angelopoulos et al., Science 2023,
 *       <a href="https://arxiv.org/abs/2301.09633">arXiv:2301.09633</a>): the judge mean on the
 *       unlabelled set plus the mean human-minus-judge residual ("rectifier") on the labelled
 *       set;</li>
 *   <li>{@link #ppiPlusPlus}: PPI++ (Angelopoulos et al., 2023,
 *       <a href="https://arxiv.org/abs/2311.01453">arXiv:2311.01453</a>): the judge term
 *       weighted by {@code λ = Cov(y, f) / ((1 + n/N) Var(f))}, the variance-minimising
 *       weight, so it never does worse than human-only asymptotically.</li>
 * </ul>
 * Chen et al. (<a href="https://arxiv.org/abs/2511.21140">arXiv:2511.21140</a>) compare PPI with
 * Rogan–Gladen; Lee et al. (<a href="https://arxiv.org/abs/2601.05420">arXiv:2601.05420</a>)
 * build intervals over both data sources. Intervals here are Wald intervals at the given z.
 */
public final class JudgeCorrection {

    private JudgeCorrection() {
    }

    /**
     * A point estimate with a normal-approximation interval.
     *
     * @param estimate the estimate
     * @param se its standard error
     * @param low the lower end of the interval
     * @param high the upper end of the interval
     */
    public record Estimate(double estimate, double se, double low, double high) {
        static Estimate of(double est, double se, double z) {
            return new Estimate(est, se, est - z * se, est + z * se);
        }

        /**
         * @param truth the true value
         * @return whether the interval covers it
         */
        public boolean covers(double truth) {
            return low <= truth && truth <= high;
        }

        /** @return the interval width */
        public double width() {
            return high - low;
        }
    }

    private static double mean(double[] x) {
        double s = 0;
        for (double v : x) {
            s += v;
        }
        return s / x.length;
    }

    private static double variance(double[] x, double mean) {
        double s = 0;
        for (double v : x) {
            s += (v - mean) * (v - mean);
        }
        return s / Math.max(1, x.length - 1);
    }

    /**
     * The raw judge mean on the unlabelled outputs.
     *
     * @param f judge verdicts (0/1 or scores)
     * @param z the normal quantile, 1.96 for 95%
     * @return the estimate
     */
    public static Estimate judgeOnly(double[] f, double z) {
        double m = mean(f);
        return Estimate.of(m, Math.sqrt(variance(f, m) / f.length), z);
    }

    /**
     * The mean of the human labels alone.
     *
     * @param y gold labels
     * @param z the normal quantile
     * @return the estimate
     */
    public static Estimate humanOnly(double[] y, double z) {
        return judgeOnly(y, z);
    }

    /**
     * Rogan–Gladen with a delta-method interval:
     * {@code Var θ ≈ [Var p̂ + θ² Var sens + (1 - θ)² Var spec] / (sens + spec - 1)²}.
     *
     * @param f judge verdicts on the unlabelled outputs, 0/1
     * @param fl judge verdicts on the labelled outputs, 0/1
     * @param y human labels of the labelled outputs, 0/1
     * @param z the normal quantile
     * @return the estimate, clipped to [0, 1]
     */
    public static Estimate roganGladen(double[] f, double[] fl, double[] y, double z) {
        double p = mean(f);
        int pos = 0;
        int neg = 0;
        double tp = 0;
        double tn = 0;
        for (int i = 0; i < y.length; i++) {
            if (y[i] > 0.5) {
                pos++;
                tp += fl[i];
            } else {
                neg++;
                tn += 1 - fl[i];
            }
        }
        double sens = pos == 0 ? 1 : tp / pos;
        double spec = neg == 0 ? 1 : tn / neg;
        double j = Math.max(1e-3, sens + spec - 1);
        double theta = Math.min(1, Math.max(0, (p + spec - 1) / j));
        double varP = p * (1 - p) / f.length;
        double varSens = pos == 0 ? 0 : sens * (1 - sens) / pos;
        double varSpec = neg == 0 ? 0 : spec * (1 - spec) / neg;
        double var = (varP + theta * theta * varSens + (1 - theta) * (1 - theta) * varSpec) / (j * j);
        return Estimate.of(theta, Math.sqrt(var), z);
    }

    /**
     * Prediction-powered inference for a mean.
     *
     * @param f judge scores on the unlabelled outputs
     * @param fl judge scores on the labelled outputs
     * @param y human labels of the labelled outputs
     * @param z the normal quantile
     * @return the estimate
     */
    public static Estimate ppi(double[] f, double[] fl, double[] y, double z) {
        return weighted(f, fl, y, 1, z);
    }

    /**
     * PPI++ with the variance-minimising weight on the judge term.
     *
     * @param f judge scores on the unlabelled outputs
     * @param fl judge scores on the labelled outputs
     * @param y human labels of the labelled outputs
     * @param z the normal quantile
     * @return the estimate
     */
    public static Estimate ppiPlusPlus(double[] f, double[] fl, double[] y, double z) {
        return weighted(f, fl, y, lambda(f, fl, y), z);
    }

    /**
     * The PPI++ weight {@code Cov(y, f) / ((1 + n/N) Var(f))}, clipped to [0, 1].
     *
     * @param f judge scores on the unlabelled outputs
     * @param fl judge scores on the labelled outputs
     * @param y human labels
     * @return λ
     */
    public static double lambda(double[] f, double[] fl, double[] y) {
        double my = mean(y);
        double mf = mean(fl);
        double cov = 0;
        for (int i = 0; i < y.length; i++) {
            cov += (y[i] - my) * (fl[i] - mf);
        }
        cov /= Math.max(1, y.length - 1);
        double[] all = new double[f.length + fl.length];
        System.arraycopy(f, 0, all, 0, f.length);
        System.arraycopy(fl, 0, all, f.length, fl.length);
        double varF = variance(all, mean(all));
        if (varF == 0) {
            return 0;
        }
        return Math.max(0, Math.min(1, cov / ((1 + (double) y.length / f.length) * varF)));
    }

    /**
     * The λ-weighted PPI estimate {@code λ mean(f) + mean(y - λ fl)}.
     *
     * @param f judge scores on the unlabelled outputs
     * @param fl judge scores on the labelled outputs
     * @param y human labels
     * @param lambda the judge weight (1 is PPI, 0 is human-only)
     * @param z the normal quantile
     * @return the estimate
     */
    public static Estimate weighted(double[] f, double[] fl, double[] y, double lambda, double z) {
        double mf = mean(f);
        double[] r = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            r[i] = y[i] - lambda * fl[i];
        }
        double mr = mean(r);
        double var = lambda * lambda * variance(f, mf) / f.length + variance(r, mr) / y.length;
        return Estimate.of(lambda * mf + mr, Math.sqrt(var), z);
    }
}
