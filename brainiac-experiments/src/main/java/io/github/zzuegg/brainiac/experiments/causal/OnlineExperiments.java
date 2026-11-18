package io.github.zzuegg.brainiac.experiments.causal;

import io.github.zzuegg.brainiac.experimental.causal.OnlineExperiments.Msprt;
import io.github.zzuegg.brainiac.experimental.causal.OnlineExperiments.Test;
import io.github.zzuegg.brainiac.experimental.causal.Regressor;
import io.github.zzuegg.brainiac.experimental.data.CausalScenarios;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Online A/B testing on simulated revenue streams (W9 1.15, {@link CausalScenarios#abTest}):
 * <ul>
 *   <li>{@link #peeking}: the false-positive rate under the null (lift 0) of a fixed-horizon test
 *       read once, the same test checked every day (stop at the first {@code p < alpha}), and the
 *       always-valid mixture SPRT monitored after every user;</li>
 *   <li>{@link #power}: the power to detect a lift, and the relative sample size needed for 80 %
 *       power, of the plain difference in means against CUPED (pre-period covariate) and CUPAC (a
 *       regression on pre-period features trained on a separate pre-period sample), and the share
 *       of tests the mSPRT stops early.</li>
 * </ul>
 */
public final class OnlineExperiments {

    private static final double ALPHA = 0.05;

    private OnlineExperiments() {
    }

    /**
     * False-positive rates under the null.
     *
     * @param simulations simulated tests
     * @param users users per test
     * @param days days (looks) per test
     * @param seed the base seed
     * @return procedure name to its rejection rate
     */
    public static Map<String, Double> peeking(int simulations, int users, int days, long seed) {
        int fixed = 0;
        int peek = 0;
        int msprt = 0;
        for (int s = 0; s < simulations; s++) {
            CausalScenarios.AbTest ab = CausalScenarios.abTest(users, 0.5, 0, days, seed + s);
            fixed += io.github.zzuegg.brainiac.experimental.causal.OnlineExperiments.tTest(ab.y(), ab.t(), users)
                    .pValue() < ALPHA ? 1 : 0;
            peek += peeked(ab, days) ? 1 : 0;
            msprt += sequential(ab, 0.3) < users ? 1 : 0;
        }
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("fixed horizon t-test", fixed / (double) simulations);
        out.put("t-test peeked daily", peek / (double) simulations);
        out.put("mSPRT (always valid)", msprt / (double) simulations);
        return out;
    }

    private static boolean peeked(CausalScenarios.AbTest ab, int days) {
        int n = ab.size();
        for (int d = 1; d <= days; d++) {
            int upTo = (int) ((long) n * d / days);
            Test t = io.github.zzuegg.brainiac.experimental.causal.OnlineExperiments.tTest(ab.y(), ab.t(), upTo);
            if (t.pValue() < ALPHA) {
                return true;
            }
        }
        return false;
    }

    /** @return the number of users after which the mSPRT rejected, or {@code n} if it never did */
    private static int sequential(CausalScenarios.AbTest ab, double tau) {
        Msprt test = new Msprt(tau, 50);
        for (int i = 0; i < ab.size(); i++) {
            if (test.add(ab.y()[i], ab.t()[i]) < ALPHA) {
                return i + 1;
            }
        }
        return ab.size();
    }

    /**
     * Power and variance reduction.
     *
     * @param simulations simulated tests
     * @param users users per test
     * @param rho the pre-period correlation
     * @param lift the true relative lift
     * @param seed the base seed
     * @return metric name to value: power per method, sample size relative to the plain test, and
     *     the mSPRT's power and mean stopping fraction
     */
    public static Map<String, Double> power(int simulations, int users, double rho, double lift, long seed) {
        CausalScenarios.AbTest prePeriod = CausalScenarios.abTest(4000, rho, 0, 1, seed + 99_999);
        double[][] preFeatures = features(prePeriod.pre());
        Regressor.Fitted cupacModel = Regressor.ridge(1e-6).fit(preFeatures, prePeriod.y());
        int plain = 0;
        int cuped = 0;
        int cupac = 0;
        int seq = 0;
        double stop = 0;
        double varPlain = 0;
        double varCuped = 0;
        double varCupac = 0;
        for (int s = 0; s < simulations; s++) {
            CausalScenarios.AbTest ab = CausalScenarios.abTest(users, rho, lift, 14, seed + s);
            Test p = io.github.zzuegg.brainiac.experimental.causal.OnlineExperiments.tTest(ab.y(), ab.t(), users);
            Test c = io.github.zzuegg.brainiac.experimental.causal.OnlineExperiments.tTest(
                    io.github.zzuegg.brainiac.experimental.causal.OnlineExperiments.cuped(ab.y(), ab.pre(), users),
                    ab.t(), users);
            Test a = io.github.zzuegg.brainiac.experimental.causal.OnlineExperiments.tTest(
                    io.github.zzuegg.brainiac.experimental.causal.OnlineExperiments.cupac(ab.y(), features(ab.pre()),
                            cupacModel, users), ab.t(), users);
            plain += p.pValue() < ALPHA ? 1 : 0;
            cuped += c.pValue() < ALPHA ? 1 : 0;
            cupac += a.pValue() < ALPHA ? 1 : 0;
            varPlain += p.se() * p.se();
            varCuped += c.se() * c.se();
            varCupac += a.se() * a.se();
            int at = sequential(ab, 0.3);
            seq += at < users ? 1 : 0;
            stop += at / (double) users / simulations;
        }
        Map<String, Double> out = new LinkedHashMap<>();
        out.put("power t-test", plain / (double) simulations);
        out.put("power CUPED", cuped / (double) simulations);
        out.put("power CUPAC", cupac / (double) simulations);
        out.put("power mSPRT", seq / (double) simulations);
        // the sample size for fixed power scales with the estimator's variance
        out.put("relative n CUPED", varCuped / varPlain);
        out.put("relative n CUPAC", varCupac / varPlain);
        out.put("mSPRT mean stop fraction", stop);
        return out;
    }

    private static double[][] features(double[] pre) {
        double[][] f = new double[pre.length][];
        for (int i = 0; i < pre.length; i++) {
            f[i] = new double[] {pre[i], Math.log(pre[i]), Math.sqrt(pre[i])};
        }
        return f;
    }
}
