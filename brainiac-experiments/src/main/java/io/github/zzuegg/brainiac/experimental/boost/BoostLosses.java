package io.github.zzuegg.brainiac.experimental.boost;

import io.github.zzuegg.brainiac.train.GradientBoostedTrees;

import java.util.Arrays;

/**
 * Common {@link GradientBoostedTrees.Loss} derivatives for the boosters of this package (core
 * keeps its own two objectives private): logistic, squared error and the pinball loss of
 * quantile regression.
 */
public final class BoostLosses {

    private BoostLosses() {
    }

    /** @return the logistic loss on 0/1 targets with a log-odds start and a sigmoid link */
    public static GradientBoostedTrees.Loss logistic() {
        return new GradientBoostedTrees.Loss() {
            @Override
            public void derivatives(double[] raw, double[] y, double[] g, double[] h) {
                for (int i = 0; i < raw.length; i++) {
                    double p = 1 / (1 + Math.exp(-raw[i]));
                    g[i] = p - y[i];
                    h[i] = Math.max(p * (1 - p), 1e-12);
                }
            }

            @Override
            public double base(double[] y) {
                double mean = Math.clamp(Arrays.stream(y).average().orElse(0.5), 1e-6, 1 - 1e-6);
                return Math.log(mean / (1 - mean));
            }

            @Override
            public double link(double raw) {
                return 1 / (1 + Math.exp(-raw));
            }
        };
    }

    /** @return half the squared error, starting from the mean */
    public static GradientBoostedTrees.Loss squared() {
        return new GradientBoostedTrees.Loss() {
            @Override
            public void derivatives(double[] raw, double[] y, double[] g, double[] h) {
                for (int i = 0; i < raw.length; i++) {
                    g[i] = raw[i] - y[i];
                    h[i] = 1;
                }
            }

            @Override
            public double base(double[] y) {
                return Arrays.stream(y).average().orElse(0);
            }
        };
    }

    /**
     * The pinball loss {@code max(tau (y - f), (tau - 1)(y - f))}, with a unit curvature so each
     * leaf takes a gradient step, starting from the empirical quantile.
     *
     * @param tau the quantile level in (0, 1)
     * @return the loss
     */
    public static GradientBoostedTrees.Loss quantile(double tau) {
        return new GradientBoostedTrees.Loss() {
            @Override
            public void derivatives(double[] raw, double[] y, double[] g, double[] h) {
                for (int i = 0; i < raw.length; i++) {
                    g[i] = raw[i] > y[i] ? 1 - tau : -tau;
                    h[i] = 1;
                }
            }

            @Override
            public double base(double[] y) {
                double[] s = y.clone();
                Arrays.sort(s);
                return s[Math.clamp((int) (tau * s.length), 0, s.length - 1)];
            }
        };
    }

    /**
     * @param y observed values
     * @param q predicted {@code tau}-quantiles
     * @param tau the level
     * @return the mean pinball loss
     */
    public static double pinball(double[] y, double[] q, double tau) {
        double s = 0;
        for (int i = 0; i < y.length; i++) {
            double d = y[i] - q[i];
            s += Math.max(tau * d, (tau - 1) * d);
        }
        return s / y.length;
    }
}
