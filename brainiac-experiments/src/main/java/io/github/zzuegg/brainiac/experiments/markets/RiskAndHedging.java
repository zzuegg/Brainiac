package io.github.zzuegg.brainiac.experiments.markets;

import io.github.zzuegg.brainiac.experimental.data.MarketPaths;
import io.github.zzuegg.brainiac.experimental.markets.DeepHedging;
import io.github.zzuegg.brainiac.experimental.markets.Volatility;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Volatility forecasting and deep hedging (wave 12, items 12.4.15-12.4.16).
 *
 * <ul>
 *   <li>{@link #volatility}: QLIKE of variance forecasts at several horizons against the true
 *       conditional variance, and 99% VaR exceedance rates, for EWMA, rolling variance,
 *       GARCH/GJR/EGARCH (Gaussian and t) and an MLP on past squared returns, fitted on the
 *       first part of a series and tested on the rest.</li>
 *   <li>{@link #hedging}: deep hedging (CVaR 95%) against no hedge, Black-Scholes delta and the
 *       Whalley-Wilmott band, on GBM or Heston paths with proportional costs.</li>
 * </ul>
 */
public final class RiskAndHedging {

    /** The return process. */
    public enum Process {
        /** GARCH(1,1) with t(6) innovations. */
        GARCH_T,
        /** GJR-GARCH with t(6) innovations (leverage). */
        GJR_T,
        /** Heston stochastic volatility. */
        HESTON,
        /** Markov regime switching with jumps. */
        REGIME
    }

    private RiskAndHedging() {
    }

    /**
     * Generates returns.
     *
     * @param process the process
     * @param days the length
     * @param rnd the random source
     * @return returns and true variances
     */
    public static MarketPaths.Returns returns(Process process, int days, SplittableRandom rnd) {
        return switch (process) {
            case GARCH_T -> MarketPaths.garch(days, 2e-6, 0.08, 0, 0.9, 6, rnd);
            case GJR_T -> MarketPaths.garch(days, 2e-6, 0.02, 0.12, 0.9, 6, rnd);
            case HESTON -> MarketPaths.hestonReturns(days, rnd);
            case REGIME -> MarketPaths.regimeSwitching(days, rnd);
        };
    }

    /**
     * Forecast evaluation.
     *
     * @param process the process
     * @param trainDays fitting days
     * @param testDays test days
     * @param horizons forecast horizons
     * @param nnSteps MLP training steps
     * @param seed the seed
     * @return model to {@code {QLIKE per horizon..., 99% VaR exceedance rate at h = 1}}
     */
    public static Map<String, double[]> volatility(Process process, int trainDays, int testDays, int[] horizons,
            int nnSteps, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        MarketPaths.Returns data = returns(process, trainDays + testDays, rnd);
        double[] r = data.returns();
        double[] train = java.util.Arrays.copyOf(r, trainDays);
        Map<String, double[]> out = new LinkedHashMap<>();
        Object[][] models = {
            {"EWMA 0.94", Volatility.Kind.EWMA, Volatility.Innovation.GAUSSIAN},
            {"GARCH-N", Volatility.Kind.GARCH, Volatility.Innovation.GAUSSIAN},
            {"GARCH-t", Volatility.Kind.GARCH, Volatility.Innovation.STUDENT_T},
            {"GJR-t", Volatility.Kind.GJR_GARCH, Volatility.Innovation.STUDENT_T},
            {"EGARCH-t", Volatility.Kind.EGARCH, Volatility.Innovation.STUDENT_T}};
        for (Object[] spec : models) {
            Volatility model = Volatility.fit((Volatility.Kind) spec[1], (Volatility.Innovation) spec[2], train);
            double[][] f = new double[horizons.length][];
            for (int h = 0; h < horizons.length; h++) {
                f[h] = model.forecast(r, horizons[h]);
            }
            double[] row = score(f, horizons, data, trainDays, testDays);
            row[horizons.length] = exceedance(r, f[0], trainDays, testDays, v -> model.valueAtRisk(v, 0.99));
            out.put((String) spec[0], row);
        }
        double[] roll = Volatility.rolling(r, 60);
        double[][] rf = new double[horizons.length][];
        java.util.Arrays.fill(rf, roll);
        double[] rollRow = score(rf, horizons, data, trainDays, testDays);
        rollRow[horizons.length] = exceedance(r, roll, trainDays, testDays,
                v -> Volatility.quantile(Double.POSITIVE_INFINITY, 0.99) * Math.sqrt(v));
        out.put("rolling 60d", rollRow);
        double[][] nn = Volatility.neuralForecaster(r, trainDays, horizons, 20, nnSteps, seed);
        double[] nnRow = score(nn, horizons, data, trainDays, testDays);
        nnRow[horizons.length] = exceedance(r, nn[0], trainDays, testDays,
                v -> Volatility.quantile(Double.POSITIVE_INFINITY, 0.99) * Math.sqrt(v));
        out.put("MLP on r^2 lags", nnRow);
        return out;
    }

    private static double[] score(double[][] f, int[] horizons, MarketPaths.Returns data, int trainDays,
            int testDays) {
        double[] row = new double[horizons.length + 1];
        int maxH = java.util.Arrays.stream(horizons).max().orElse(1);
        int count = testDays - maxH;
        for (int h = 0; h < horizons.length; h++) {
            double[] fc = new double[count];
            double[] truth = new double[count];
            for (int i = 0; i < count; i++) {
                int origin = trainDays + i;
                fc[i] = f[h][origin];
                // floored: Heston's truncated variance can touch zero
                truth[i] = Math.max(data.variance()[origin + horizons[h] - 1], 1e-6);
            }
            row[h] = Volatility.qlike(fc, truth);
        }
        return row;
    }

    private static double exceedance(double[] r, double[] oneStep, int trainDays, int testDays,
            java.util.function.DoubleUnaryOperator var) {
        int hits = 0;
        for (int t = trainDays; t < trainDays + testDays; t++) {
            hits += -r[t] > var.applyAsDouble(oneStep[t]) ? 1 : 0;
        }
        return hits / (double) testDays;
    }

    /**
     * Hedging a short at-the-money call over 30 rebalancing dates.
     *
     * @param heston Heston paths (else GBM)
     * @param cost the proportional cost
     * @param trainPaths training paths
     * @param testPaths test paths
     * @param steps training steps
     * @param seed the seed
     * @return strategy to {@code {mean PnL, std PnL, CVaR95 of loss, mean costs}}, plus the deep
     *     hedge's indifference price under {@code "deep hedge indifference price"}
     */
    public static Map<String, double[]> hedging(boolean heston, double cost, int trainPaths, int testPaths, int steps,
            long seed) {
        int dates = 30;
        DeepHedging.Setup setup = new DeepHedging.Setup(1, 30 / 252.0, 0.2, cost);
        SplittableRandom rnd = new SplittableRandom(seed);
        MarketPaths.Paths train = paths(heston, trainPaths, dates, setup, rnd);
        MarketPaths.Paths test = paths(heston, testPaths, dates, setup, rnd);
        DeepHedging deep = new DeepHedging(setup, DeepHedging.Objective.CVAR, 0.95, seed);
        deep.train(train.prices(), train.variances(), steps, 128, 5e-3f, seed);
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("no hedge", row(deep.noHedge(test.prices())));
        out.put("BS delta (sigma 0.2)", row(deep.blackScholesDelta(test.prices())));
        out.put("Whalley-Wilmott band", row(deep.whalleyWilmott(test.prices(), 1)));
        out.put("deep hedge (CVaR95)", row(deep.hedge(test.prices(), test.variances())));
        out.put("deep hedge indifference price", new double[] {deep.indifferencePrice(test.prices(),
                test.variances()), DeepHedging.callPrice(1, 1, setup.maturity(), setup.sigma())});
        return out;
    }

    private static MarketPaths.Paths paths(boolean heston, int count, int dates, DeepHedging.Setup setup,
            SplittableRandom rnd) {
        double dt = setup.maturity() / dates;
        return heston ? MarketPaths.heston(count, dates, 1, 0.04, 3, 0.04, 0.6, -0.7, dt, rnd)
                : MarketPaths.gbm(count, dates, 1, 0, setup.sigma(), dt, rnd);
    }

    private static double[] row(DeepHedging.Stats s) {
        return new double[] {s.meanPnl(), s.stdPnl(), s.cvar95(), s.costs()};
    }
}
