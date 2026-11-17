package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.experimental.nn.EchoStateReservoir;
import io.github.zzuegg.brainiac.experimental.nn.NextGenReservoirFeatures;
import io.github.zzuegg.brainiac.experimental.train.RidgeRegression;

/**
 * Reservoir computing on chaotic series: an echo state network (Jaeger and Haas,
 * <a href="https://doi.org/10.1126/science.1091277">Science 2004</a>) against next
 * generation reservoir computing (Gauthier et al., <a href="https://arxiv.org/abs/2106.07688">2106.07688</a>).
 * Both learn only a ridge readout from teacher-forced data, then forecast in closed loop by
 * feeding their own predictions back. The figure of merit is the valid time: how many steps
 * the free-running forecast stays within a normalized error of the true trajectory.
 */
@SuppressWarnings("exports")   // experimental.* packages are exported by the lead at integration
public final class ReservoirForecast {
    private ReservoirForecast() {
    }

    /**
     * A forecast.
     *
     * @param predicted the closed-loop predictions for the steps after training
     * @param trainingError the normalized RMSE of one-step predictions on the training data
     */
    public record Forecast(double[][] predicted, double trainingError) {
    }

    private static double[] std(double[][] s, int n) {
        int d = s[0].length;
        double[] mean = new double[d];
        double[] var = new double[d];
        for (int t = 0; t < n; t++) {
            for (int j = 0; j < d; j++) {
                mean[j] += s[t][j] / n;
            }
        }
        for (int t = 0; t < n; t++) {
            for (int j = 0; j < d; j++) {
                var[j] += (s[t][j] - mean[j]) * (s[t][j] - mean[j]) / n;
            }
        }
        for (int j = 0; j < d; j++) {
            var[j] = Math.sqrt(var[j]);
        }
        return var;
    }

    private static double nrmse(double[] p, double[] y, double[] scale) {
        double s = 0;
        for (int j = 0; j < p.length; j++) {
            double e = (p[j] - y[j]) / scale[j];
            s += e * e;
        }
        return Math.sqrt(s / p.length);
    }

    /**
     * NG-RC: ridge-regresses the next increment {@code x_(t+1) - x_t} on the NG-RC features
     * of the first {@code train} samples, then forecasts {@code horizon} steps.
     *
     * @param series {@code [time][dimension]}, at least {@code train} long
     * @param train the number of training samples
     * @param horizon how many steps to forecast after them
     * @param delays the taps k
     * @param skip the tap spacing
     * @param lambda the ridge penalty
     * @return the forecast
     */
    public static Forecast nextGeneration(double[][] series, int train, int horizon, int delays, int skip,
            double lambda) {
        int d = series[0].length;
        NextGenReservoirFeatures f = NextGenReservoirFeatures.of(d, delays, skip);
        RidgeRegression r = RidgeRegression.of(f.size(), d);
        for (int t = f.warmup(); t < train - 1; t++) {
            double[] dx = new double[d];
            for (int j = 0; j < d; j++) {
                dx[j] = series[t + 1][j] - series[t][j];
            }
            r.add(f.features(series, t), dx);
        }
        double[][] w = r.solve(lambda);
        double[] scale = std(series, train);
        double err = 0;
        int count = 0;
        for (int t = f.warmup(); t < train - 1; t++) {
            double[] p = RidgeRegression.predict(w, f.features(series, t));
            for (int j = 0; j < d; j++) {
                p[j] += series[t][j];
            }
            err += Math.pow(nrmse(p, series[t + 1], scale), 2);
            count++;
        }
        double[][] buffer = new double[train + horizon][];
        System.arraycopy(series, 0, buffer, 0, train);
        for (int t = train - 1; t < train + horizon - 1; t++) {
            double[] p = RidgeRegression.predict(w, f.features(buffer, t));
            for (int j = 0; j < d; j++) {
                p[j] += buffer[t][j];
            }
            buffer[t + 1] = p;
        }
        double[][] out = new double[horizon][];
        System.arraycopy(buffer, train, out, 0, horizon);
        return new Forecast(out, Math.sqrt(err / count));
    }

    /**
     * Echo state network: drives the reservoir with the normalized series, ridge-regresses the
     * next sample on {@code [1, state, input]} after a washout, then runs in closed loop.
     *
     * @param series {@code [time][dimension]}
     * @param train the number of training samples
     * @param horizon how many steps to forecast
     * @param reservoir the reservoir, with {@code dimension} inputs
     * @param washout the first steps whose states are not used
     * @param lambda the ridge penalty
     * @return the forecast
     */
    public static Forecast echoState(double[][] series, int train, int horizon, EchoStateReservoir reservoir,
            int washout, double lambda) {
        int d = series[0].length;
        double[] scale = std(series, train);
        double[] mean = new double[d];
        for (int t = 0; t < train; t++) {
            for (int j = 0; j < d; j++) {
                mean[j] += series[t][j] / train;
            }
        }
        int n = reservoir.units();
        RidgeRegression r = RidgeRegression.of(1 + n + d, d);
        reservoir.reset();
        double[][] features = new double[train - 1][];
        for (int t = 0; t < train - 1; t++) {
            double[] u = normalize(series[t], mean, scale);
            features[t] = concat(reservoir.step(u), u);
            if (t >= washout) {
                r.add(features[t], normalize(series[t + 1], mean, scale));
            }
        }
        double[][] w = r.solve(lambda);
        double err = 0;
        for (int t = washout; t < train - 1; t++) {
            double[] p = RidgeRegression.predict(w, features[t]);
            err += Math.pow(nrmse(p, normalize(series[t + 1], mean, scale), ones(d)), 2);
        }
        double[][] out = new double[horizon][];
        double[] u = normalize(series[train - 1], mean, scale);
        for (int h = 0; h < horizon; h++) {
            u = RidgeRegression.predict(w, concat(reservoir.step(u), u));
            out[h] = new double[d];
            for (int j = 0; j < d; j++) {
                out[h][j] = u[j] * scale[j] + mean[j];
            }
        }
        return new Forecast(out, Math.sqrt(err / (train - 1 - washout)));
    }

    private static double[] ones(int d) {
        double[] o = new double[d];
        java.util.Arrays.fill(o, 1);
        return o;
    }

    private static double[] normalize(double[] x, double[] mean, double[] scale) {
        double[] r = new double[x.length];
        for (int j = 0; j < x.length; j++) {
            r[j] = (x[j] - mean[j]) / scale[j];
        }
        return r;
    }

    private static double[] concat(double[] state, double[] u) {
        double[] f = new double[1 + state.length + u.length];
        f[0] = 1;
        System.arraycopy(state, 0, f, 1, state.length);
        System.arraycopy(u, 0, f, 1 + state.length, u.length);
        return f;
    }

    /**
     * Counts the leading forecast steps whose error, normalized per dimension by the series'
     * standard deviation, stays below a threshold.
     *
     * @param predicted the forecast
     * @param series the full true series
     * @param start the index in {@code series} of the first forecast step
     * @param threshold the error limit, for example 0.3
     * @return the valid steps
     */
    public static int validSteps(double[][] predicted, double[][] series, int start, double threshold) {
        double[] scale = std(series, series.length);
        for (int h = 0; h < predicted.length; h++) {
            if (!(nrmse(predicted[h], series[start + h], scale) < threshold)) {
                return h;
            }
        }
        return predicted.length;
    }
}
