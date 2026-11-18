package io.github.zzuegg.brainiac.experimental.timeseries;

/**
 * Forecasts for intermittent demand (mostly zeros). Every method returns a flat per-period
 * demand rate for all future periods.
 *
 * <ul>
 *   <li>{@link Method#CROSTON} (Croston 1972, <a href="https://doi.org/10.1057/jors.1972.50">doi</a>):
 *       simple exponential smoothing of the non-zero sizes {@code z} and of the intervals
 *       {@code p} between them, updated only in periods with demand; the forecast is
 *       {@code z / p};</li>
 *   <li>{@link Method#SBA} (Syntetos and Boylan 2005,
 *       <a href="https://doi.org/10.1016/j.ijforecast.2004.10.001">doi</a>): Croston times
 *       {@code 1 - alpha / 2}, which removes its positive bias;</li>
 *   <li>{@link Method#TSB} (Teunter, Syntetos and Babai 2011,
 *       <a href="https://doi.org/10.1016/j.ejor.2011.05.018">doi</a>): the demand probability is
 *       smoothed every period (towards 1 or 0, with {@code beta}) and the sizes when demand
 *       occurs; the forecast is {@code probability * size}, so it decays during long runs of
 *       zeros and tracks obsolescence;</li>
 *   <li>{@link Method#ADIDA} (Nikolopoulos et al. 2011): the series is summed into
 *       non-overlapping buckets of the mean inter-demand interval (from the end), the bucket
 *       totals are forecast by SES, and the forecast is divided by the bucket length;</li>
 *   <li>{@link Method#IMAPA} (Petropoulos and Kourentzes 2015): the mean of the ADIDA forecasts
 *       over bucket lengths 1 to {@code max(2, 2 * mean interval)} (capped at 12).</li>
 * </ul>
 */
public final class IntermittentDemand {

    private IntermittentDemand() {
    }

    /** A method. */
    public enum Method {
        /** Croston's method. */
        CROSTON,
        /** The Syntetos-Boylan approximation. */
        SBA,
        /** Teunter-Syntetos-Babai. */
        TSB,
        /** Aggregate, disaggregate. */
        ADIDA,
        /** Multiple aggregation levels. */
        IMAPA
    }

    /**
     * Simple exponential smoothing initialised at the first value.
     *
     * @param y the series
     * @param alpha the smoothing weight
     * @return the final level
     */
    public static double ses(double[] y, double alpha) {
        if (y.length == 0) {
            return 0;
        }
        double level = y[0];
        for (int t = 1; t < y.length; t++) {
            level += alpha * (y[t] - level);
        }
        return level;
    }

    /**
     * The mean interval between demand occurrences.
     *
     * @param y the series
     * @return the mean interval, at least 1 ({@code y.length} without demand)
     */
    public static double meanInterval(double[] y) {
        int count = 0;
        for (double v : y) {
            count += v > 0 ? 1 : 0;
        }
        return count == 0 ? y.length : Math.max(1, y.length / (double) count);
    }

    /**
     * The per-period forecast.
     *
     * @param y the history
     * @param method the method
     * @param alpha the smoothing weight of sizes and intervals (0.1 is the usual choice)
     * @param beta the smoothing weight of TSB's demand probability (ignored otherwise)
     * @return the flat forecast
     */
    public static double forecast(double[] y, Method method, double alpha, double beta) {
        return switch (method) {
            case CROSTON -> croston(y, alpha);
            case SBA -> (1 - alpha / 2) * croston(y, alpha);
            case TSB -> tsb(y, alpha, beta);
            case ADIDA -> adida(y, (int) Math.max(1, Math.round(meanInterval(y))), alpha);
            case IMAPA -> {
                int max = (int) Math.min(12, Math.max(2, Math.round(2 * meanInterval(y))));
                double s = 0;
                for (int k = 1; k <= max; k++) {
                    s += adida(y, k, alpha) / max;
                }
                yield s;
            }
        };
    }

    private static double croston(double[] y, double alpha) {
        double z = Double.NaN;
        double p = Double.NaN;
        int since = 1;
        for (double v : y) {
            if (v > 0) {
                if (Double.isNaN(z)) {
                    z = v;
                    p = since;
                } else {
                    z += alpha * (v - z);
                    p += alpha * (since - p);
                }
                since = 1;
            } else {
                since++;
            }
        }
        return Double.isNaN(z) ? 0 : z / p;
    }

    private static double tsb(double[] y, double alpha, double beta) {
        double z = Double.NaN;
        int occurrences = 0;
        for (double v : y) {
            occurrences += v > 0 ? 1 : 0;
        }
        if (occurrences == 0) {
            return 0;
        }
        double prob = occurrences / (double) y.length;
        for (double v : y) {
            boolean demand = v > 0;
            prob += beta * ((demand ? 1 : 0) - prob);
            if (demand) {
                z = Double.isNaN(z) ? v : z + alpha * (v - z);
            }
        }
        return prob * z;
    }

    private static double adida(double[] y, int bucket, double alpha) {
        int buckets = y.length / bucket;
        if (buckets == 0) {
            return 0;
        }
        double[] agg = new double[buckets];
        int offset = y.length - buckets * bucket;
        for (int b = 0; b < buckets; b++) {
            for (int k = 0; k < bucket; k++) {
                agg[b] += y[offset + b * bucket + k];
            }
        }
        // initialise at the mean of the first few buckets for stability
        double m = 0;
        int w = Math.min(4, buckets);
        for (int b = 0; b < w; b++) {
            m += agg[b] / w;
        }
        double level = m;
        for (double v : agg) {
            level += alpha * (v - level);
        }
        return level / bucket;
    }
}
