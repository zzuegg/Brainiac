package io.github.zzuegg.brainiac.experimental.sketch;

/**
 * Ski rental with predictions: renting costs 1 per day, buying costs {@code b}, and the
 * number of ski days {@code y} is unknown. Buying at the start of day {@code j} costs
 * {@code y} if {@code y < j} and {@code j - 1 + b} otherwise; the offline optimum is
 * {@code min(y, b)}.
 *
 * <ul>
 *   <li>{@link #breakEven}: buy on day {@code b}, 2-competitive;</li>
 *   <li>{@link #expectedClassicRandomized}: Karlin et al.'s randomised rule,
 *       {@code e / (e - 1)}-competitive;</li>
 *   <li>{@link #deterministic} (Purohit, Svitkina and Kumar, NeurIPS 2018, Algorithm 2): with
 *       a predicted horizon {@code p}, buy on day {@code ceil(lambda b)} if {@code p >= b}, else
 *       on day {@code ceil(b / lambda)}; {@code (1 + 1/lambda)}-robust and
 *       {@code (1 + lambda)}-consistent;</li>
 *   <li>{@link #expectedRandomized} (their Algorithm 3): the same split with a randomised
 *       buy day, {@code (1 + 1/b)/(1 - e^-(lambda - 1/b))}-robust and
 *       {@code lambda/(1 - e^-lambda)}-consistent.</li>
 * </ul>
 */
public final class SkiRental {

    private SkiRental() {
    }

    /**
     * The cost of buying at the start of a day.
     *
     * @param buyDay the day (from 1)
     * @param days the ski days {@code y}
     * @param b the buy price
     * @return the cost
     */
    public static double cost(int buyDay, int days, int b) {
        return days < buyDay ? days : buyDay - 1 + b;
    }

    /**
     * @param days the ski days
     * @param b the buy price
     * @return the offline optimum
     */
    public static double optimum(int days, int b) {
        return Math.min(days, b);
    }

    /**
     * @param b the buy price
     * @return the break-even buy day
     */
    public static int breakEven(int b) {
        return b;
    }

    /**
     * The deterministic prediction-augmented buy day.
     *
     * @param predicted the predicted ski days
     * @param b the buy price
     * @param lambda the trust parameter in {@code (0, 1]} (small trusts the prediction)
     * @return the buy day
     */
    public static int deterministic(double predicted, int b, double lambda) {
        return predicted >= b ? Math.max(1, (int) Math.ceil(lambda * b)) : (int) Math.ceil(b / lambda);
    }

    /**
     * The expected cost of Karlin's randomised rule: buy on day {@code i <= b} with
     * probability proportional to {@code (1 - 1/b)^(b - i)}.
     *
     * @param days the ski days
     * @param b the buy price
     * @return the expected cost
     */
    public static double expectedClassicRandomized(int days, int b) {
        return expected(b, days, b);
    }

    private static double expected(int horizon, int days, int b) {
        double q = 1 - 1.0 / b;
        double z = 0;
        for (int i = 1; i <= horizon; i++) {
            z += Math.pow(q, horizon - i);
        }
        double e = 0;
        for (int i = 1; i <= horizon; i++) {
            e += Math.pow(q, horizon - i) / z * cost(i, days, b);
        }
        return e;
    }

    /**
     * The expected cost of the randomised prediction-augmented rule: with {@code k =
     * floor(lambda b)} if the prediction is at least {@code b} and {@code l = ceil(b / lambda)}
     * otherwise, buy on day {@code i <= k} (or {@code l}) with probability proportional to
     * {@code (1 - 1/b)^(k - i)}.
     *
     * @param predicted the predicted ski days
     * @param days the actual ski days
     * @param b the buy price
     * @param lambda the trust parameter in {@code (1/b, 1]}
     * @return the expected cost
     */
    public static double expectedRandomized(double predicted, int days, int b, double lambda) {
        int horizon = predicted >= b ? Math.max(1, (int) Math.floor(lambda * b)) : (int) Math.ceil(b / lambda);
        return expected(horizon, days, b);
    }
}
