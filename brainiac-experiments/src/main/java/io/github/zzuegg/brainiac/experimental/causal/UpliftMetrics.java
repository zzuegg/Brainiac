package io.github.zzuegg.brainiac.experimental.causal;

/**
 * How uplift models are judged (Radcliffe 2007; Gutierrez and Gérardy 2017,
 * <a href="https://proceedings.mlr.press/v67/gutierrez17a.html">PMLR v67</a>). Customers are sorted
 * by predicted uplift and the curves track what targeting the top fraction would have gained, read
 * off a randomised test set:
 * <ul>
 *   <li>{@link #qiniCurve}: {@code Q(k) = Y_T(k) - Y_C(k) N_T(k) / N_C(k)}, the incremental
 *       conversions among the top {@code k} (treated conversions minus control conversions scaled
 *       to the treated count);</li>
 *   <li>{@link #qiniCoefficient}: the area between the model's Qini curve and the straight line of
 *       random targeting, per customer (0 for random, larger is better);</li>
 *   <li>{@link #auuc}: the area under the uplift curve
 *       {@code U(k) = (Y_T(k) / N_T(k) - Y_C(k) / N_C(k)) (N_T(k) + N_C(k))}, per customer;</li>
 *   <li>{@link #bestProfit} and {@link #policyProfit}: the expected profit of targeting a fraction
 *       or of the rule "treat if uplift times value exceeds the cost", computed from the true
 *       per-customer uplift when the scenario knows it.</li>
 * </ul>
 */
public final class UpliftMetrics {

    private UpliftMetrics() {
    }

    private static int[] order(double[] scores) {
        return java.util.stream.IntStream.range(0, scores.length).boxed()
                .sorted((a, b) -> Double.compare(scores[b], scores[a])).mapToInt(Integer::intValue).toArray();
    }

    /**
     * The Qini curve at every prefix.
     *
     * @param scores predicted uplift per customer
     * @param t treatments
     * @param y 0/1 outcomes
     * @return {@code Q(k)} for {@code k = 0..n}
     */
    public static double[] qiniCurve(double[] scores, int[] t, int[] y) {
        int[] o = order(scores);
        double[] q = new double[o.length + 1];
        double nt = 0;
        double nc = 0;
        double yt = 0;
        double yc = 0;
        for (int k = 0; k < o.length; k++) {
            int i = o[k];
            if (t[i] == 1) {
                nt++;
                yt += y[i];
            } else {
                nc++;
                yc += y[i];
            }
            q[k + 1] = yt - (nc > 0 ? yc * nt / nc : 0);
        }
        return q;
    }

    /**
     * The Qini coefficient: the mean gap between the model's Qini curve and random targeting, per
     * customer.
     *
     * @param scores predicted uplift
     * @param t treatments
     * @param y 0/1 outcomes
     * @return the coefficient (0 for random)
     */
    public static double qiniCoefficient(double[] scores, int[] t, int[] y) {
        double[] q = qiniCurve(scores, t, y);
        int n = scores.length;
        double area = 0;
        for (int k = 1; k <= n; k++) {
            area += q[k] - q[n] * k / n;
        }
        return area / ((double) n * n);
    }

    /**
     * The area under the uplift curve, per customer.
     *
     * @param scores predicted uplift
     * @param t treatments
     * @param y 0/1 outcomes
     * @return the AUUC
     */
    public static double auuc(double[] scores, int[] t, int[] y) {
        int[] o = order(scores);
        double nt = 0;
        double nc = 0;
        double yt = 0;
        double yc = 0;
        double area = 0;
        for (int k = 0; k < o.length; k++) {
            int i = o[k];
            if (t[i] == 1) {
                nt++;
                yt += y[i];
            } else {
                nc++;
                yc += y[i];
            }
            if (nt > 0 && nc > 0) {
                area += (yt / nt - yc / nc) * (nt + nc);
            }
        }
        return area / ((double) o.length * o.length);
    }

    /**
     * The best expected profit over targeting fractions when the customers are ranked by a model:
     * {@code max_k sum over the top k of (value * uplift_i - cost)}, per customer.
     *
     * @param scores predicted uplift (only the ranking matters)
     * @param trueUplift the true uplift per customer
     * @param value the value of one conversion
     * @param cost the cost of one contact
     * @return {@code {profit per customer, best fraction}}
     */
    public static double[] bestProfit(double[] scores, double[] trueUplift, double value, double cost) {
        int[] o = order(scores);
        double acc = 0;
        double best = 0;
        int bestK = 0;
        for (int k = 0; k < o.length; k++) {
            acc += value * trueUplift[o[k]] - cost;
            if (acc > best) {
                best = acc;
                bestK = k + 1;
            }
        }
        return new double[] {best / o.length, bestK / (double) o.length};
    }

    /**
     * The expected profit of the policy "treat if {@code value * predicted uplift > cost}", per
     * customer.
     *
     * @param scores predicted uplift
     * @param trueUplift the true uplift
     * @param value the value of a conversion
     * @param cost the cost of a contact
     * @return the profit per customer
     */
    public static double policyProfit(double[] scores, double[] trueUplift, double value, double cost) {
        double s = 0;
        for (int i = 0; i < scores.length; i++) {
            if (value * scores[i] > cost) {
                s += value * trueUplift[i] - cost;
            }
        }
        return s / scores.length;
    }
}
