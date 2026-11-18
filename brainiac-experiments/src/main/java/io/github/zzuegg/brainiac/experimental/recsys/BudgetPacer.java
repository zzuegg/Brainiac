package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.experimental.ml.LinearProgram;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Budget pacing for an advertiser in a stream of second-price auctions, plus the simulator and
 * the hindsight optimum to score it against.
 * <ul>
 *   <li>{@link #truthful}: bid the value until the budget runs out (the baseline that exhausts
 *       early in the expensive hours);</li>
 *   <li>{@link #throttle}: participate in each auction with a fixed probability;</li>
 *   <li>{@link #pid}: probabilistic throttling whose participation rate is driven by a PID
 *       controller on the gap between the planned (uniform) and the actual cumulative spend
 *       (Agarwal et al., KDD 2014; Xu et al., WSDM 2015);</li>
 *   <li>{@link #dual}: bid shading by the budget's dual variable, {@code bid = v / (1 + mu)},
 *       updated after every auction by online dual (mirror) descent,
 *       {@code mu <- max(0, mu - eps (B/T - spend_t))} (Balseiro and Gur, Management Science
 *       2019), which is asymptotically optimal for stochastic arrivals.</li>
 * </ul>
 * {@link #hindsight} is the best fractional allocation with all prices known in advance: a
 * fractional knapsack, which the greedy value/price order solves exactly; {@link #hindsightLp}
 * solves the same linear program with {@link LinearProgram#maximize} for small instances.
 */
public final class BudgetPacer {

    /** A bidding policy. */
    public interface Policy {
        /**
         * @param value the impression's value
         * @param t the auction index
         * @return the bid (0 to skip)
         */
        double bid(double value, int t);

        /**
         * @param paid what the auction cost (0 if lost or skipped)
         * @param t the auction index
         */
        void observe(double paid, int t);
    }

    /**
     * The result of a simulated day.
     *
     * @param value total value won
     * @param spend total spend
     * @param depletion the fraction of the day at which the budget ran out (1 if it never did)
     * @param wins auctions won
     */
    public record Outcome(double value, double spend, double depletion, int wins) {
    }

    private BudgetPacer() {
    }

    /** @return a policy that bids the value */
    public static Policy truthful() {
        return new Policy() {
            @Override
            public double bid(double value, int t) {
                return value;
            }

            @Override
            public void observe(double paid, int t) {
            }
        };
    }

    /**
     * @param probability the participation probability
     * @param seed the coin seed
     * @return a policy that bids the value in a random share of the auctions
     */
    public static Policy throttle(double probability, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        return new Policy() {
            @Override
            public double bid(double value, int t) {
                return r.nextDouble() < probability ? value : 0;
            }

            @Override
            public void observe(double paid, int t) {
            }
        };
    }

    /**
     * PID-controlled probabilistic throttling.
     *
     * @param budget the day's budget
     * @param auctions the day's auctions
     * @param interval auctions between controller updates
     * @param kp proportional gain
     * @param ki integral gain
     * @param kd derivative gain
     * @param seed the coin seed
     * @return the policy
     */
    public static Policy pid(double budget, int auctions, int interval, double kp, double ki, double kd, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        return new Policy() {
            private double rate = 0.5;
            private double spent;
            private double integral;
            private double previous;

            @Override
            public double bid(double value, int t) {
                return r.nextDouble() < rate ? value : 0;
            }

            @Override
            public void observe(double paid, int t) {
                spent += paid;
                if ((t + 1) % interval == 0) {
                    double error = (budget * (t + 1) / auctions - spent) / budget;
                    integral += error;
                    rate = Math.max(0.001, Math.min(1, rate + kp * error + ki * integral + kd * (error - previous)));
                    previous = error;
                }
            }
        };
    }

    /**
     * Dual-gradient bid shading.
     *
     * @param budget the day's budget
     * @param auctions the day's auctions
     * @param eta the step size relative to the per-auction target spend
     * @return the policy
     */
    public static Policy dual(double budget, int auctions, double eta) {
        double rho = budget / auctions;
        return new Policy() {
            private double mu;

            @Override
            public double bid(double value, int t) {
                return value / (1 + mu);
            }

            @Override
            public void observe(double paid, int t) {
                mu = Math.max(0, mu - eta / rho * (rho - paid));
            }
        };
    }

    /**
     * Runs a day of second-price auctions: the advertiser wins when its bid beats the competing
     * bid and the remaining budget covers the price, and pays the competing bid.
     *
     * @param value value per auction
     * @param competing highest competing bid per auction
     * @param budget the budget
     * @param policy the policy
     * @return the outcome
     */
    public static Outcome simulate(double[] value, double[] competing, double budget, Policy policy) {
        int n = value.length;
        double spent = 0;
        double won = 0;
        int wins = 0;
        double depletion = 1;
        for (int t = 0; t < n; t++) {
            double bid = policy.bid(value[t], t);
            double paid = 0;
            if (bid > competing[t] && spent + competing[t] <= budget) {
                paid = competing[t];
                spent += paid;
                won += value[t];
                wins++;
            }
            policy.observe(paid, t);
            if (depletion == 1 && budget - spent < 0.01 * budget) {
                depletion = (t + 1) / (double) n;
            }
        }
        return new Outcome(won, spent, depletion, wins);
    }

    /**
     * The hindsight optimum: maximise {@code sum v_i x_i} subject to {@code sum c_i x_i <= B},
     * {@code 0 <= x_i <= 1}, solved exactly by taking auctions in decreasing value/price order.
     * Only auctions whose value exceeds the price are worth winning in a second-price auction
     * with a truthful-at-most bidder, so others are left out.
     *
     * @param value values
     * @param price prices (competing bids)
     * @param budget the budget
     * @return the optimal value
     */
    public static double hindsight(double[] value, double[] price, double budget) {
        Integer[] order = new Integer[value.length];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(value[b] / price[b], value[a] / price[a]));
        double left = budget;
        double total = 0;
        for (int i : order) {
            if (value[i] <= price[i] || left <= 0) {
                continue;
            }
            double x = Math.min(1, left / price[i]);
            total += x * value[i];
            left -= x * price[i];
        }
        return total;
    }

    /**
     * The same hindsight program solved by the simplex method (for a few hundred auctions).
     *
     * @param value values
     * @param price prices
     * @param budget the budget
     * @return the optimal value
     */
    public static double hindsightLp(double[] value, double[] price, double budget) {
        int n = value.length;
        double[] c = new double[n];
        double[][] a = new double[n + 1][n];
        double[] b = new double[n + 1];
        for (int i = 0; i < n; i++) {
            c[i] = value[i] > price[i] ? value[i] : 0;
            a[0][i] = price[i];
            a[i + 1][i] = 1;
            b[i + 1] = 1;
        }
        b[0] = budget;
        return LinearProgram.maximize(c, a, b).objective();
    }
}
