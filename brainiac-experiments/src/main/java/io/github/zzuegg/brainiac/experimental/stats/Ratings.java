package io.github.zzuegg.brainiac.experimental.stats;

import io.github.zzuegg.brainiac.experimental.ml.NormalTails;
import io.github.zzuegg.brainiac.experimental.rl.EloLeague;

import java.util.Arrays;

/**
 * Online rating systems for two-player games without draws, behind one interface: predict the
 * chance that one player beats another, then update on the result.
 *
 * <ul>
 *   <li>{@link #elo}: Elo with a fixed K-factor ({@link EloLeague#expected}), the baseline.</li>
 *   <li>{@link #trueSkill}: TrueSkill (Herbrich, Minka and Graepel, NeurIPS 2006): a Gaussian
 *       skill {@code N(mu, sigma^2)} per player updated by the moment-matched (expectation
 *       propagation) truncated-Gaussian factor; new players' large {@code sigma} makes them move
 *       fast, and a dynamics noise {@code tau} lets skills drift.</li>
 *   <li>{@link #glicko2}: Glicko-2 (Glickman, 2012): rating, rating deviation and a volatility
 *       updated per game by the Illinois root finder.</li>
 * </ul>
 */
public final class Ratings {

    /** A rating system. */
    public interface RatingSystem {

        /**
         * The predicted probability that {@code a} beats {@code b}.
         *
         * @param a a player
         * @param b another player
         * @return the probability
         */
        double predict(int a, int b);

        /**
         * Records a result.
         *
         * @param winner the winner
         * @param loser the loser
         */
        void update(int winner, int loser);

        /**
         * Resets one player to the prior (a newcomer taking the slot).
         *
         * @param player the player
         */
        void reset(int player);

        /**
         * The current point estimate of a player's strength.
         *
         * @param player the player
         * @return the rating
         */
        double rating(int player);
    }

    private Ratings() {
    }

    /**
     * Elo on the 400-point scale.
     *
     * @param players the number of players
     * @param k the K-factor
     * @return the system
     */
    public static RatingSystem elo(int players, double k) {
        double[] r = new double[players];
        Arrays.fill(r, 1500);
        return new RatingSystem() {
            @Override
            public double predict(int a, int b) {
                return EloLeague.expected(r[a], r[b]);
            }

            @Override
            public void update(int winner, int loser) {
                double e = EloLeague.expected(r[winner], r[loser]);
                r[winner] += k * (1 - e);
                r[loser] -= k * (1 - e);
            }

            @Override
            public void reset(int player) {
                r[player] = 1500;
            }

            @Override
            public double rating(int player) {
                return r[player];
            }
        };
    }

    /**
     * TrueSkill for one-on-one games with the standard constants ({@code mu0 = 25},
     * {@code sigma0 = 25/3}, {@code beta = sigma0 / 2}, {@code tau = sigma0 / 100}).
     *
     * @param players the number of players
     * @return the system
     */
    public static RatingSystem trueSkill(int players) {
        double mu0 = 25;
        double s0 = 25.0 / 3;
        double beta = s0 / 2;
        double tau = s0 / 100;
        double[] mu = new double[players];
        double[] var = new double[players];
        Arrays.fill(mu, mu0);
        Arrays.fill(var, s0 * s0);
        return new RatingSystem() {
            @Override
            public double predict(int a, int b) {
                double c = Math.sqrt(2 * beta * beta + var[a] + var[b] + 2 * tau * tau);
                return NormalTails.cdf((mu[a] - mu[b]) / c);
            }

            @Override
            public void update(int w, int l) {
                var[w] += tau * tau;
                var[l] += tau * tau;
                double c2 = 2 * beta * beta + var[w] + var[l];
                double c = Math.sqrt(c2);
                double t = (mu[w] - mu[l]) / c;
                double pdf = Math.exp(-t * t / 2) / Math.sqrt(2 * Math.PI);
                double cdf = Math.max(NormalTails.cdf(t), 1e-300);
                double v = pdf / cdf;
                double ww = v * (v + t);
                mu[w] += var[w] / c * v;
                mu[l] -= var[l] / c * v;
                var[w] *= Math.max(1 - var[w] / c2 * ww, 1e-4);
                var[l] *= Math.max(1 - var[l] / c2 * ww, 1e-4);
            }

            @Override
            public void reset(int player) {
                mu[player] = mu0;
                var[player] = s0 * s0;
            }

            @Override
            public double rating(int player) {
                return mu[player];
            }
        };
    }

    /**
     * Glicko-2 with each game as its own rating period ({@code tau = 0.5}, initial rating 1500,
     * deviation 350, volatility 0.06).
     *
     * @param players the number of players
     * @return the system
     */
    public static RatingSystem glicko2(int players) {
        double scale = 173.7178;
        double[] mu = new double[players];
        double[] phi = new double[players];
        double[] sigma = new double[players];
        Arrays.fill(phi, 350 / scale);
        Arrays.fill(sigma, 0.06);
        double tau = 0.5;
        return new RatingSystem() {
            private double g(double p) {
                return 1 / Math.sqrt(1 + 3 * p * p / (Math.PI * Math.PI));
            }

            @Override
            public double predict(int a, int b) {
                double p = Math.sqrt(phi[a] * phi[a] + phi[b] * phi[b]);
                return 1 / (1 + Math.exp(-g(p) * (mu[a] - mu[b])));
            }

            @Override
            public void update(int w, int l) {
                double[] nw = step(w, l, 1);
                double[] nl = step(l, w, 0);
                mu[w] = nw[0];
                phi[w] = nw[1];
                sigma[w] = nw[2];
                mu[l] = nl[0];
                phi[l] = nl[1];
                sigma[l] = nl[2];
            }

            private double[] step(int a, int b, double score) {
                double gb = g(phi[b]);
                double e = 1 / (1 + Math.exp(-gb * (mu[a] - mu[b])));
                double v = 1 / (gb * gb * e * (1 - e));
                double delta = v * gb * (score - e);
                double aa = Math.log(sigma[a] * sigma[a]);
                double p2 = phi[a] * phi[a];
                java.util.function.DoubleUnaryOperator f = x -> {
                    double ex = Math.exp(x);
                    return ex * (delta * delta - p2 - v - ex) / (2 * Math.pow(p2 + v + ex, 2)) - (x - aa) / (tau * tau);
                };
                double lo = aa;
                double hi;
                if (delta * delta > p2 + v) {
                    hi = Math.log(delta * delta - p2 - v);
                } else {
                    int k = 1;
                    while (f.applyAsDouble(aa - k * tau) < 0) {
                        k++;
                    }
                    hi = aa - k * tau;
                }
                double fa = f.applyAsDouble(lo);
                double fb = f.applyAsDouble(hi);
                for (int it = 0; it < 100 && Math.abs(hi - lo) > 1e-6; it++) {
                    double c = lo + (lo - hi) * fa / (fb - fa);
                    double fc = f.applyAsDouble(c);
                    if (fc * fb <= 0) {
                        lo = hi;
                        fa = fb;
                    } else {
                        fa /= 2;
                    }
                    hi = c;
                    fb = fc;
                }
                double ns = Math.exp(lo / 2);
                double pStar = Math.sqrt(p2 + ns * ns);
                double np = 1 / Math.sqrt(1 / (pStar * pStar) + 1 / v);
                double nm = mu[a] + np * np * gb * (score - e);
                return new double[] {nm, np, ns};
            }

            @Override
            public void reset(int player) {
                mu[player] = 0;
                phi[player] = 350 / scale;
                sigma[player] = 0.06;
            }

            @Override
            public double rating(int player) {
                return mu[player];
            }
        };
    }
}
