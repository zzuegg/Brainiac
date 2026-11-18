package io.github.zzuegg.brainiac.experimental.markets;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.MarketPaths;
import io.github.zzuegg.brainiac.experimental.ml.NormalTails;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Deep hedging of a short European call (Buehler, Gonon, Teichmann and Wood, Quantitative
 * Finance 2019, <a href="https://arxiv.org/abs/1802.03042">arXiv:1802.03042</a>).
 *
 * <p>A feed-forward policy, shared across dates and fed back its previous position (so it is
 * recurrent in the holding), maps {@code (time to maturity, log moneyness, previous holding,
 * volatility input)} to the next stock holding. The hedged P&amp;L of the short call is
 * {@code premium - payoff + sum delta_k (S_(k+1) - S_k) - cost sum |delta_k - delta_(k-1)| S_k}
 * (including the final liquidation), and training minimises a convex risk measure of it:
 * {@link Objective#CVAR} by the Rockafellar-Uryasev form {@code w + E[(-PnL - w)+] / (1 - alpha)}
 * with the auxiliary {@code w} learned jointly, or {@link Objective#ENTROPIC}
 * {@code (1 / lambda) ln E[exp(-lambda PnL)]}. Both are cash invariant, so the risk of the
 * hedged position without premium is the indifference price.
 *
 * <p>Baselines: {@link #noHedge}, {@link #blackScholesDelta} and {@link #whalleyWilmott}'s
 * no-trade band around the delta.
 */
public final class DeepHedging {

    /** The convex risk measure. */
    public enum Objective {
        /** conditional value at risk at level alpha. */
        CVAR,
        /** entropic risk with risk aversion lambda. */
        ENTROPIC
    }

    /**
     * A hedging problem.
     *
     * @param strike the call strike
     * @param maturity the maturity in years
     * @param sigma the Black-Scholes volatility used by the baselines and the premium
     * @param cost the proportional transaction cost
     */
    public record Setup(double strike, double maturity, double sigma, double cost) {
    }

    /**
     * Hedging statistics.
     *
     * @param meanPnl mean P&amp;L
     * @param stdPnl P&amp;L standard deviation
     * @param cvar95 CVaR at 95% of the loss {@code -PnL}
     * @param costs mean transaction costs paid
     */
    public record Stats(double meanPnl, double stdPnl, double cvar95, double costs) {
    }

    private final Setup setup;
    private final Objective objective;
    private final double level;
    private final Mlp policy;
    private final Var aux;

    /**
     * Creates an untrained hedger.
     *
     * @param setup the problem
     * @param objective the risk measure
     * @param level CVaR's alpha or the entropic lambda
     * @param seed the seed
     */
    public DeepHedging(Setup setup, Objective objective, double level, long seed) {
        this.setup = setup;
        this.objective = objective;
        this.level = level;
        this.policy = new Mlp(new int[] {4, 32, 32, 1}, new SplittableRandom(seed));
        this.aux = Var.param(Tensor.of(Shape.of(1), 0f));
    }

    /**
     * The Black-Scholes call price.
     *
     * @param s spot
     * @param k strike
     * @param tau time to maturity
     * @param sigma volatility
     * @return the price
     */
    public static double callPrice(double s, double k, double tau, double sigma) {
        if (tau <= 0) {
            return Math.max(s - k, 0);
        }
        double d1 = (Math.log(s / k) + 0.5 * sigma * sigma * tau) / (sigma * Math.sqrt(tau));
        return s * NormalTails.cdf(d1) - k * NormalTails.cdf(d1 - sigma * Math.sqrt(tau));
    }

    /**
     * The Black-Scholes delta.
     *
     * @param s spot
     * @param k strike
     * @param tau time to maturity
     * @param sigma volatility
     * @return {@code N(d1)}
     */
    public static double delta(double s, double k, double tau, double sigma) {
        if (tau <= 0) {
            return s > k ? 1 : 0;
        }
        return NormalTails.cdf((Math.log(s / k) + 0.5 * sigma * sigma * tau) / (sigma * Math.sqrt(tau)));
    }

    private static double gammaBs(double s, double k, double tau, double sigma) {
        double sd = sigma * Math.sqrt(Math.max(tau, 1e-8));
        double d1 = (Math.log(s / k) + 0.5 * sd * sd) / sd;
        return Math.exp(-0.5 * d1 * d1) / Math.sqrt(2 * Math.PI) / (s * sd);
    }

    private double dt(int steps) {
        return setup.maturity() / steps;
    }

    private Stats evaluate(double[][] prices, double[][] holdings) {
        int paths = prices.length;
        int steps = prices[0].length - 1;
        double premium = callPrice(prices[0][0], setup.strike(), setup.maturity(), setup.sigma());
        double[] pnl = new double[paths];
        double costs = 0;
        for (int p = 0; p < paths; p++) {
            double prev = 0;
            double v = premium - Math.max(prices[p][steps] - setup.strike(), 0);
            for (int k = 0; k < steps; k++) {
                double c = setup.cost() * Math.abs(holdings[p][k] - prev) * prices[p][k];
                v += holdings[p][k] * (prices[p][k + 1] - prices[p][k]) - c;
                costs += c / paths;
                prev = holdings[p][k];
            }
            double c = setup.cost() * Math.abs(prev) * prices[p][steps];
            v -= c;
            costs += c / paths;
            pnl[p] = v;
        }
        return stats(pnl, costs);
    }

    private static Stats stats(double[] pnl, double costs) {
        double mean = Arrays.stream(pnl).average().orElse(0);
        double var = 0;
        for (double v : pnl) {
            var += (v - mean) * (v - mean) / pnl.length;
        }
        double[] loss = Arrays.stream(pnl).map(v -> -v).sorted().toArray();
        int from = (int) Math.floor(0.95 * loss.length);
        double cvar = 0;
        for (int i = from; i < loss.length; i++) {
            cvar += loss[i] / (loss.length - from);
        }
        return new Stats(mean, Math.sqrt(var), cvar, costs);
    }

    /**
     * No hedge at all.
     *
     * @param prices {@code [path][step]}
     * @return the statistics
     */
    public Stats noHedge(double[][] prices) {
        return evaluate(prices, new double[prices.length][prices[0].length - 1]);
    }

    /**
     * Black-Scholes delta hedging at every date.
     *
     * @param prices {@code [path][step]}
     * @return the statistics
     */
    public Stats blackScholesDelta(double[][] prices) {
        int steps = prices[0].length - 1;
        double[][] h = new double[prices.length][steps];
        for (int p = 0; p < prices.length; p++) {
            for (int k = 0; k < steps; k++) {
                h[p][k] = delta(prices[p][k], setup.strike(), setup.maturity() - k * dt(steps), setup.sigma());
            }
        }
        return evaluate(prices, h);
    }

    /**
     * Delta hedging with the Whalley-Wilmott no-trade band
     * {@code |delta - h| <= (3 cost S Gamma^2 / (2 riskAversion))^(1/3)}: trade to the band edge
     * only when outside it.
     *
     * @param prices {@code [path][step]}
     * @param riskAversion the band's risk aversion
     * @return the statistics
     */
    public Stats whalleyWilmott(double[][] prices, double riskAversion) {
        int steps = prices[0].length - 1;
        double[][] h = new double[prices.length][steps];
        for (int p = 0; p < prices.length; p++) {
            double prev = 0;
            for (int k = 0; k < steps; k++) {
                double tau = setup.maturity() - k * dt(steps);
                double s = prices[p][k];
                double d = delta(s, setup.strike(), tau, setup.sigma());
                double g = gammaBs(s, setup.strike(), tau, setup.sigma());
                double band = Math.cbrt(1.5 * setup.cost() * s * g * g / riskAversion);
                prev = Math.max(d - band, Math.min(d + band, prev));
                h[p][k] = prev;
            }
        }
        return evaluate(prices, h);
    }

    private Var features(double[][] prices, double[][] vol, int k, Var prev, int from, int count) {
        int steps = prices[0].length - 1;
        float[] f = new float[count * 3];
        for (int i = 0; i < count; i++) {
            int p = from + i;
            f[i * 3] = (float) ((setup.maturity() - k * dt(steps)) / setup.maturity());
            f[i * 3 + 1] = (float) (Math.log(prices[p][k] / setup.strike()) / (setup.sigma()
                    * Math.sqrt(setup.maturity())));
            f[i * 3 + 2] = (float) (vol == null ? 1 : Math.sqrt(Math.max(vol[p][k], 0)) / setup.sigma());
        }
        Var x = Var.of(Tensor.of(Shape.of(count, 3), f));
        return Var.concat(1, x.slice(1, 0, 2), prev, x.slice(1, 2, 3));
    }

    private Var[] rollout(double[][] prices, double[][] vol, int from, int count) {
        int steps = prices[0].length - 1;
        double premium = callPrice(prices[0][0], setup.strike(), setup.maturity(), setup.sigma());
        Var prev = Var.of(Tensor.zeros(Shape.of(count, 1)));
        float[] pay = new float[count];
        for (int i = 0; i < count; i++) {
            pay[i] = (float) (premium - Math.max(prices[from + i][steps] - setup.strike(), 0));
        }
        Var pnl = Var.of(Tensor.of(Shape.of(count, 1), pay));
        Var cost = Var.of(Tensor.zeros(Shape.of(count, 1)));
        for (int k = 0; k <= steps; k++) {
            float[] s = new float[count];
            float[] ds = new float[count];
            for (int i = 0; i < count; i++) {
                s[i] = (float) prices[from + i][k];
                ds[i] = k < steps ? (float) (prices[from + i][k + 1] - prices[from + i][k]) : 0;
            }
            Tensor st = Tensor.of(Shape.of(count, 1), s);
            Var h = k < steps ? policy.forward(features(prices, vol, k, prev, from, count)).sigmoid().mul(1.2f)
                    .sub(0.1f) : Var.of(Tensor.zeros(Shape.of(count, 1)));
            Var c = h.sub(prev).abs().mul(st).mul((float) setup.cost());
            cost = cost.add(c);
            pnl = pnl.add(h.mul(Tensor.of(Shape.of(count, 1), ds))).sub(c);
            prev = h;
        }
        return new Var[] {pnl, cost};
    }

    private Var risk(Var pnl) {
        if (objective == Objective.CVAR) {
            return aux.add(pnl.neg().sub(aux).relu().mean().div((float) (1 - level)));
        }
        return pnl.mul((float) -level).logsumexp(0, false).sub((float) Math.log(pnl.shape().dim(0)))
                .div((float) level).sum();
    }

    /**
     * Trains the policy.
     *
     * @param prices training paths {@code [path][step]}
     * @param variances instantaneous variances (fed to the policy) or {@code null}
     * @param steps Adam steps
     * @param batch paths per step
     * @param learningRate the step size
     * @param seed the seed
     * @return the final batch risk
     */
    public double train(double[][] prices, double[][] variances, int steps, int batch, float learningRate,
            long seed) {
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom rnd = new SplittableRandom(seed);
        List<Var> params = new ArrayList<>(policy.parameters());
        params.add(aux);
        double last = 0;
        for (int step = 0; step < steps; step++) {
            int from = rnd.nextInt(prices.length - batch + 1);
            Var loss = risk(rollout(prices, variances, from, batch)[0]);
            loss.backward();
            last = loss.value().item();
            opt.step(params);
        }
        return last;
    }

    /**
     * The trained policy's statistics.
     *
     * @param prices test paths
     * @param variances variances or {@code null}
     * @return the statistics
     */
    public Stats hedge(double[][] prices, double[][] variances) {
        Var[] r = Var.noGrad(() -> rollout(prices, variances, 0, prices.length));
        float[] pnl = r[0].value().toArray();
        float[] cost = r[1].value().toArray();
        double[] p = new double[pnl.length];
        double c = 0;
        for (int i = 0; i < pnl.length; i++) {
            p[i] = pnl[i];
            c += cost[i] / pnl.length;
        }
        return stats(p, c);
    }

    /**
     * The indifference price under the training risk measure: the risk of the trained hedge of
     * the short call without premium.
     *
     * @param prices test paths
     * @param variances variances or {@code null}
     * @return the price
     */
    public double indifferencePrice(double[][] prices, double[][] variances) {
        double premium = callPrice(prices[0][0], setup.strike(), setup.maturity(), setup.sigma());
        return Var.noGrad(() -> risk(rollout(prices, variances, 0, prices.length)[0])).value().item() + premium;
    }

    /**
     * Convenience: GBM paths for the setup.
     *
     * @param paths paths
     * @param steps rebalancing dates
     * @param setup the problem
     * @param rnd the random source
     * @return prices
     */
    public static double[][] gbm(int paths, int steps, Setup setup, SplittableRandom rnd) {
        return MarketPaths.gbm(paths, steps, 1, 0, setup.sigma(), setup.maturity() / steps, rnd).prices();
    }
}
