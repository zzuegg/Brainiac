package io.github.zzuegg.brainiac.experimental.stream;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.SlotOptimizer;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Parameter-free optimizers by coin betting, which need no learning rate: each coordinate is a
 * gambler that bets on the sign of the negative gradient, and its displacement from the starting
 * point is the bet, a fraction of the wealth won so far.
 * <ul>
 *   <li>{@link #cocob()} COCOB-Backprop: with {@code L} the largest gradient magnitude seen,
 *       {@code G} the sum of magnitudes, {@code theta} the sum of gradients and {@code Reward} the
 *       wealth won,
 *       <pre>
 *       Reward = max(Reward - (w - w1) g, 0)
 *       w = w1 - theta / (L max(G + L, alpha L)) (L + Reward)
 *       </pre>
 *       which adapts to the unknown distance to the optimum with optimal regret;</li>
 *   <li>{@link #kt()} Krichevsky-Trofimov betting: the gradient scaled by {@code L} is the coin's
 *       outcome {@code c}, {@code Wealth += c (w - w1)}, and the bet is
 *       {@code w - w1 = (sum c / (t + 1)) Wealth}.</li>
 * </ul>
 * Both implement the core {@link io.github.zzuegg.brainiac.train.Optimizer} interface; the
 * learning rate is ignored (kept at 1 for the interface).
 *
 * <p>Papers: Orabona and Tommasi, "Training Deep Networks without Learning Rates Through Coin
 * Betting", NeurIPS 2017, <a href="https://arxiv.org/abs/1705.07795">arXiv:1705.07795</a>;
 * Orabona and Pál, "Coin Betting and Parameter-Free Online Learning", NeurIPS 2016.
 */
public final class CoinBetting extends SlotOptimizer {

    private final boolean kt;
    private final float alpha;
    private final float initialWealth;

    private CoinBetting(boolean kt, float alpha, float initialWealth) {
        super(1, "w1", "L", "G", "reward", "theta");
        this.kt = kt;
        this.alpha = alpha;
        this.initialWealth = initialWealth;
    }

    /** @return COCOB-Backprop with {@code alpha = 100} */
    public static CoinBetting cocob() {
        return cocob(100);
    }

    /**
     * @param alpha the parameter that caps the early bets, 100 in the paper
     * @return COCOB-Backprop
     */
    public static CoinBetting cocob(float alpha) {
        return new CoinBetting(false, alpha, 0);
    }

    /** @return per-coordinate KT betting with initial wealth 1 */
    public static CoinBetting kt() {
        return kt(1);
    }

    /**
     * @param initialWealth each coordinate's starting wealth, in parameter units
     * @return per-coordinate KT betting
     */
    public static CoinBetting kt(float initialWealth) {
        return new CoinBetting(true, 0, initialWealth);
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor w = p.value();
        Tensor w1 = s[0] == null ? w.copy() : s[0];
        Tensor l = s[1] == null ? Tensor.full(g.shape(), 1e-8f) : s[1];
        Tensor big = s[2] == null ? Tensor.zeros(g.shape()) : s[2];
        Tensor reward = s[3] == null ? Tensor.zeros(g.shape()) : s[3];
        Tensor theta = s[4] == null ? Tensor.zeros(g.shape()) : s[4];
        Tensor x = w.sub(w1);
        l = l.maximum(g.abs());
        if (kt) {
            // reward is the wealth beyond the initial one; coin c = -g / L in [-1, 1]
            Tensor c = g.neg().div(l);
            reward = reward.add(c.mul(x));
            theta = theta.add(c);
            Tensor wealth = reward.add(initialWealth);
            p.assign(w1.add(theta.div(t + 1f).mul(wealth)));
            return new Tensor[] {w1, l, big, reward, theta};
        }
        big = big.add(g.abs());
        reward = reward.sub(x.mul(g)).relu();
        theta = theta.add(g);
        Tensor denom = l.mul(big.add(l).maximum(l.mul(alpha)));
        p.assign(w1.sub(theta.div(denom).mul(l.add(reward))));
        return new Tensor[] {w1, l, big, reward, theta};
    }

    @Override
    public String toString() {
        return kt ? "KT(wealth=" + initialWealth + ")" : "COCOB(alpha=" + alpha + ")";
    }
}
