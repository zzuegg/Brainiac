package io.github.zzuegg.brainiac.experimental.online;

import java.util.Arrays;

/**
 * Prediction with expert advice: every round each of {@code N} experts predicts, the learner
 * plays a weighted combination, then every expert's loss in {@code [0, 1]} is revealed. Regret
 * is measured against the best expert in hindsight.
 *
 * <ul>
 *   <li>{@link Kind#FOLLOW_THE_LEADER}: all weight on the expert with the smallest cumulative
 *       loss; regret can grow linearly on adversarial data.</li>
 *   <li>{@link Kind#HEDGE}: {@code w_i ∝ exp(-η L_i)} with a fixed rate (Freund and Schapire,
 *       JCSS 1997); regret {@code O(sqrt(T log N))} at {@code η = sqrt(8 ln N / T)}.</li>
 *   <li>{@link Kind#ADAHEDGE}: Hedge with {@code η_t = ln N / Δ_t-1}, where {@code Δ} sums
 *       the mixability gaps {@code h_t - m_t} between the learner's loss and the mix loss
 *       (de Rooij, van Erven, Grünwald and Koolen, JMLR 2014,
 *       <a href="https://arxiv.org/abs/1301.0534">arXiv:1301.0534</a>); no tuning, and constant
 *       regret when one expert is clearly best.</li>
 *   <li>{@link Kind#SQUINT}: weights {@code π_i E_η[η exp(η R_i - η^2 V_i)]} with the
 *       instantaneous regrets {@code r = h - ℓ_i}, their sum {@code R_i} and squared sum
 *       {@code V_i}, averaging over a geometric grid of rates in {@code (0, 1/2]} (Koolen and
 *       van Erven, COLT 2015); second-order, quantile regret bounds.</li>
 *   <li>{@link Kind#FIXED_SHARE}: Hedge followed by mixing a share {@code α} of the weight back
 *       to every expert (Herbster and Warmuth, Machine Learning 1998), so the learner tracks the
 *       best expert after it switches.</li>
 * </ul>
 */
public final class ExpertAdvice {

    /** The aggregation rule. */
    public enum Kind {
        /** follow the leader */
        FOLLOW_THE_LEADER,
        /** exponential weights with a fixed rate */
        HEDGE,
        /** exponential weights with an adaptive rate */
        ADAHEDGE,
        /** Squint with a discrete prior over rates */
        SQUINT,
        /** Hedge with fixed-share mixing */
        FIXED_SHARE
    }

    private final Kind kind;
    private final int n;
    private final double eta;
    private final double share;
    private final double[] cumulative;
    private final double[] regret;
    private final double[] squares;
    private final double[] shareWeights;
    private final double[] etas;
    private double gap;
    private double learnerLoss;
    private int rounds;

    private ExpertAdvice(Kind kind, int n, double eta, double share) {
        this.kind = kind;
        this.n = n;
        this.eta = eta;
        this.share = share;
        this.cumulative = new double[n];
        this.regret = new double[n];
        this.squares = new double[n];
        this.shareWeights = new double[n];
        Arrays.fill(shareWeights, 1.0 / n);
        this.etas = new double[12];
        for (int i = 0; i < etas.length; i++) {
            etas[i] = 0.5 / Math.pow(2, i * 0.5);
        }
    }

    /**
     * A learner.
     *
     * @param kind the rule
     * @param experts the number of experts
     * @param eta the Hedge and fixed-share rate (ignored by the others)
     * @param share the fixed-share mixing rate {@code α} (ignored by the others)
     * @return the learner
     */
    public static ExpertAdvice of(Kind kind, int experts, double eta, double share) {
        return new ExpertAdvice(kind, experts, eta, share);
    }

    /**
     * The Hedge rate tuned for a known horizon, {@code sqrt(8 ln N / T)}.
     *
     * @param experts {@code N}
     * @param horizon {@code T}
     * @return the rate
     */
    public static double tunedRate(int experts, int horizon) {
        return Math.sqrt(8 * Math.log(experts) / Math.max(1, horizon));
    }

    /** @return the current weights, summing to one */
    public double[] weights() {
        double[] w = new double[n];
        switch (kind) {
            case FOLLOW_THE_LEADER -> {
                double best = Arrays.stream(cumulative).min().orElse(0);
                int ties = 0;
                for (double c : cumulative) {
                    ties += c == best ? 1 : 0;
                }
                for (int i = 0; i < n; i++) {
                    w[i] = cumulative[i] == best ? 1.0 / ties : 0;
                }
                return w;
            }
            case HEDGE -> {
                return exponential(eta);
            }
            case ADAHEDGE -> {
                return exponential(gap <= 0 ? Double.POSITIVE_INFINITY : Math.log(n) / gap);
            }
            case FIXED_SHARE -> {
                return shareWeights.clone();
            }
            default -> {
                double sum = 0;
                for (int i = 0; i < n; i++) {
                    double v = 0;
                    for (double e : etas) {
                        v += e * Math.exp(Math.min(700, e * regret[i] - e * e * squares[i]));
                    }
                    w[i] = Math.max(0, v);
                    sum += w[i];
                }
                if (!(sum > 0)) {
                    Arrays.fill(w, 1.0 / n);
                    return w;
                }
                for (int i = 0; i < n; i++) {
                    w[i] /= sum;
                }
                return w;
            }
        }
    }

    private double[] exponential(double rate) {
        double best = Arrays.stream(cumulative).min().orElse(0);
        double[] w = new double[n];
        double sum = 0;
        for (int i = 0; i < n; i++) {
            w[i] = Double.isInfinite(rate) ? (cumulative[i] == best ? 1 : 0) : Math.exp(-rate * (cumulative[i] - best));
            sum += w[i];
        }
        for (int i = 0; i < n; i++) {
            w[i] /= sum;
        }
        return w;
    }

    /**
     * The combined prediction.
     *
     * @param predictions one prediction per expert
     * @return the weighted mean
     */
    public double predict(double[] predictions) {
        double[] w = weights();
        double s = 0;
        for (int i = 0; i < n; i++) {
            s += w[i] * predictions[i];
        }
        return s;
    }

    /**
     * Reveals the losses of a round.
     *
     * @param losses one loss in {@code [0, 1]} per expert
     */
    public void update(double[] losses) {
        double[] w = weights();
        double h = 0;
        for (int i = 0; i < n; i++) {
            h += w[i] * losses[i];
        }
        learnerLoss += h;
        if (kind == Kind.ADAHEDGE) {
            double rate = gap <= 0 ? Double.POSITIVE_INFINITY : Math.log(n) / gap;
            double min = Arrays.stream(losses).min().orElse(0);
            double mix;
            if (Double.isInfinite(rate)) {
                double best = Arrays.stream(cumulative).min().orElse(0);
                mix = Double.POSITIVE_INFINITY;
                for (int i = 0; i < n; i++) {
                    if (cumulative[i] == best) {
                        mix = Math.min(mix, losses[i]);
                    }
                }
            } else {
                double s = 0;
                for (int i = 0; i < n; i++) {
                    s += w[i] * Math.exp(-rate * (losses[i] - min));
                }
                mix = min - Math.log(s) / rate;
            }
            gap += Math.max(0, h - mix);
        }
        for (int i = 0; i < n; i++) {
            cumulative[i] += losses[i];
            double r = h - losses[i];
            regret[i] += r;
            squares[i] += r * r;
        }
        if (kind == Kind.FIXED_SHARE) {
            double sum = 0;
            for (int i = 0; i < n; i++) {
                shareWeights[i] *= Math.exp(-eta * losses[i]);
                sum += shareWeights[i];
            }
            for (int i = 0; i < n; i++) {
                shareWeights[i] = (1 - share) * shareWeights[i] / sum + share / n;
            }
        }
        rounds++;
    }

    /** @return the learner's cumulative (expected) loss */
    public double loss() {
        return learnerLoss;
    }

    /** @return the cumulative loss of each expert */
    public double[] expertLosses() {
        return cumulative.clone();
    }

    /** @return the regret against the best expert in hindsight */
    public double regret() {
        return learnerLoss - Arrays.stream(cumulative).min().orElse(0);
    }

    /** @return the rounds played */
    public int rounds() {
        return rounds;
    }

    /**
     * The loss of the best sequence of experts with at most {@code switches} switches, by
     * dynamic programming over a loss matrix.
     *
     * @param losses {@code [round][expert]}
     * @param switches the switch budget
     * @return the smallest cumulative loss
     */
    public static double bestSwitching(double[][] losses, int switches) {
        int k = losses[0].length;
        double[][] dp = new double[switches + 1][k];
        for (double[] row : losses) {
            double[][] next = new double[switches + 1][k];
            for (int s = 0; s <= switches; s++) {
                double bestPrev = Double.POSITIVE_INFINITY;
                if (s > 0) {
                    for (int j = 0; j < k; j++) {
                        bestPrev = Math.min(bestPrev, dp[s - 1][j]);
                    }
                }
                for (int j = 0; j < k; j++) {
                    next[s][j] = Math.min(dp[s][j], bestPrev) + row[j];
                }
            }
            dp = next;
        }
        double best = Double.POSITIVE_INFINITY;
        for (double[] row : dp) {
            for (double v : row) {
                best = Math.min(best, v);
            }
        }
        return best;
    }
}
