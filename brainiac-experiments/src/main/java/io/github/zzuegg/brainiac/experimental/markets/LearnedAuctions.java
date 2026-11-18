package io.github.zzuegg.brainiac.experimental.markets;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Learned multi-item auctions for additive bidders with independent private values.
 *
 * <ul>
 *   <li>{@link Kind#REGRET_NET}: MLPs map all bids to the allocation (a softmax over bidders
 *       plus a dummy per item) and to payments as a fraction of each bidder's reported value of
 *       her bundle (individually rational by construction); training maximises revenue under an
 *       augmented Lagrangian on each bidder's ex-post regret, found by gradient ascent on
 *       misreports (Dutting et al., ICML 2019,
 *       <a href="https://arxiv.org/abs/1706.03459">arXiv:1706.03459</a>).</li>
 *   <li>{@link Kind#REGRET_FORMER}: the same mechanism head on a permutation-equivariant body of
 *       alternating attention across items (within a bidder) and across bidders (within an
 *       item) over per-cell bid embeddings (Ivanov et al., NeurIPS 2022,
 *       <a href="https://arxiv.org/abs/2202.13110">arXiv:2202.13110</a>), with the Lagrangian
 *       regret penalty in place of its regret-budget loss.</li>
 * </ul>
 *
 * <p>Truthful baselines: {@link #secondPrice} per item with a reserve (Myerson's optimum for
 * one item and uniform values at reserve 1/2, and separate per-item Myerson for several), and
 * {@link #manelliVincent}, the optimal menu for one bidder and two uniform items.
 */
public final class LearnedAuctions {

    /** The architecture. */
    public enum Kind {
        /** fully connected allocation and payment networks. */
        REGRET_NET,
        /** attention over bidders and items. */
        REGRET_FORMER
    }

    /**
     * Mechanism quality on a sample.
     *
     * @param revenue mean revenue
     * @param meanRegret mean ex-post regret per bidder
     * @param maxRegret the largest mean regret over bidders
     */
    public record Result(double revenue, double meanRegret, double maxRegret) {
    }

    private final Kind kind;
    private final int bidders;
    private final int items;
    private final int width;
    private final List<Var> params = new ArrayList<>();
    private Mlp alloc;
    private Mlp pay;
    private Var embed;
    private final List<Var[]> blocks = new ArrayList<>();
    private Var allocHead;
    private Var payHead;
    private Var payBias;

    /**
     * Creates an untrained mechanism.
     *
     * @param kind the architecture
     * @param bidders bidders
     * @param items items
     * @param width hidden width (MLP units or attention dimension)
     * @param seed the seed
     */
    public LearnedAuctions(Kind kind, int bidders, int items, int width, long seed) {
        this.kind = kind;
        this.bidders = bidders;
        this.items = items;
        this.width = width;
        SplittableRandom rnd = new SplittableRandom(seed);
        int cells = bidders * items;
        if (kind == Kind.REGRET_NET) {
            alloc = new Mlp(new int[] {cells, width, width, (bidders + 1) * items}, rnd);
            pay = new Mlp(new int[] {cells, width, width, bidders}, rnd);
            params.addAll(alloc.parameters());
            params.addAll(pay.parameters());
        } else {
            embed = Var.param(Mlp.random(3, width, rnd));
            params.add(embed);
            for (int b = 0; b < 2; b++) {
                Var[] w = new Var[8];
                for (int i = 0; i < 6; i++) {
                    w[i] = Var.param(Mlp.random(width, width, rnd));
                }
                w[6] = Var.param(Mlp.random(width, width, rnd));
                w[7] = Var.param(Mlp.random(width, width, rnd));
                blocks.add(w);
                params.addAll(List.of(w));
            }
            allocHead = Var.param(Mlp.random(width, 1, rnd));
            payHead = Var.param(Mlp.random(width, 1, rnd));
            payBias = Var.param(Tensor.zeros(Shape.of(1)));
            params.addAll(List.of(allocHead, payHead, payBias));
        }
    }

    /** @return the number of learned scalars */
    public long parameterCount() {
        long c = 0;
        for (Var v : params) {
            c += v.value().size();
        }
        return c;
    }

    /**
     * The mechanism on a batch of bids.
     *
     * @param bids {@code [batch, bidders, items]}
     * @return {@code {allocation [batch, bidders, items], payment [batch, bidders]}}
     */
    public Var[] mechanism(Var bids) {
        int batch = bids.shape().dim(0);
        Var logits;
        Var frac;
        if (kind == Kind.REGRET_NET) {
            Var flat = bids.reshape(batch, bidders * items);
            logits = alloc.forward(flat).reshape(batch, bidders + 1, items);
            frac = pay.forward(flat).sigmoid();
        } else {
            Var rowMean = bids.mean(2, true).expand(Shape.of(batch, bidders, items));
            Var colMean = bids.mean(1, true).expand(Shape.of(batch, bidders, items));
            Var in = Var.concat(3, bids.reshape(batch, bidders, items, 1), rowMean.reshape(batch, bidders, items, 1),
                    colMean.reshape(batch, bidders, items, 1));
            Var h = in.reshape(batch * bidders * items, 3).matmul(embed).reshape(batch, bidders, items, width);
            for (Var[] w : blocks) {
                h = h.add(attend(h, w[0], w[1], w[2], batch * bidders, items));
                Var t = h.permute(0, 2, 1, 3);
                t = t.add(attend(t, w[3], w[4], w[5], batch * items, bidders));
                h = t.permute(0, 2, 1, 3);
                Var flat = h.reshape(batch * bidders * items, width);
                h = h.add(flat.matmul(w[6]).tanh().matmul(w[7]).reshape(batch, bidders, items, width));
            }
            Var cell = h.reshape(batch * bidders * items, width).matmul(allocHead).reshape(batch, bidders, items);
            logits = Var.concat(1, cell, Var.of(Tensor.zeros(Shape.of(batch, 1, items))));
            frac = h.mean(2, false).reshape(batch * bidders, width).matmul(payHead).add(payBias)
                    .reshape(batch, bidders).sigmoid();
        }
        Var allocation = logits.softmax(1).slice(1, 0, bidders);
        Var payment = frac.mul(allocation.mul(bids).sum(2, false));
        return new Var[] {allocation, payment};
    }

    private Var attend(Var h, Var wq, Var wk, Var wv, int groups, int length) {
        Var flat = h.reshape(groups * length, width);
        Var q = flat.matmul(wq).reshape(groups, 1, length, width);
        Var k = flat.matmul(wk).reshape(groups, 1, length, width);
        Var v = flat.matmul(wv).reshape(groups, 1, length, width);
        Var a = Functional.attention(q, k, v, null, (float) (1 / Math.sqrt(width)), 0f, null);
        return a.reshape(h.shape().dim(0), h.shape().dim(1), length, width);
    }

    private static Tensor tensor(double[][][] v, int from, int count) {
        int n = v[0].length;
        int m = v[0][0].length;
        float[] f = new float[count * n * m];
        for (int s = 0; s < count; s++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < m; j++) {
                    f[(s * n + i) * m + j] = (float) v[from + s][i][j];
                }
            }
        }
        return Tensor.of(Shape.of(count, n, m), f);
    }

    /** Bids with bidder {@code i}'s row replaced by the misreport {@code r [batch, 1, items]}. */
    private Var replace(Tensor values, int i, Var r) {
        Var v = Var.of(values);
        List<Var> rows = new ArrayList<>();
        for (int b = 0; b < bidders; b++) {
            rows.add(b == i ? r : v.slice(1, b, b + 1));
        }
        return Var.concat(1, rows.toArray(new Var[0]));
    }

    private Var utility(Tensor values, int i, Var bids) {
        Var[] mech = mechanism(bids);
        Var vi = Var.of(values).slice(1, i, i + 1);
        return mech[0].slice(1, i, i + 1).mul(vi).sum(2, false).reshape(values.shape().dim(0))
                .sub(mech[1].slice(1, i, i + 1).reshape(values.shape().dim(0)));
    }

    /** The best misreports found by projected gradient ascent from several starts. */
    private Tensor[] misreports(Tensor values, int steps, int restarts, float rate, SplittableRandom rnd) {
        int batch = values.shape().dim(0);
        Tensor[] best = new Tensor[bidders];
        float[][] bestU = new float[bidders][];
        for (int i = 0; i < bidders; i++) {
            float[] vf = values.toArray();
            for (int start = 0; start < restarts; start++) {
                float[] r = new float[batch * items];
                for (int s = 0; s < batch; s++) {
                    for (int j = 0; j < items; j++) {
                        r[s * items + j] = start == 0 ? vf[(s * bidders + i) * items + j] : (float) rnd.nextDouble();
                    }
                }
                for (int it = 0; it <= steps; it++) {
                    Var rv = Var.param(Tensor.of(Shape.of(batch, 1, items), r));
                    Var u = utility(values, i, replace(values, i, rv));
                    if (it == steps) {
                        float[] uf = u.value().toArray();
                        if (bestU[i] == null) {
                            bestU[i] = uf;
                            best[i] = rv.value();
                        } else {
                            float[] cur = best[i].toArray();
                            for (int s = 0; s < batch; s++) {
                                if (uf[s] > bestU[i][s]) {
                                    bestU[i][s] = uf[s];
                                    System.arraycopy(r, s * items, cur, s * items, items);
                                }
                            }
                            best[i] = Tensor.of(Shape.of(batch, 1, items), cur);
                        }
                        break;
                    }
                    float[] g = Var.gradients(u.sum(), List.of(rv), false).get(0).value().toArray();
                    for (int k = 0; k < r.length; k++) {
                        r[k] = Math.max(0, Math.min(1, r[k] + rate * g[k]));
                    }
                }
            }
        }
        return best;
    }

    private Var[] regrets(Tensor values, Tensor[] mis) {
        Var[] out = new Var[bidders];
        for (int i = 0; i < bidders; i++) {
            Var truthful = utility(values, i, Var.of(values));
            Var deviate = utility(values, i, replace(values, i, Var.of(mis[i])));
            out[i] = deviate.sub(truthful).relu().mean();
        }
        return out;
    }

    /**
     * Trains by the augmented Lagrangian on regret.
     *
     * @param values training valuations {@code [sample][bidder][item]}
     * @param steps outer steps
     * @param batch valuations per step
     * @param misreportSteps inner gradient-ascent steps
     * @param rho the quadratic penalty weight (the multipliers start at {@code rho / 4} and are
     *     updated every 10 steps)
     * @param learningRate the Adam step size
     * @param seed the seed
     */
    public void train(double[][][] values, int steps, int batch, int misreportSteps, double rho, float learningRate,
            long seed) {
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom rnd = new SplittableRandom(seed);
        double[] lambda = new double[bidders];
        Arrays.fill(lambda, rho / 4);
        double[] acc = new double[bidders];
        for (int step = 0; step < steps; step++) {
            int from = rnd.nextInt(values.length - batch + 1);
            Tensor v = tensor(values, from, batch);
            Tensor[] mis = misreports(v, misreportSteps, 1, 0.1f, rnd);
            Var[] rg = regrets(v, mis);
            Var revenue = mechanism(Var.of(v))[1].sum(1, false).mean();
            Var loss = revenue.neg();
            for (int i = 0; i < bidders; i++) {
                loss = loss.add(rg[i].mul((float) lambda[i])).add(rg[i].square().mul((float) (rho / 2)));
                acc[i] += rg[i].value().item();
            }
            loss.backward();
            opt.step(params);
            if ((step + 1) % 10 == 0) {
                for (int i = 0; i < bidders; i++) {
                    lambda[i] += rho * acc[i] / 10;
                    acc[i] = 0;
                }
            }
        }
    }

    /**
     * Revenue and regret on test valuations (misreports from the truth and two random starts).
     *
     * @param values test valuations
     * @param misreportSteps gradient-ascent steps per start
     * @param seed the seed
     * @return the result
     */
    public Result evaluate(double[][][] values, int misreportSteps, long seed) {
        Tensor v = tensor(values, 0, values.length);
        Tensor[] mis = misreports(v, misreportSteps, 3, 0.05f, new SplittableRandom(seed));
        return Var.noGrad(() -> {
            Var[] rg = regrets(v, mis);
            double rev = mechanism(Var.of(v))[1].sum(1, false).mean().value().item();
            double mean = 0;
            double max = 0;
            for (Var r : rg) {
                mean += r.value().item() / bidders;
                max = Math.max(max, r.value().item());
            }
            return new Result(rev, mean, max);
        });
    }

    /**
     * Per-item second-price auctions with a reserve (a posted price for one bidder).
     *
     * @param values valuations {@code [sample][bidder][item]}
     * @param reserve the reserve price
     * @return the mean revenue
     */
    public static double secondPrice(double[][][] values, double reserve) {
        double total = 0;
        for (double[][] s : values) {
            for (int j = 0; j < s[0].length; j++) {
                double first = 0;
                double second = 0;
                for (double[] b : s) {
                    if (b[j] > first) {
                        second = first;
                        first = b[j];
                    } else if (b[j] > second) {
                        second = b[j];
                    }
                }
                total += first >= reserve ? Math.max(second, reserve) : 0;
            }
        }
        return total / values.length;
    }

    /**
     * The Manelli-Vincent menu for one bidder and two items with values uniform on {@code [0, 1]}:
     * each item at 2/3 or both at {@code (4 - sqrt 2) / 3}.
     *
     * @param values valuations {@code [sample][1][2]}
     * @return the mean revenue
     */
    public static double manelliVincent(double[][][] values) {
        double single = 2.0 / 3;
        double bundle = (4 - Math.sqrt(2)) / 3;
        double total = 0;
        for (double[][] s : values) {
            double a = s[0][0];
            double b = s[0][1];
            double[] utility = {0, a - single, b - single, a + b - bundle};
            double[] price = {0, single, single, bundle};
            int best = 0;
            for (int o = 1; o < 4; o++) {
                if (utility[o] > utility[best] + 1e-12 || Math.abs(utility[o] - utility[best]) <= 1e-12
                        && price[o] > price[best]) {
                    best = o;
                }
            }
            total += price[best];
        }
        return total / values.length;
    }
}
