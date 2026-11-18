package io.github.zzuegg.brainiac.experimental.edge;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.evolve.DifferentialEvolution;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Picks an {@link ElasticSupernet} subnet for a hardware budget, as once-for-all does: an MLP
 * accuracy predictor fitted to measured (subnet, accuracy) pairs, then an evolutionary search
 * (core-experimental {@link DifferentialEvolution} over a continuous encoding of the choices)
 * that maximises predicted accuracy subject to multiply-add and peak-memory limits computed from
 * the layer graph.
 *
 * <p>Paper: Cai et al., "Once-for-All", ICLR 2020,
 * <a href="https://arxiv.org/abs/1908.09791">arXiv:1908.09791</a>.
 */
public final class SubnetSearch {

    private final ElasticSupernet net;
    private final int[] kernels;
    private final int[] depths;
    private final int[] expands;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;
    private double mean;

    /**
     * Creates a search over a supernet's space.
     *
     * @param net the supernet
     * @param kernels kernel choices
     * @param depths depth choices
     * @param expands expansion choices
     * @param seed the predictor's seed
     */
    public SubnetSearch(ElasticSupernet net, int[] kernels, int[] depths, int[] expands, long seed) {
        this.net = net;
        this.kernels = kernels.clone();
        this.depths = depths.clone();
        this.expands = expands.clone();
        SplittableRandom r = new SplittableRandom(seed);
        int f = features();
        w1 = Var.param(Tensor.normal(Shape.of(f, 32), 0, (float) Math.sqrt(2.0 / f), r));
        b1 = Var.param(Tensor.zeros(Shape.of(32)));
        w2 = Var.param(Tensor.normal(Shape.of(32, 1), 0, 0.1f, r));
        b2 = Var.param(Tensor.zeros(Shape.of(1)));
    }

    private int features() {
        return net.stages() + 3 * net.slots();
    }

    /**
     * Encodes a subnet for the predictor: depth per stage, and per block slot whether it is
     * active, its kernel and its expansion (zero when inactive).
     *
     * @param c the subnet
     * @return the features
     */
    public float[] encode(ElasticSupernet.Config c) {
        float[] f = new float[features()];
        int maxD = net.maxDepth();
        int maxE = Arrays.stream(expands).max().orElse(1);
        for (int s = 0; s < net.stages(); s++) {
            f[s] = c.depths()[s] / (float) maxD;
            for (int b = 0; b < maxD; b++) {
                int slot = s * maxD + b;
                boolean on = b < c.depths()[s];
                int o = net.stages() + 3 * slot;
                f[o] = on ? 1 : 0;
                f[o + 1] = on && c.kernels()[slot] == 5 ? 1 : 0;
                f[o + 2] = on ? c.expands()[slot] / (float) maxE : 0;
            }
        }
        return f;
    }

    /**
     * Fits the accuracy predictor (squared error, Adam).
     *
     * @param configs measured subnets
     * @param accuracy their accuracies
     * @param steps training steps
     * @return the final training mean squared error
     */
    public double fit(List<ElasticSupernet.Config> configs, double[] accuracy, int steps) {
        int n = configs.size();
        int f = features();
        float[] x = new float[n * f];
        mean = Arrays.stream(accuracy).average().orElse(0);
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(encode(configs.get(i)), 0, x, i * f, f);
            y[i] = (float) (accuracy[i] - mean);
        }
        Tensor xt = Tensor.of(Shape.of(n, f), x);
        Tensor yt = Tensor.of(Shape.of(n, 1), y);
        Optimizer opt = Optimizer.adam(1e-2f);
        List<Var> p = List.of(w1, b1, w2, b2);
        double mse = 0;
        for (int s = 0; s < steps; s++) {
            p.forEach(Var::zeroGrad);
            Var loss = Var.of(xt).matmul(w1).add(b1).relu().matmul(w2).add(b2).sub(yt).square().mean();
            mse = loss.value().item();
            loss.backward();
            opt.step(p);
        }
        p.forEach(Var::zeroGrad);
        return mse;
    }

    /**
     * Predicted accuracy.
     *
     * @param c the subnet
     * @return the prediction
     */
    public double predict(ElasticSupernet.Config c) {
        float[] f = encode(c);
        return mean + Var.noGrad(() -> Var.of(Tensor.of(Shape.of(1, f.length), f)).matmul(w1).add(b1).relu()
                .matmul(w2).add(b2)).value().item();
    }

    /**
     * Decodes a point of the unit cube into a subnet.
     *
     * @param g genes in {@code [0, 1]}: one depth gene per stage, then a kernel and an expansion
     *     gene per slot
     * @return the subnet
     */
    public ElasticSupernet.Config decode(double[] g) {
        int[] d = new int[net.stages()];
        for (int s = 0; s < d.length; s++) {
            d[s] = depths[pick(g[s], depths.length)];
        }
        int[] k = new int[net.slots()];
        int[] e = new int[net.slots()];
        for (int i = 0; i < k.length; i++) {
            k[i] = kernels[pick(g[d.length + 2 * i], kernels.length)];
            e[i] = expands[pick(g[d.length + 2 * i + 1], expands.length)];
        }
        return new ElasticSupernet.Config(d, k, e);
    }

    private static int pick(double g, int n) {
        return Math.max(0, Math.min(n - 1, (int) Math.floor(g * n)));
    }

    /**
     * Whether a subnet fits a budget.
     *
     * @param c the subnet
     * @param maxMacs the multiply-add limit
     * @param maxPeakBytes the peak activation limit (int8)
     * @return whether it fits
     */
    public boolean fits(ElasticSupernet.Config c, long maxMacs, long maxPeakBytes) {
        return net.macs(c) <= maxMacs && net.peakBytes(c) <= maxPeakBytes;
    }

    /**
     * Evolutionary search for the subnet with the best predicted accuracy within a budget
     * (infeasible points are penalised by how far they overshoot).
     *
     * @param maxMacs the multiply-add limit
     * @param maxPeakBytes the peak activation limit
     * @param evaluations predictor evaluations
     * @param seed the seed
     * @return the best feasible subnet found
     */
    public ElasticSupernet.Config evolve(long maxMacs, long maxPeakBytes, int evaluations, long seed) {
        int n = net.stages() + 2 * net.slots();
        double[] lo = new double[n];
        double[] hi = new double[n];
        Arrays.fill(hi, 1);
        DifferentialEvolution de = DifferentialEvolution.of(lo, hi).population(24).seed(seed);
        de.minimize(g -> {
            ElasticSupernet.Config c = decode(g);
            double over = Math.max(0, net.macs(c) / (double) maxMacs - 1)
                    + Math.max(0, net.peakBytes(c) / (double) maxPeakBytes - 1);
            return over > 0 ? 10 + over : -predict(c);
        }, evaluations);
        return decode(de.best());
    }

    /**
     * Random search baseline: the best measured subnet among random feasible ones.
     *
     * @param maxMacs the multiply-add limit
     * @param maxPeakBytes the peak activation limit
     * @param samples subnets to measure
     * @param measure measures a subnet's accuracy
     * @param seed the seed
     * @return the best subnet found
     */
    public ElasticSupernet.Config randomSearch(long maxMacs, long maxPeakBytes, int samples,
            java.util.function.ToDoubleFunction<ElasticSupernet.Config> measure, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        ElasticSupernet.Config best = null;
        double bestAcc = -1;
        int found = 0;
        for (int tries = 0; found < samples && tries < 100 * samples; tries++) {
            ElasticSupernet.Config c = net.sample(r, kernels, depths, expands);
            if (!fits(c, maxMacs, maxPeakBytes)) {
                continue;
            }
            found++;
            double a = measure.applyAsDouble(c);
            if (a > bestAcc) {
                bestAcc = a;
                best = c;
            }
        }
        return best;
    }

    /**
     * Kendall's rank correlation between two score lists.
     *
     * @param a scores
     * @param b scores
     * @return tau in {@code [-1, 1]}
     */
    public static double kendallTau(double[] a, double[] b) {
        long conc = 0;
        long disc = 0;
        for (int i = 0; i < a.length; i++) {
            for (int j = i + 1; j < a.length; j++) {
                double s = Math.signum(a[i] - a[j]) * Math.signum(b[i] - b[j]);
                if (s > 0) {
                    conc++;
                } else if (s < 0) {
                    disc++;
                }
            }
        }
        return conc + disc == 0 ? 0 : (conc - disc) / (double) (conc + disc);
    }
}
