package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.experimental.train.RidgeRegression;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * RegMix (Liu et al. 2024, "RegMix: Data Mixture as Regression for Language Model
 * Pre-training", <a href="https://arxiv.org/abs/2407.01492">arXiv:2407.01492</a>): choose
 * the domain mixture of a large run from many tiny ones. Train small proxy models on
 * random mixtures (Dirichlet draws), fit a regression from mixture weights to the
 * target-domain validation loss, then search many candidate mixtures for the lowest
 * predicted loss and average the best few. The paper's proxies are 1M-parameter models on
 * 1B tokens, ranked by a LightGBM regressor; the ranking of mixtures transfers to models
 * 1000 times larger, and the result matches or beats DoReMi at a fraction of the compute.
 *
 * <p>The regression here is ridge regression ({@link RidgeRegression}) on the weights, their
 * squares and their pairwise products, enough to model the curvature that makes an
 * interior mixture optimal. Use it with {@link DataMixture#constant} for the final run.
 */
public final class RegressionMixer {

    private final int domains;
    private final double lambda;
    private final List<double[]> mixtures = new ArrayList<>();
    private final List<Double> losses = new ArrayList<>();
    private double[][] coefficients;

    /**
     * @param domains the number of domains
     * @param lambda the ridge penalty, for example 1e-3
     */
    public RegressionMixer(int domains, double lambda) {
        if (domains < 2 || lambda < 0) {
            throw new IllegalArgumentException("need at least two domains and lambda >= 0");
        }
        this.domains = domains;
        this.lambda = lambda;
    }

    /**
     * Random mixtures to train proxies on: Dirichlet draws with concentration
     * {@code alpha} (small values give peaked mixtures that explore the corners).
     *
     * @param n the mixtures
     * @param alpha the concentration, for example 1
     * @param seed the seed
     * @return {@code n} weight vectors summing to 1
     */
    public List<double[]> proposals(int n, double alpha, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<double[]> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            out.add(dirichlet(alpha, r));
        }
        return out;
    }

    private double[] dirichlet(double alpha, SplittableRandom r) {
        double[] w = new double[domains];
        double s = 0;
        for (int k = 0; k < domains; k++) {
            w[k] = gamma(alpha, r);
            s += w[k];
        }
        for (int k = 0; k < domains; k++) {
            w[k] /= s;
        }
        return w;
    }

    /** Marsaglia-Tsang gamma sampling (with the boost for shape below 1). */
    private static double gamma(double shape, SplittableRandom r) {
        if (shape < 1) {
            return gamma(shape + 1, r) * Math.pow(r.nextDouble(), 1 / shape);
        }
        double d = shape - 1 / 3.0;
        double c = 1 / Math.sqrt(9 * d);
        while (true) {
            double x;
            double v;
            do {
                x = gaussian(r);
                v = 1 + c * x;
            } while (v <= 0);
            v = v * v * v;
            double u = r.nextDouble();
            if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) {
                return d * v;
            }
        }
    }

    private static double gaussian(SplittableRandom r) {
        return Math.sqrt(-2 * Math.log(1 - r.nextDouble())) * Math.cos(2 * Math.PI * r.nextDouble());
    }

    /**
     * Records one proxy run.
     *
     * @param weights its mixture
     * @param targetLoss its loss on the target validation data
     */
    public void observe(double[] weights, double targetLoss) {
        if (weights.length != domains) {
            throw new IllegalArgumentException("expected " + domains + " weights");
        }
        mixtures.add(weights.clone());
        losses.add(targetLoss);
        coefficients = null;
    }

    private double[] features(double[] w) {
        double[] f = new double[1 + 2 * domains + domains * (domains - 1) / 2];
        int i = 0;
        f[i++] = 1;
        for (double x : w) {
            f[i++] = x;
        }
        for (double x : w) {
            f[i++] = x * x;
        }
        for (int a = 0; a < domains; a++) {
            for (int b = a + 1; b < domains; b++) {
                f[i++] = w[a] * w[b];
            }
        }
        return f;
    }

    /** Fits the regression to the runs observed so far. */
    public void fit() {
        if (mixtures.isEmpty()) {
            throw new IllegalStateException("no runs observed");
        }
        RidgeRegression ridge = RidgeRegression.of(features(mixtures.get(0)).length, 1);
        for (int i = 0; i < mixtures.size(); i++) {
            ridge.add(features(mixtures.get(i)), new double[] {losses.get(i)});
        }
        coefficients = ridge.solve(lambda);
    }

    /**
     * @param weights a mixture
     * @return the predicted target loss
     */
    public double predict(double[] weights) {
        if (coefficients == null) {
            fit();
        }
        return RidgeRegression.predict(coefficients, features(weights))[0];
    }

    /**
     * Searches Dirichlet candidates for the lowest predicted loss and averages the
     * {@code topK} best, as the paper does to reduce the regressor's noise.
     *
     * @param candidates the candidates scored
     * @param topK the best ones averaged
     * @param seed the seed
     * @return the chosen mixture
     */
    public double[] best(int candidates, int topK, long seed) {
        List<double[]> c = proposals(candidates, 1, seed);
        double[] pred = new double[c.size()];
        Integer[] order = new Integer[c.size()];
        for (int i = 0; i < pred.length; i++) {
            pred[i] = predict(c.get(i));
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(pred[a], pred[b]));
        double[] w = new double[domains];
        int k = Math.min(topK, c.size());
        for (int i = 0; i < k; i++) {
            double[] x = c.get(order[i]);
            for (int d = 0; d < domains; d++) {
                w[d] += x[d] / k;
            }
        }
        return w;
    }

    /** @return the proxy runs observed */
    public int observations() {
        return mixtures.size();
    }
}
