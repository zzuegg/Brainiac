package io.github.zzuegg.brainiac.experimental.quadrature;

import io.github.zzuegg.brainiac.tensor.Matrices;

import java.util.ArrayList;
import java.util.List;
import java.util.function.ToDoubleFunction;

/**
 * Deterministic cubature for {@code N(0, I)}: Gauss-Hermite rules (probabilists' Hermite
 * polynomials) by the Golub-Welsch algorithm (nodes are the eigenvalues of the symmetric
 * tridiagonal Jacobi matrix with off-diagonal {@code sqrt(k)}, weights the squared first
 * components of its eigenvectors, via core {@link Matrices#symmetricEigen}), their tensor
 * {@link #product}, and {@link #smolyak} sparse grids (Gerstner and Griebel 1998) by the
 * combination technique {@code sum_{q-d+1 <= |i| <= q} (-1)^(q-|i|) C(d-1, q-|i|) U^(i_1) x
 * ... x U^(i_d)} with {@code 2i - 1} points at level {@code i}.
 */
public final class Cubature {

    private Cubature() {
    }

    /**
     * A cubature rule.
     *
     * @param nodes the nodes {@code [n][d]}
     * @param weights the weights (summing to 1)
     */
    public record Rule(double[][] nodes, double[] weights) {

        /**
         * Applies the rule.
         *
         * @param f the integrand
         * @return {@code sum_i w_i f(x_i)}
         */
        public double integrate(ToDoubleFunction<double[]> f) {
            double s = 0;
            for (int i = 0; i < weights.length; i++) {
                s += weights[i] * f.applyAsDouble(nodes[i]);
            }
            return s;
        }

        /**
         * The number of nodes.
         *
         * @return the size
         */
        public int size() {
            return weights.length;
        }
    }

    /**
     * The one-dimensional Gauss-Hermite rule for {@code N(0, 1)}.
     *
     * @param n the number of nodes
     * @return {@code {nodes, weights}}
     */
    public static double[][] gaussHermite(int n) {
        if (n == 1) {
            return new double[][] {{0}, {1}};
        }
        double[][] j = new double[n][n];
        for (int k = 1; k < n; k++) {
            j[k - 1][k] = Math.sqrt(k);
            j[k][k - 1] = Math.sqrt(k);
        }
        Matrices.SymmetricEigen e = Matrices.symmetricEigen(j);
        double[] x = e.values().clone();
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            w[i] = e.vectors()[0][i] * e.vectors()[0][i];
        }
        return new double[][] {x, w};
    }

    /**
     * The tensor-product Gauss-Hermite rule.
     *
     * @param n nodes per axis
     * @param d the dimension
     * @return the rule with {@code n^d} nodes
     */
    public static Rule product(int n, int d) {
        double[][] g = gaussHermite(n);
        int total = (int) Math.pow(n, d);
        double[][] nodes = new double[total][d];
        double[] w = new double[total];
        for (int i = 0; i < total; i++) {
            int rest = i;
            double wi = 1;
            for (int c = 0; c < d; c++) {
                int k = rest % n;
                rest /= n;
                nodes[i][c] = g[0][k];
                wi *= g[1][k];
            }
            w[i] = wi;
        }
        return new Rule(nodes, w);
    }

    /**
     * A Smolyak sparse grid.
     *
     * @param level the level {@code L >= 1} ({@code q = d + L - 1})
     * @param d the dimension
     * @return the rule (duplicate nodes are kept, with their combined signed weights)
     */
    public static Rule smolyak(int level, int d) {
        int q = d + level - 1;
        List<double[]> nodes = new ArrayList<>();
        List<Double> weights = new ArrayList<>();
        int[] idx = new int[d];
        java.util.Arrays.fill(idx, 1);
        while (true) {
            int sum = 0;
            for (int v : idx) {
                sum += v;
            }
            if (sum >= q - d + 1 && sum <= q) {
                double coef = ((q - sum) % 2 == 0 ? 1 : -1) * binomial(d - 1, q - sum);
                addTensor(idx, coef, nodes, weights);
            }
            int c = 0;
            while (c < d) {
                idx[c]++;
                if (idx[c] <= level) {
                    break;
                }
                idx[c] = 1;
                c++;
            }
            if (c == d) {
                break;
            }
        }
        double[] w = new double[weights.size()];
        for (int i = 0; i < w.length; i++) {
            w[i] = weights.get(i);
        }
        return new Rule(nodes.toArray(new double[0][]), w);
    }

    private static void addTensor(int[] levels, double coef, List<double[]> nodes, List<Double> weights) {
        int d = levels.length;
        double[][][] rules = new double[d][][];
        int total = 1;
        for (int c = 0; c < d; c++) {
            rules[c] = gaussHermite(2 * levels[c] - 1);
            total *= rules[c][0].length;
        }
        for (int i = 0; i < total; i++) {
            int rest = i;
            double[] x = new double[d];
            double w = coef;
            for (int c = 0; c < d; c++) {
                int m = rules[c][0].length;
                int k = rest % m;
                rest /= m;
                x[c] = rules[c][0][k];
                w *= rules[c][1][k];
            }
            nodes.add(x);
            weights.add(w);
        }
    }

    private static double binomial(int n, int k) {
        double r = 1;
        for (int i = 1; i <= k; i++) {
            r = r * (n - k + i) / i;
        }
        return r;
    }
}
