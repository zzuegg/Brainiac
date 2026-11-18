package io.github.zzuegg.brainiac.autograd;

import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Entropy-regularized optimal transport by Sinkhorn iterations in the log domain. Every step is a
 * {@link Var} operation, so gradients flow to the cost matrix (and through it to whatever produced
 * it). Runs on the cost's device.
 */
public final class OptimalTransport {

    private OptimalTransport() {
    }

    /**
     * Computes the transport plan between two distributions.
     *
     * @param cost the cost of moving mass from row i to column j, shape {@code [n, m]}
     * @param a the source weights, shape {@code [n]}, positive, summing to 1
     * @param b the target weights, shape {@code [m]}, positive, summing to 1
     * @param epsilon the entropy weight; smaller is closer to exact transport but needs more steps
     * @param iterations the Sinkhorn steps
     * @return the plan, shape {@code [n, m]}; its rows sum to {@code a} and columns to {@code b}
     */
    public static Var sinkhorn(Var cost, Tensor a, Tensor b, float epsilon, int iterations) {
        if (cost.shape().rank() != 2) {
            throw new IllegalArgumentException("the cost must be a matrix, got " + cost.shape());
        }
        int n = cost.shape().dim(0);
        int m = cost.shape().dim(1);
        if (a.shape().size() != n || b.shape().size() != m) {
            throw new IllegalArgumentException("weights of sizes " + a.shape().size() + " and " + b.shape().size()
                    + " do not fit a " + n + "x" + m + " cost");
        }
        if (!(epsilon > 0) || iterations < 1) {
            throw new IllegalArgumentException("epsilon must be positive and iterations at least 1");
        }
        Tensor logA = a.reshape(n, 1).log();
        Tensor logB = b.reshape(1, m).log();
        Var logK = cost.mul(-1f / epsilon);
        Var u = Var.of(logA);
        Var v = Var.of(logB.sub(logB));
        for (int i = 0; i < iterations; i++) {
            u = logK.add(v).logsumexp(1, true).neg().add(logA);
            v = logK.add(u).logsumexp(0, true).neg().add(logB);
        }
        return logK.add(u).add(v).exp();
    }

    /**
     * Computes the Sinkhorn transport cost: the sum of the plan times the cost.
     *
     * @param cost the cost matrix, shape {@code [n, m]}
     * @param a the source weights, shape {@code [n]}
     * @param b the target weights, shape {@code [m]}
     * @param epsilon the entropy weight
     * @param iterations the Sinkhorn steps
     * @return a scalar
     */
    public static Var sinkhornCost(Var cost, Tensor a, Tensor b, float epsilon, int iterations) {
        return sinkhorn(cost, a, b, epsilon, iterations).mul(cost).sum();
    }

    /**
     * Squared Euclidean distances between the rows of two point sets, a common transport cost.
     *
     * @param x points, shape {@code [n, d]}
     * @param y points, shape {@code [m, d]}
     * @return shape {@code [n, m]}
     */
    public static Var squaredDistances(Var x, Var y) {
        Var xx = x.mul(x).sum(1, true);
        Var yy = y.mul(y).sum(1, true).transpose();
        return xx.add(yy).sub(x.matmul(y.transpose()).mul(2f));
    }
}
