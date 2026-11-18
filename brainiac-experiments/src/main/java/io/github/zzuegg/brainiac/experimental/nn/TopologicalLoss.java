package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.PersistentHomology;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Differentiable 0-dimensional persistence losses (Moor et al. 2020,
 * <a href="https://arxiv.org/abs/1906.00722">arXiv:1906.00722</a>; Gabrielsson et al. 2020,
 * <a href="https://arxiv.org/abs/1905.12200">arXiv:1905.12200</a>).
 *
 * <p>Every finite {@code H0} death of a Rips filtration is the length of one minimum spanning
 * tree edge ({@link PersistentHomology#h0}). The edges are chosen without gradient, and the
 * distances along them are gathered from the points differentiably, so the gradient of a death
 * flows to the two points that created it.
 */
public final class TopologicalLoss {

    private TopologicalLoss() {
    }

    private static double[][] distances(float[] v, int n, int d) {
        double[][] out = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double s = 0;
                for (int k = 0; k < d; k++) {
                    double t = v[i * d + k] - v[j * d + k];
                    s += t * t;
                }
                out[i][j] = Math.sqrt(s);
                out[j][i] = out[i][j];
            }
        }
        return out;
    }

    private static double max(double[][] d) {
        double m = 1e-12;
        for (double[] row : d) {
            for (double v : row) {
                m = Math.max(m, v);
            }
        }
        return m;
    }

    private static int[][] mstEdges(Tensor points) {
        int n = points.shape().dim(0);
        List<PersistentHomology.Pair> pairs = PersistentHomology.h0(distances(points.toArray(), n,
                points.shape().dim(1)));
        int[][] edges = new int[pairs.size()][];
        for (int k = 0; k < edges.length; k++) {
            edges[k] = pairs.get(k).deathEdge();
        }
        return edges;
    }

    /**
     * Distances between pairs of points, differentiable in the points.
     *
     * @param points {@code [n, d]}
     * @param edges the pairs {@code {i, j}}
     * @return {@code [edges]}
     */
    public static Var pairDistances(Var points, int[][] edges) {
        float[] a = new float[edges.length];
        float[] b = new float[edges.length];
        for (int k = 0; k < edges.length; k++) {
            a[k] = edges[k][0];
            b[k] = edges[k][1];
        }
        Var pa = points.gather(0, Tensor.of(Shape.of(edges.length), a));
        Var pb = points.gather(0, Tensor.of(Shape.of(edges.length), b));
        return pa.sub(pb).square().sum(1, false).add(1e-9f).sqrt();
    }

    /**
     * The topological autoencoder loss: along the persistence (MST) edges of the input, the
     * latent distances should match the input distances, and vice versa:
     * {@code |A_X[pi_X] - A_Z[pi_X]|^2 / 2 + |A_Z[pi_Z] - A_X[pi_Z]|^2 / 2}, averaged over edges.
     * Both distance matrices are divided by their largest entry in the batch (a constant), so
     * the latent matches the input's shape, not its scale.
     *
     * @param input the input batch {@code [n, D]} (constant)
     * @param latent the latent batch {@code [n, d]}
     * @return the scalar loss
     */
    public static Var autoencoder(Tensor input, Var latent) {
        int n = input.shape().dim(0);
        double[][] ax = distances(input.toArray(), n, input.shape().dim(1));
        double[][] az = distances(latent.value().toArray(), n, latent.shape().dim(1));
        float sx = (float) (1 / max(ax));
        float sz = (float) (1 / max(az));
        int[][] px = mstEdges(input);
        int[][] pz = mstEdges(latent.value());
        Var x = Var.of(input);
        Var xToZ = pairDistances(latent, px).mul(sz).sub(pairDistances(x, px).mul(sx)).square().mean();
        float[] target = new float[pz.length];
        for (int k = 0; k < pz.length; k++) {
            target[k] = (float) ax[pz[k][0]][pz[k][1]] * sx;
        }
        Var zToX = pairDistances(latent, pz).mul(sz).sub(Var.of(Tensor.of(Shape.of(pz.length), target))).square()
                .mean();
        return xToZ.add(zToX).mul(0.5f);
    }

    /**
     * The total {@code H0} persistence of a point set: the sum of its MST edge lengths beyond
     * the {@code keep} longest. Minimising it pulls the points into at most {@code keep + 1}
     * clusters; maximising the {@code keep} longest (with a negative weight) pushes them apart.
     *
     * @param points {@code [n, d]}
     * @param keep the number of longest bars to leave out
     * @return the scalar sum
     */
    public static Var componentPersistence(Var points, int keep) {
        int[][] edges = mstEdges(points.value());
        int[][] shortest = java.util.Arrays.copyOf(edges, Math.max(0, edges.length - keep));
        return shortest.length == 0 ? points.mul(0f).sum() : pairDistances(points, shortest).sum();
    }
}
