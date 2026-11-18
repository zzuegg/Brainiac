package io.github.zzuegg.brainiac.experimental.ml;

import java.util.SplittableRandom;

/**
 * Kohonen's self-organising map (Kohonen 1990, <a href="https://ieeexplore.ieee.org/document/58325">The
 * self-organizing map</a>): prototypes on a fixed 2-D grid. Each sample pulls its best-matching
 * unit and, with a Gaussian weight in grid distance, the unit's grid neighbours towards it;
 * the learning rate and the neighbourhood radius shrink over training. Neighbouring units end
 * up with neighbouring prototypes, so the grid is a topology-preserving map of the data.
 *
 * <p>Scored by the quantisation error (mean distance to the best-matching unit) and the
 * topographic error (the share of samples whose two best units are not grid neighbours).
 * {@link GrowingNeuralGas} learns the topology instead of fixing it.
 */
public final class SelfOrganizingMap {

    private final int rows;
    private final int cols;
    private final double[][] prototypes;

    private SelfOrganizingMap(int rows, int cols, double[][] prototypes) {
        this.rows = rows;
        this.cols = cols;
        this.prototypes = prototypes;
    }

    /**
     * Trains a map online.
     *
     * @param x the data {@code [n][d]}
     * @param rows grid rows
     * @param cols grid columns
     * @param epochs passes over the data
     * @param seed the seed
     * @return the trained map
     */
    public static SelfOrganizingMap fit(double[][] x, int rows, int cols, int epochs, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int units = rows * cols;
        int d = x[0].length;
        double[][] w = new double[units][];
        for (int u = 0; u < units; u++) {
            w[u] = x[r.nextInt(x.length)].clone();
        }
        int total = epochs * x.length;
        double sigma0 = Math.max(rows, cols) / 2.0;
        int step = 0;
        int[] order = new int[x.length];
        for (int e = 0; e < epochs; e++) {
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            for (int i = order.length - 1; i > 0; i--) {
                int j = r.nextInt(i + 1);
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
            for (int idx : order) {
                double frac = step++ / (double) total;
                double lr = 0.5 * Math.pow(0.02, frac);
                double sigma = sigma0 * Math.pow(0.5 / sigma0, frac);
                int bmu = KMeans.nearest(w, x[idx]);
                int br = bmu / cols;
                int bc = bmu % cols;
                for (int u = 0; u < units; u++) {
                    int dr = u / cols - br;
                    int dc = u % cols - bc;
                    double h = Math.exp(-(dr * dr + dc * dc) / (2 * sigma * sigma));
                    if (h < 1e-4) {
                        continue;
                    }
                    for (int t = 0; t < d; t++) {
                        w[u][t] += lr * h * (x[idx][t] - w[u][t]);
                    }
                }
            }
        }
        return new SelfOrganizingMap(rows, cols, w);
    }

    /**
     * The mean Euclidean distance of samples to their best-matching unit.
     *
     * @param x samples
     * @return the quantisation error
     */
    public double quantizationError(double[][] x) {
        return meanDistance(prototypes, x);
    }

    /**
     * The mean Euclidean distance of samples to the nearest prototype.
     *
     * @param prototypes the prototypes
     * @param x samples
     * @return the mean distance
     */
    public static double meanDistance(double[][] prototypes, double[][] x) {
        double s = 0;
        for (double[] p : x) {
            s += Math.sqrt(MlMatrices.squaredDistance(p, prototypes[KMeans.nearest(prototypes, p)]));
        }
        return s / x.length;
    }

    /**
     * The share of samples whose best and second-best units are not adjacent on the grid
     * (4-neighbourhood).
     *
     * @param x samples
     * @return the topographic error in {@code [0, 1]}
     */
    public double topographicError(double[][] x) {
        int bad = 0;
        for (double[] p : x) {
            int[] two = twoNearest(prototypes, p);
            int dr = Math.abs(two[0] / cols - two[1] / cols);
            int dc = Math.abs(two[0] % cols - two[1] % cols);
            bad += dr + dc == 1 ? 0 : 1;
        }
        return bad / (double) x.length;
    }

    /**
     * The indices of the two nearest prototypes.
     *
     * @param prototypes the prototypes
     * @param p a point
     * @return {nearest, second nearest}
     */
    static int[] twoNearest(double[][] prototypes, double[] p) {
        int a = -1;
        int b = -1;
        double da = Double.POSITIVE_INFINITY;
        double db = Double.POSITIVE_INFINITY;
        for (int u = 0; u < prototypes.length; u++) {
            double d = MlMatrices.squaredDistance(prototypes[u], p);
            if (d < da) {
                b = a;
                db = da;
                a = u;
                da = d;
            } else if (d < db) {
                b = u;
                db = d;
            }
        }
        return new int[] {a, b};
    }

    /**
     * Returns the prototypes.
     *
     * @return {@code [rows * cols][d]}, row-major on the grid (copies)
     */
    public double[][] prototypes() {
        double[][] c = new double[prototypes.length][];
        for (int i = 0; i < c.length; i++) {
            c[i] = prototypes[i].clone();
        }
        return c;
    }

    /** @return the grid rows */
    public int rows() {
        return rows;
    }

    /** @return the grid columns */
    public int cols() {
        return cols;
    }
}
