package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * A trees-favouring tabular benchmark after Grinsztajn, Oyallon and Varoquaux, "Why do
 * tree-based models still outperform deep learning on tabular data?" (2022,
 * <a href="https://arxiv.org/abs/2207.08815">arXiv:2207.08815</a>). The paper names three
 * properties of real tables that favour trees, and this suite plants each one so their effect can
 * be measured:
 * <ul>
 *   <li><em>irregular targets</em>: the label is the sign of a sum of planted step functions
 *       {@code g_j(x_j)} of a few informative columns, so the decision boundary is not smooth;</li>
 *   <li><em>uninformative features</em>: a chosen share of the columns is pure noise;</li>
 *   <li><em>non-rotation-invariant data</em>: optionally every row is multiplied by one random
 *       orthogonal matrix, which mixes the columns. Trees split on single columns and lose their
 *       advantage; rotation-invariant MLPs do not care.</li>
 * </ul>
 * The planted shape functions are exposed through {@link #effect} so that additive models can be
 * checked against them.
 */
public final class TabularStressSuite {

    private final int informative;
    private final int columns;
    private final boolean rotate;
    private final double[][] steps;
    private final double[][] levels;
    private final double[][] rotation;

    /**
     * Creates a suite.
     *
     * @param informative the number of informative columns
     * @param uninformativeShare the share of all columns that are noise, in {@code [0, 1)}
     * @param rotate whether to rotate the feature space randomly
     * @param seed seeds the planted functions and the rotation
     */
    public TabularStressSuite(int informative, double uninformativeShare, boolean rotate, long seed) {
        if (uninformativeShare < 0 || uninformativeShare >= 1) {
            throw new IllegalArgumentException("uninformative share in [0, 1): " + uninformativeShare);
        }
        this.informative = informative;
        this.columns = Math.max(informative, (int) Math.round(informative / (1 - uninformativeShare)));
        this.rotate = rotate;
        SplittableRandom r = new SplittableRandom(seed);
        steps = new double[informative][];
        levels = new double[informative][];
        for (int j = 0; j < informative; j++) {
            int k = 3 + r.nextInt(3);
            steps[j] = new double[k];
            levels[j] = new double[k + 1];
            for (int s = 0; s < k; s++) {
                steps[j][s] = -1.6 + 3.2 * (s + 0.2 + 0.6 * r.nextDouble()) / k;
            }
            for (int s = 0; s <= k; s++) {
                levels[j][s] = (s % 2 == 0 ? 1 : -1) * (0.5 + r.nextDouble());
            }
        }
        rotation = rotate ? orthogonal(columns, r) : null;
    }

    /** @return the number of columns */
    public int columns() {
        return columns;
    }

    /** @return the number of informative columns (the first ones before any rotation) */
    public int informative() {
        return informative;
    }

    /** @return whether the feature space is rotated */
    public boolean rotated() {
        return rotate;
    }

    /**
     * The planted shape function of an informative column.
     *
     * @param column an informative column, before rotation
     * @param value the column's value
     * @return {@code g_column(value)}
     */
    public double effect(int column, double value) {
        int s = 0;
        while (s < steps[column].length && value >= steps[column][s]) {
            s++;
        }
        return levels[column][s];
    }

    /**
     * Draws rows: features uniform in {@code [-2, 2]}, label {@code sum_j g_j(x_j) + noise > 0}.
     *
     * @param rows the rows
     * @param seed the seed of the rows (the planted functions stay the same)
     * @return the table, features {@code [rows, columns]} and labels {@code [rows, 1]}
     */
    public SyntheticRecsys.Table table(int rows, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] x = new float[rows * columns];
        float[] y = new float[rows];
        double[] row = new double[columns];
        for (int i = 0; i < rows; i++) {
            double s = 0;
            for (int c = 0; c < columns; c++) {
                row[c] = -2 + 4 * r.nextDouble();
            }
            for (int j = 0; j < informative; j++) {
                s += effect(j, row[j]);
            }
            y[i] = s + 0.3 * r.nextGaussian() > 0 ? 1 : 0;
            for (int c = 0; c < columns; c++) {
                double v = row[c];
                if (rotate) {
                    v = 0;
                    for (int k = 0; k < columns; k++) {
                        v += row[k] * rotation[k][c];
                    }
                }
                x[i * columns + c] = (float) v;
            }
        }
        return new SyntheticRecsys.Table(Tensor.of(Shape.of(rows, columns), x), Tensor.of(Shape.of(rows, 1), y));
    }

    /** A random orthogonal matrix by Gram-Schmidt on a Gaussian matrix. */
    private static double[][] orthogonal(int n, SplittableRandom r) {
        double[][] q = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                q[i][j] = r.nextGaussian();
            }
            for (int k = 0; k < i; k++) {
                double dot = 0;
                for (int j = 0; j < n; j++) {
                    dot += q[i][j] * q[k][j];
                }
                for (int j = 0; j < n; j++) {
                    q[i][j] -= dot * q[k][j];
                }
            }
            double norm = 0;
            for (int j = 0; j < n; j++) {
                norm += q[i][j] * q[i][j];
            }
            norm = Math.sqrt(norm);
            for (int j = 0; j < n; j++) {
                q[i][j] /= norm;
            }
        }
        return q;
    }
}
