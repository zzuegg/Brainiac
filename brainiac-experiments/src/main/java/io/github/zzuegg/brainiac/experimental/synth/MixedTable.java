package io.github.zzuegg.brainiac.experimental.synth;

import java.util.Arrays;

/**
 * A table with numeric and categorical columns, the format the tabular synthesisers read and
 * write.
 *
 * @param numeric numeric columns {@code [rows][numericColumns]}
 * @param categorical category codes {@code [rows][categoricalColumns]}
 * @param categories the number of categories of each categorical column
 */
public record MixedTable(double[][] numeric, int[][] categorical, int[] categories) {

    /** @return the number of rows */
    public int rows() {
        return numeric.length;
    }

    /** @return numeric columns */
    public int numericColumns() {
        return numeric.length == 0 ? 0 : numeric[0].length;
    }

    /** @return categorical columns */
    public int categoricalColumns() {
        return categories.length;
    }

    /**
     * Some rows.
     *
     * @param from the first row
     * @param to one past the last
     * @return those rows
     */
    public MixedTable rows(int from, int to) {
        return new MixedTable(Arrays.copyOfRange(numeric, from, to), Arrays.copyOfRange(categorical, from, to),
                categories);
    }

    /**
     * Per-column mean and standard deviation of the numeric columns.
     *
     * @return {@code {means, sds}}
     */
    public double[][] numericStats() {
        int d = numericColumns();
        double[] m = new double[d];
        double[] s = new double[d];
        for (double[] row : numeric) {
            for (int j = 0; j < d; j++) {
                m[j] += row[j] / rows();
            }
        }
        for (double[] row : numeric) {
            for (int j = 0; j < d; j++) {
                s[j] += (row[j] - m[j]) * (row[j] - m[j]) / rows();
            }
        }
        for (int j = 0; j < d; j++) {
            s[j] = Math.sqrt(Math.max(1e-12, s[j]));
        }
        return new double[][] {m, s};
    }

    /**
     * Rows as feature vectors: numeric columns standardised with the given statistics, then a
     * one-hot block per categorical column, leaving out the categorical columns listed in
     * {@code exclude}.
     *
     * @param stats {@code {means, sds}}
     * @param exclude categorical column to leave out (for example the label), or -1
     * @return {@code [rows][features]}
     */
    public double[][] encode(double[][] stats, int exclude) {
        int width = numericColumns();
        for (int c = 0; c < categories.length; c++) {
            width += c == exclude ? 0 : categories[c];
        }
        double[][] out = new double[rows()][width];
        for (int i = 0; i < rows(); i++) {
            int k = 0;
            for (int j = 0; j < numericColumns(); j++) {
                out[i][k++] = (numeric[i][j] - stats[0][j]) / stats[1][j];
            }
            for (int c = 0; c < categories.length; c++) {
                if (c == exclude) {
                    continue;
                }
                out[i][k + categorical[i][c]] = 1;
                k += categories[c];
            }
        }
        return out;
    }

    /**
     * One categorical column.
     *
     * @param c the column
     * @return its codes
     */
    public int[] column(int c) {
        int[] v = new int[rows()];
        for (int i = 0; i < v.length; i++) {
            v[i] = categorical[i][c];
        }
        return v;
    }
}
