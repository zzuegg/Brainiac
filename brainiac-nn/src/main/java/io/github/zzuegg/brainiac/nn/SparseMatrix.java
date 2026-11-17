package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;

/**
 * A sparse matrix in coordinate form: the row, column and value of each stored entry. Its
 * product with a dense matrix gathers the dense rows its columns name, scales them and adds
 * them into its rows, so it runs on any device and is differentiable in the dense matrix and
 * in the values, as for graph adjacency, bag-of-words inputs and pruned weights. It stores
 * at least one entry, since tensors are never empty.
 *
 * @param rows the number of rows
 * @param cols the number of columns
 * @param rowIds each entry's row {@code [nnz]}
 * @param colIds each entry's column {@code [nnz]}
 * @param values each entry's value {@code [nnz]}
 */
public record SparseMatrix(int rows, int cols, Tensor rowIds, Tensor colIds, Var values) {

    /** Checks the parts agree. */
    public SparseMatrix {
        long nnz = values.shape().size();
        if (rowIds.size() != nnz || colIds.size() != nnz || values.shape().rank() != 1) {
            throw new IllegalArgumentException("rows " + rowIds.shape() + ", columns " + colIds.shape()
                    + " and values " + values.shape() + " must be [nnz]");
        }
    }

    /**
     * @param rows the number of rows
     * @param cols the number of columns
     * @param rowIds each entry's row
     * @param colIds each entry's column
     * @param values each entry's value
     * @param device where the parts live
     * @return the matrix, values fixed
     */
    public static SparseMatrix of(int rows, int cols, int[] rowIds, int[] colIds, float[] values, Device device) {
        int n = values.length;
        if (rowIds.length != n || colIds.length != n) {
            throw new IllegalArgumentException(n + " values need as many rows and columns");
        }
        for (int i = 0; i < n; i++) {
            if (rowIds[i] < 0 || rowIds[i] >= rows || colIds[i] < 0 || colIds[i] >= cols) {
                throw new IllegalArgumentException("entry " + i + " at (" + rowIds[i] + ", " + colIds[i]
                        + ") is outside " + rows + " x " + cols);
            }
        }
        return new SparseMatrix(rows, cols, ids(rowIds, device), ids(colIds, device),
                Var.of(Tensor.of(Shape.of(n), values).to(device)));
    }

    /**
     * @param dense a {@code [rows, cols]} matrix
     * @param threshold entries whose magnitude is at most this are dropped
     * @return its larger entries
     */
    public static SparseMatrix fromDense(Tensor dense, float threshold) {
        int r = dense.shape().dim(0);
        int c = dense.shape().dim(1);
        float[] v = dense.toArray();
        int[] ri = new int[v.length];
        int[] ci = new int[v.length];
        float[] vals = new float[v.length];
        int n = 0;
        for (int i = 0; i < v.length; i++) {
            if (Math.abs(v[i]) > threshold) {
                ri[n] = i / c;
                ci[n] = i % c;
                vals[n++] = v[i];
            }
        }
        if (n == 0) {
            n = 1;   // tensors cannot be empty: keep one explicit zero
            vals[0] = 0;
        }
        return of(r, c, Arrays.copyOf(ri, n), Arrays.copyOf(ci, n), Arrays.copyOf(vals, n), dense.device());
    }

    private static Tensor ids(int[] ids, Device device) {
        float[] f = new float[ids.length];
        for (int i = 0; i < ids.length; i++) {
            f[i] = ids[i];
        }
        return Tensor.of(Shape.of(ids.length), f).to(device);
    }

    /** @return how many entries are stored */
    public int nonZeros() {
        return Math.toIntExact(values.shape().size());
    }

    /**
     * @param newValues the values, such as trainable ones, {@code [nnz]}
     * @return this pattern with those values
     */
    public SparseMatrix withValues(Var newValues) {
        return new SparseMatrix(rows, cols, rowIds, colIds, newValues);
    }

    /** @return the transpose, sharing the parts */
    public SparseMatrix transpose() {
        return new SparseMatrix(cols, rows, colIds, rowIds, values);
    }

    /**
     * @param dense {@code [cols, n]}
     * @return {@code [rows, n]}, this times {@code dense}
     */
    public Var matmul(Var dense) {
        if (dense.shape().rank() != 2 || dense.shape().dim(0) != cols) {
            throw new IllegalArgumentException(rows + " x " + cols + " sparse times " + dense.shape());
        }
        return dense.gather(0, colIds).mul(values.reshape(nonZeros(), 1)).scatterAdd(0, rowIds, rows);
    }

    /** @return the matrix as a dense {@code [rows, cols]} value; repeated entries add up */
    public Var toDense() {
        Tensor flat = rowIds.mul(cols).add(colIds);
        return values.scatterAdd(0, flat, rows * cols).reshape(rows, cols);
    }
}
