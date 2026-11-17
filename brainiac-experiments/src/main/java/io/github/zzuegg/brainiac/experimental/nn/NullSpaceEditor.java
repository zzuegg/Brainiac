package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Null-space constrained knowledge editing (AlphaEdit). Like {@link RomeEditor}, a dense
 * layer is treated as a key-value memory and each edit writes a new value {@code v*} for a
 * subject's key. The update is first projected onto the null space of the keys that must
 * be preserved, so their outputs do not change at all. Edits made earlier in a sequence
 * are kept by a least-squares term on their keys.
 *
 * <p>The projector {@code P} comes from the eigenvectors of {@code K0^T K0} whose
 * eigenvalues are below {@code threshold} times the largest. For a new key {@code k} with
 * residual {@code r = v* - k W} and earlier edited keys {@code Kp}, the update is
 *
 * <pre>Δ = P (P (k^T k + Kp^T Kp) P + λ I)^-1 P k^T r</pre>
 *
 * <p>which is the paper's closed form written for row-vector keys ({@code x W}), with a
 * configurable ridge {@code λ} (the paper uses 1).
 *
 * <p>Fang et al. 2025, "AlphaEdit: Null-Space Constrained Knowledge Editing for Language
 * Models", <a href="https://arxiv.org/abs/2410.02355">arXiv:2410.02355</a>.
 */
public final class NullSpaceEditor {

    private final Model model;
    private final int layer;
    private final double[][] projector;
    private final double[][] editedGram;
    private final int nullity;
    private double ridge = 1;

    private NullSpaceEditor(Model model, int layer, double[][] projector, int nullity) {
        this.model = model;
        this.layer = layer;
        this.projector = projector;
        this.nullity = nullity;
        this.editedGram = new double[projector.length][projector.length];
    }

    /**
     * An editor for one dense layer.
     *
     * @param model a sequential classifier with probability outputs
     * @param layer the index of the dense layer to edit
     * @param preserved model inputs whose answers must not change
     * @param threshold eigenvalues below this share of the largest count as null
     * @return the editor
     */
    public static NullSpaceEditor of(Model model, int layer, Tensor preserved, double threshold) {
        Dense dense = RomeEditor.dense(model, layer);
        int d = dense.weight().shape().dim(0);
        float[] k = RomeEditor.key(model, layer, preserved);
        int n = k.length / d;
        double[][] gram = new double[d][d];
        for (int s = 0; s < n; s++) {
            for (int i = 0; i < d; i++) {
                double ki = k[s * d + i];
                for (int j = 0; j < d; j++) {
                    gram[i][j] += ki * k[s * d + j];
                }
            }
        }
        Tensor[] eig = Linalg.eigh(QuantMath.float64(gram));
        double[] values = eig[0].toDoubleArray();
        double[] vectors = eig[1].toDoubleArray();
        double cut = threshold * values[d - 1];
        double[][] p = new double[d][d];
        int nullity = 0;
        for (int e = 0; e < d; e++) {
            if (values[e] >= cut) {
                continue;
            }
            nullity++;
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    p[i][j] += vectors[i * d + e] * vectors[j * d + e];
                }
            }
        }
        return new NullSpaceEditor(model, layer, p, nullity);
    }

    /**
     * Sets the ridge term {@code I} of the closed form (default 1). Smaller values write
     * edits more fully when keys have small norms.
     *
     * @param ridge the ridge strength, positive
     * @return this editor
     */
    public NullSpaceEditor ridge(double ridge) {
        if (!(ridge > 0)) {
            throw new IllegalArgumentException("ridge must be positive: " + ridge);
        }
        this.ridge = ridge;
        return this;
    }

    /** @return the dimension of the null space the edits live in */
    public int nullity() {
        return nullity;
    }

    /**
     * Makes the model answer {@code target} for {@code subject}.
     *
     * @param subject one model input {@code [1, inputs]}
     * @param target the class wanted
     * @param steps optimisation steps for the new value
     * @return the edit
     */
    public RomeEditor.Edit edit(Tensor subject, int target, int steps) {
        Dense dense = RomeEditor.dense(model, layer);
        float[] k = RomeEditor.key(model, layer, subject);
        int in = k.length;
        float[] w = dense.weight().value().to(DType.FLOAT32).toArray();
        int out = w.length / in;
        float[] old = new float[out];
        for (int j = 0; j < out; j++) {
            for (int i = 0; i < in; i++) {
                old[j] += k[i] * w[i * out + j];
            }
        }
        float[] vStar = RomeEditor.targetValue(model, layer, old, target, steps);
        double[] pk = multiply(projector, k);
        for (int i = 0; i < in; i++) {
            for (int j = 0; j < in; j++) {
                editedGram[i][j] += k[i] * (double) k[j];
            }
        }
        // A = P (k^T k + Kp^T Kp) P + I, with k^T k already folded into editedGram
        double[][] a = sandwich(projector, editedGram);
        for (int i = 0; i < in; i++) {
            a[i][i] += ridge;
        }
        double[][] aInv = QuantMath.inverse(a);
        double[] u = multiply(projector, multiply(aInv, pk));
        for (int i = 0; i < in; i++) {
            for (int j = 0; j < out; j++) {
                w[i * out + j] += (float) (u[i] * (vStar[j] - old[j]));
            }
        }
        Tensor nw = Tensor.of(dense.weight().shape(), w, dense.weight().value().device());
        dense.weight().assign(nw.to(dense.weight().value().dtype()));
        return new RomeEditor.Edit(k, old, vStar);
    }

    private static double[] multiply(double[][] m, float[] v) {
        double[] d = new double[v.length];
        for (int i = 0; i < v.length; i++) {
            d[i] = v[i];
        }
        return multiply(m, d);
    }

    private static double[] multiply(double[][] m, double[] v) {
        double[] out = new double[m.length];
        for (int i = 0; i < m.length; i++) {
            double s = 0;
            for (int j = 0; j < v.length; j++) {
                s += m[i][j] * v[j];
            }
            out[i] = s;
        }
        return out;
    }

    /** {@code P G P} for symmetric {@code P}. */
    private static double[][] sandwich(double[][] p, double[][] g) {
        return product(product(p, g), p);
    }

    private static double[][] product(double[][] a, double[][] b) {
        int n = a.length;
        int m = b[0].length;
        double[][] out = new double[n][m];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < b.length; k++) {
                double aik = a[i][k];
                if (aik == 0) {
                    continue;
                }
                for (int j = 0; j < m; j++) {
                    out[i][j] += aik * b[k][j];
                }
            }
        }
        return out;
    }

    @Override
    public String toString() {
        return "NullSpaceEditor(layer " + layer + ", nullity " + nullity + ")";
    }
}
