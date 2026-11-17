package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;

/**
 * Rank-One Model Editing: rewrites one association stored in an MLP layer with a single
 * rank-one weight update. The layer is viewed as a key-value memory: its input for a
 * subject is the key {@code k*}; ROME first finds the value {@code v*} that, put out by the
 * layer for that key, makes the rest of the model give the new answer (a small
 * optimisation), then changes the weights by
 *
 * <pre>W' = W + C^-1 k* (v* - k* W)^T / (k*^T C^-1 k*)</pre>
 *
 * <p>which maps {@code k*} exactly to {@code v*} while disturbing other keys as little as
 * possible, measured by the key covariance {@code C} ({@link #keyCovariance}).
 *
 * <p>Meng et al. 2022, "Locating and Editing Factual Associations in GPT",
 * <a href="https://arxiv.org/abs/2202.05262">arXiv:2202.05262</a>. Works on a {@link Dense}
 * layer of a sequential model; the key is the layer's input and the value its output before
 * bias and activation.
 */
public final class RomeEditor {

    /**
     * The outcome of an edit.
     *
     * @param key the subject's key
     * @param oldValue the layer's value for the key before the edit
     * @param newValue the value written
     */
    public record Edit(float[] key, float[] oldValue, float[] newValue) {
    }

    private RomeEditor() {
    }

    /**
     * The (regularised) second moment of a layer's inputs, {@code K^T K / n + lambda I}.
     *
     * @param model a sequential model
     * @param layer the index of the dense layer
     * @param inputs model inputs whose keys represent what must not change
     * @param lambda the ridge term
     * @return the matrix
     */
    public static double[][] keyCovariance(Model model, int layer, Tensor inputs, double lambda) {
        float[] k = key(model, layer, inputs);
        int d = dense(model, layer).weight().shape().dim(0);
        int n = k.length / d;
        double[][] c = new double[d][d];
        for (int s = 0; s < n; s++) {
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    c[i][j] += k[s * d + i] * k[s * d + j] / (double) n;
                }
            }
        }
        for (int i = 0; i < d; i++) {
            c[i][i] += lambda;
        }
        return c;
    }

    static Dense dense(Model model, int layer) {
        Layer l = model.layers().get(layer);
        if (!(l instanceof Dense d)) {
            throw new IllegalArgumentException("layer " + layer + " is not a dense layer: " + l);
        }
        return d;
    }

    static float[] key(Model model, int layer, Tensor inputs) {
        Var x = Var.of(inputs);
        if (layer > 0) {
            x = LayerHooks.trace(model, x).get(layer - 1);
        }
        return x.value().to(DType.FLOAT32).toArray();
    }

    /**
     * Makes the model answer {@code target} for {@code subject}.
     *
     * @param model a sequential classifier
     * @param layer the index of the dense layer to edit
     * @param subject one model input {@code [1, inputs]}
     * @param target the class the model should output for it
     * @param covariance the key covariance from {@link #keyCovariance}
     * @param steps optimisation steps for the new value
     * @return the edit
     */
    public static Edit edit(Model model, int layer, Tensor subject, int target, double[][] covariance, int steps) {
        Dense dense = dense(model, layer);
        float[] k = key(model, layer, subject);
        int in = k.length;
        float[] w = dense.weight().value().to(DType.FLOAT32).toArray();
        int out = w.length / in;
        float[] old = new float[out];
        for (int j = 0; j < out; j++) {
            for (int i = 0; i < in; i++) {
                old[j] += k[i] * w[i * out + j];
            }
        }
        float[] vStar = targetValue(model, layer, old, target, steps);
        // rank-one update: W += (C^-1 k) (v* - k W)^T / (k^T C^-1 k)
        double[] ck = solve(covariance, k);
        double denom = 0;
        for (int i = 0; i < in; i++) {
            denom += k[i] * ck[i];
        }
        for (int i = 0; i < in; i++) {
            for (int j = 0; j < out; j++) {
                w[i * out + j] += (float) (ck[i] * (vStar[j] - old[j]) / denom);
            }
        }
        Tensor nw = Tensor.of(dense.weight().shape(), w, dense.weight().value().device());
        dense.weight().assign(nw.to(dense.weight().value().dtype()));
        return new Edit(k, old, vStar);
    }

    /**
     * The layer value {@code v*} that makes the rest of the model answer {@code target}: a
     * small optimisation from the current value, with a weak pull back towards it.
     *
     * @param model a sequential classifier with probability outputs
     * @param layer the index of the dense layer
     * @param old the layer's current value for the subject (before bias and activation)
     * @param target the class wanted
     * @param steps optimisation steps
     * @return the value
     */
    public static float[] targetValue(Model model, int layer, float[] old, int target, int steps) {
        Dense dense = dense(model, layer);
        int out = old.length;
        Var v = Var.param(Tensor.of(Shape.of(1, out), old.clone()));
        Optimizer opt = Optimizer.adam(0.05f);
        Tensor label = Tensor.of(Shape.of(1), new float[] {target});
        Var bias = dense.bias();
        Tensor oldT = Tensor.of(Shape.of(1, out), old);
        for (int s = 0; s < steps; s++) {
            Var h = dense.activation().forward(bias == null ? v : v.add(bias.value()));
            Var logits = LayerHooks.resume(model, layer, h);
            Var loss = Functional.crossEntropy(logits.add(1e-6f).log(), label)
                    .add(v.sub(oldT).square().mean().mul(1e-3f));
            loss.backward();
            opt.step(List.of(v));
        }
        return v.value().to(DType.FLOAT32).toArray();
    }

    /** Solves {@code A x = b} in double precision. */
    static double[] solve(double[][] a, float[] b) {
        double[] r = new double[b.length];
        for (int i = 0; i < r.length; i++) {
            r[i] = b[i];
        }
        return QuantMath.solve(a, r);
    }
}
