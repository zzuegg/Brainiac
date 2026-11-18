package io.github.zzuegg.brainiac.experimental.interp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.IntFunction;

/**
 * Distributed alignment search (DAS, Geiger et al., CLeaR 2024): test whether a hidden
 * representation encodes a high-level causal variable (such as "the carry bit") even when it
 * is spread over many neurons. DAS learns an orthogonal rotation {@code R} of one layer's
 * activations and keeps its first {@code k} coordinates as the variable's subspace. An
 * <em>interchange intervention</em> runs the frozen model on a base input with those
 * coordinates replaced by their values on a source input,
 *
 * <pre>
 * h' = h_base + (h_source - h_base) U^T U,     U = the first k rows of R,
 * </pre>
 *
 * <p>and the rotation is trained so the model's output matches what the high-level causal
 * model predicts after setting the variable to its source value. Interchange-intervention
 * accuracy (IIA) measures how well the variable is localised.
 *
 * <p>{@code R} is a product of Householder reflections {@code H_i = I - 2 v v^T / |v|^2}
 * applied to the first {@code k} unit vectors, so {@code U} stays orthonormal by
 * construction and is trained by ordinary autograd. The model is any core {@link Model},
 * intervened on through {@link Model#forward(Var, boolean, Model.Hook)}. Fixed bases (a probe
 * direction, top principal components, single neurons) are scored by
 * {@link #interchangeAccuracy} for comparison. Unlike {@code LowRankIntervention}, which trains
 * a ReFT task intervention, nothing here edits what the model computes beyond swapping
 * coordinates between two runs.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2303.02536">arXiv:2303.02536</a>.
 */
public final class AlignmentSearch {

    /**
     * A batch of interchange examples.
     *
     * @param base base inputs
     * @param source source inputs
     * @param labels the high-level model's output after the interchange (class ids)
     */
    public record Interchange(Tensor base, Tensor source, Tensor labels) {
    }

    private final Model model;
    private final int step;
    private final int width;
    private final int k;
    private final List<Var> reflectors = new ArrayList<>();

    /**
     * Creates a search over one step of a model.
     *
     * @param model the frozen model
     * @param step the graph step whose output is intervened on
     * @param width that output's width
     * @param k the subspace size
     * @param reflectors Householder reflections ({@code width} reaches every rotation)
     * @param seed the seed
     */
    public AlignmentSearch(Model model, int step, int width, int k, int reflectors, long seed) {
        this.model = model;
        this.step = step;
        this.width = width;
        this.k = k;
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < reflectors; i++) {
            this.reflectors.add(Var.param(Tensor.normal(Shape.of(width, 1), 0, 1, r)));
        }
    }

    /** @return the subspace basis {@code U^T}, {@code [width, k]} with orthonormal columns */
    public Var basis() {
        float[] e = new float[width * k];
        for (int i = 0; i < k; i++) {
            e[i * k + i] = 1;
        }
        Var x = Var.of(Tensor.of(Shape.of(width, k), e));
        for (int i = reflectors.size() - 1; i >= 0; i--) {
            Var v = reflectors.get(i);
            Var vtx = v.transpose().matmul(x);
            x = x.sub(v.matmul(vtx).mul(2f).div(v.square().sum().add(1e-12f)));
        }
        return x;
    }

    /**
     * Logits of the model on {@code base} with the subspace swapped in from {@code source}.
     *
     * @param model the model
     * @param step the step
     * @param basis {@code [width, k]} orthonormal columns
     * @param base base inputs
     * @param source source inputs
     * @return the intervened logits
     */
    public static Var intervene(Model model, int step, Var basis, Tensor base, Tensor source) {
        Tensor hs = ModelStitcher.activation(model, step, source);
        return model.forward(Var.of(base), false, (s, out) -> {
            if (s.index() != step) {
                return out;
            }
            Var delta = Var.of(hs).sub(out.detach());
            return out.add(delta.matmul(basis).matmul(basis.transpose()));
        });
    }

    /**
     * Trains the rotation by interchange-intervention training.
     *
     * @param batches interchange batch {@code i}
     * @param steps Adam steps
     * @param learningRate the step size
     */
    public void train(IntFunction<Interchange> batches, int steps, float learningRate) {
        Optimizer opt = Optimizer.adam(learningRate);
        for (int s = 0; s < steps; s++) {
            Interchange b = batches.apply(s);
            Var loss = Functional.crossEntropy(intervene(model, step, basis(), b.base(), b.source()), b.labels());
            loss.backward();
            opt.step(reflectors);
            for (Var p : model.parameters()) {
                p.zeroGrad();
            }
        }
    }

    /**
     * Interchange-intervention accuracy of the learned subspace.
     *
     * @param batch interchange examples
     * @return the share whose intervened prediction matches the counterfactual label
     */
    public double interchangeAccuracy(Interchange batch) {
        Tensor u = Var.noGrad(this::basis).value();
        return interchangeAccuracy(model, step, u, batch);
    }

    /**
     * Interchange-intervention accuracy of a fixed basis.
     *
     * @param model the model
     * @param step the step
     * @param basis {@code [width, k]} orthonormal columns
     * @param batch interchange examples
     * @return the share whose intervened prediction matches the counterfactual label
     */
    public static double interchangeAccuracy(Model model, int step, Tensor basis, Interchange batch) {
        float[] pred = Var.noGrad(() -> intervene(model, step, Var.of(basis), batch.base(), batch.source()))
                .value().argmax(1).toArray();
        float[] y = batch.labels().toArray();
        int hit = 0;
        for (int i = 0; i < y.length; i++) {
            hit += pred[i] == y[i] ? 1 : 0;
        }
        return (double) hit / y.length;
    }

    /**
     * An orthonormal basis tensor from column vectors (Gram-Schmidt).
     *
     * @param columns {@code [k][width]} directions
     * @return {@code [width, k]}
     */
    public static Tensor orthonormal(double[][] columns) {
        int k = columns.length;
        int d = columns[0].length;
        double[][] q = new double[k][];
        for (int i = 0; i < k; i++) {
            double[] v = columns[i].clone();
            for (int j = 0; j < i; j++) {
                double c = InterpMath.dot(v, q[j]);
                for (int t = 0; t < d; t++) {
                    v[t] -= c * q[j][t];
                }
            }
            double n = Math.sqrt(Math.max(InterpMath.dot(v, v), 1e-300));
            for (int t = 0; t < d; t++) {
                v[t] /= n;
            }
            q[i] = v;
        }
        float[] out = new float[d * k];
        for (int t = 0; t < d; t++) {
            for (int i = 0; i < k; i++) {
                out[t * k + i] = (float) q[i][t];
            }
        }
        return Tensor.of(Shape.of(d, k), out);
    }
}
