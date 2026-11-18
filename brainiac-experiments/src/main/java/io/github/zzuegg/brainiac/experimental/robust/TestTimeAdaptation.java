package io.github.zzuegg.brainiac.experimental.robust;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.BatchNorm;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.function.UnaryOperator;

/**
 * Test-time adaptation of a trained classifier to covariate shift, without labels:
 *
 * <ul>
 *   <li>{@link Method#BN_STATS}: normalise with the test batch's statistics instead of the
 *       training running averages (Schneider et al., NeurIPS 2020,
 *       <a href="https://arxiv.org/abs/2006.16971">arXiv:2006.16971</a>).</li>
 *   <li>{@link Method#TENT}: test-batch statistics plus one online gradient step per batch on
 *       the batch-norm affine parameters only, minimising the mean prediction entropy (Wang et
 *       al., ICLR 2021, <a href="https://arxiv.org/abs/2006.10726">arXiv:2006.10726</a>). Updates
 *       carry over from batch to batch, which is why long or label-shifted streams can
 *       collapse it.</li>
 *   <li>{@link Method#MEMO}: per single test point, the entropy of the marginal prediction
 *       over augmented copies is minimised by one step on all parameters, the point is
 *       predicted, and the model is reset (Zhang, Levine and Finn, NeurIPS 2022,
 *       <a href="https://arxiv.org/abs/2110.09506">arXiv:2110.09506</a>); it needs no batch.</li>
 * </ul>
 *
 * <p>{@link #reset} restores the parameters and batch-norm buffers the adapter started with.
 */
public final class TestTimeAdaptation {

    /** The adaptation. */
    public enum Method {
        /** the model as trained */
        NONE,
        /** test-batch normalisation statistics */
        BN_STATS,
        /** entropy minimisation on BN affine parameters, online */
        TENT,
        /** marginal entropy over augmentations, per point */
        MEMO
    }

    private final Model model;
    private final Method method;
    private final List<Var> all;
    private final List<Var> buffers;
    private final List<Tensor> initial;
    private final List<Tensor> initialBuffers;
    private final List<Var> bnAffine = new ArrayList<>();
    private float learningRate = 1e-3f;
    private UnaryOperator<Tensor> augment;
    private Optimizer optimizer;

    private TestTimeAdaptation(Model model, Method method) {
        this.model = Objects.requireNonNull(model);
        this.method = Objects.requireNonNull(method);
        this.all = model.parameters();
        this.buffers = model.buffers();
        this.initial = values(all);
        this.initialBuffers = values(buffers);
        for (Layer l : model.layers()) {
            if (l instanceof BatchNorm) {
                bnAffine.addAll(l.parameters());
            }
        }
    }

    /**
     * Adapts a model.
     *
     * @param model the trained model, adapted in place
     * @param method the adaptation
     * @return the adapter
     */
    public static TestTimeAdaptation of(Model model, Method method) {
        return new TestTimeAdaptation(model, method);
    }

    /**
     * @param lr the step size of TENT (Adam) or MEMO (SGD)
     * @return this
     */
    public TestTimeAdaptation learningRate(float lr) {
        this.learningRate = lr;
        this.optimizer = null;
        return this;
    }

    /**
     * @param augmenter maps one input {@code [1, ...]} to a batch of augmented copies (MEMO)
     * @return this
     */
    public TestTimeAdaptation augmentations(UnaryOperator<Tensor> augmenter) {
        this.augment = augmenter;
        return this;
    }

    /** @return the batch-norm affine parameters TENT adapts */
    public List<Var> adaptedParameters() {
        return method == Method.TENT ? List.copyOf(bnAffine) : all;
    }

    /**
     * Predicts a test batch, adapting as the method prescribes.
     *
     * @param x the inputs
     * @return the logits
     */
    public Tensor predict(Tensor x) {
        return switch (method) {
            case NONE -> model.forward(Var.of(x), false).value();
            case BN_STATS -> {
                Tensor out = model.forward(Var.of(x), true).value();
                restoreBuffers();
                yield out;
            }
            case TENT -> {
                if (optimizer == null) {
                    optimizer = Optimizer.adam(learningRate);
                }
                Var logits = model.forward(Var.of(x), true);
                restoreBuffers();
                Var loss = entropy(logits).mean();
                for (Var p : all) {
                    p.zeroGrad();
                }
                List<Var> g = Var.gradients(loss, bnAffine, false);
                for (int i = 0; i < g.size(); i++) {
                    bnAffine.get(i).setGrad(g.get(i).value());
                }
                optimizer.step(bnAffine);
                yield logits.value();
            }
            case MEMO -> memo(x);
        };
    }

    private Tensor memo(Tensor x) {
        if (augment == null) {
            throw new IllegalStateException("MEMO needs augmentations");
        }
        int n = x.shape().dim(0);
        int d = Math.toIntExact(x.size() / n);
        int[] dims = x.shape().toArray();
        dims[0] = 1;
        float[] v = x.toArray();
        List<Tensor> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            Tensor one = Tensor.of(io.github.zzuegg.brainiac.tensor.Shape.of(dims),
                    java.util.Arrays.copyOfRange(v, i * d, (i + 1) * d));
            Var logits = model.forward(Var.of(augment.apply(one)), false);
            Var logP = logits.sub(logits.logsumexp(1, true));
            Var marginal = logP.exp().mean(0, true);
            Var loss = marginal.mul(marginal.add(1e-12f).log()).sum().neg();
            List<Var> g = Var.gradients(loss, all, false);
            for (int k = 0; k < all.size(); k++) {
                Var p = all.get(k);
                p.assign(p.value().sub(g.get(k).value().mul(learningRate)));
            }
            out.add(model.forward(Var.of(one), false).value());
            reset();
        }
        return Tensor.concat(0, out.toArray(Tensor[]::new));
    }

    /**
     * The per-row prediction entropy.
     *
     * @param logits {@code [n, classes]}
     * @return {@code [n]} entropies
     */
    public static Var entropy(Var logits) {
        Var logP = logits.sub(logits.logsumexp(1, true));
        return logP.exp().mul(logP).sum(1, false).neg();
    }

    /** Restores the parameters and buffers the adapter started with. */
    public void reset() {
        for (int i = 0; i < all.size(); i++) {
            all.get(i).assign(copy(initial.get(i)));
        }
        restoreBuffers();
        optimizer = null;
    }

    private void restoreBuffers() {
        for (int i = 0; i < buffers.size(); i++) {
            buffers.get(i).assign(copy(initialBuffers.get(i)));
        }
    }

    private static List<Tensor> values(List<Var> vars) {
        List<Tensor> out = new ArrayList<>();
        for (Var v : vars) {
            out.add(copy(v.value()));
        }
        return out;
    }

    /** A private copy: batch norm releases the running statistics it replaces. */
    private static Tensor copy(Tensor t) {
        return Tensor.of(t.shape(), t.toArray());
    }
}
