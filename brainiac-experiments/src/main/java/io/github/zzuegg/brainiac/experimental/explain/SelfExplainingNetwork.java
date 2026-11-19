package io.github.zzuegg.brainiac.experimental.explain;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A self-explaining neural network (Alvarez-Melis and Jaakkola, NeurIPS 2018,
 * <a href="https://arxiv.org/abs/1806.07538">arXiv:1806.07538</a>):
 * {@code f(x) = sum_k theta_k(x) h_k(x) + b}, a linear model in interpretable basis concepts
 * {@code h(x)} (the raw features, or the codes of an autoencoder trained alongside) whose
 * coefficients {@code theta(x)} come from a network. The robustness regulariser
 * {@code lambda ||grad_x f - theta(x)^T J_h(x)||^2}, computed by double backpropagation, makes
 * {@code theta} locally behave like the true coefficients, so the relevances are faithful local
 * explanations. As a {@link Layer} the forward pass returns {@code f(x)} ({@code [batch, 1]}).
 */
public final class SelfExplainingNetwork implements Layer {

    private final int inputs;
    private final int concepts;
    private final boolean autoencoder;
    private final Var t1;
    private final Var tb1;
    private final Var t2;
    private final Var tb2;
    private final Var t3;
    private final Var tb3;
    private final Var bias;
    private final Var enc;
    private final Var encB;
    private final Var dec;
    private final Var decB;

    /**
     * Creates the network.
     *
     * @param inputs input features
     * @param concepts the number of autoencoder concepts, or 0 for the identity (raw features)
     * @param hidden the width of the relevance network
     * @param seed the initial weights
     */
    public SelfExplainingNetwork(int inputs, int concepts, int hidden, long seed) {
        this.inputs = inputs;
        this.autoencoder = concepts > 0;
        this.concepts = autoencoder ? concepts : inputs;
        SplittableRandom r = new SplittableRandom(seed);
        t1 = Var.param(Tensor.normal(Shape.of(inputs, hidden), 0, (float) Math.sqrt(1.0 / inputs), r));
        tb1 = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        t2 = Var.param(Tensor.normal(Shape.of(hidden, hidden), 0, (float) Math.sqrt(1.0 / hidden), r));
        tb2 = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        t3 = Var.param(Tensor.normal(Shape.of(hidden, this.concepts), 0, (float) Math.sqrt(1.0 / hidden), r));
        tb3 = Var.param(Tensor.zeros(Shape.of(1, this.concepts)));
        bias = Var.param(Tensor.zeros(Shape.of(1, 1)));
        int k = Math.max(1, concepts);
        enc = Var.param(Tensor.normal(Shape.of(inputs, k), 0, (float) Math.sqrt(1.0 / inputs), r));
        encB = Var.param(Tensor.zeros(Shape.of(1, k)));
        dec = Var.param(Tensor.normal(Shape.of(k, inputs), 0, (float) Math.sqrt(1.0 / k), r));
        decB = Var.param(Tensor.zeros(Shape.of(1, inputs)));
    }

    /**
     * A spec with identity concepts.
     *
     * @param hidden the relevance network width
     * @return the spec
     */
    public static LayerSpec spec(int hidden) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(1);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new SelfExplainingNetwork(input.dim(input.rank() - 1), 0, hidden, random.nextLong());
            }
        };
    }

    /**
     * The relevances {@code theta(x)}.
     *
     * @param x inputs {@code [b, d]}
     * @return {@code [b, k]}
     */
    public Var relevances(Var x) {
        return x.matmul(t1).add(tb1).tanh().matmul(t2).add(tb2).tanh().matmul(t3).add(tb3);
    }

    /**
     * The concepts {@code h(x)}.
     *
     * @param x inputs
     * @return {@code [b, k]}
     */
    public Var concepts(Var x) {
        return autoencoder ? x.matmul(enc).add(encB).tanh() : x;
    }

    @Override
    public Var forward(Var input) {
        return relevances(input).mul(concepts(input)).sum(1, true).add(bias);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(t1, tb1, t2, tb2, t3, tb3, bias));
        if (autoencoder) {
            p.addAll(List.of(enc, encB, dec, decB));
        }
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return 1;
    }

    /**
     * The robustness penalty {@code mean ||grad_x f - theta^T J_h||^2} of a batch, differentiable
     * in the parameters (double backpropagation).
     *
     * @param x the batch
     * @return the scalar penalty
     */
    public Var robustness(Tensor x) {
        Var in = Var.param(x);
        Var theta = relevances(in);
        Var f = theta.mul(concepts(in)).sum(1, true).add(bias);
        Var gradF = Var.gradients(f.sum(), List.of(in), true).get(0);
        Var gradH = Var.gradients(Var.of(theta.value()).mul(concepts(in)).sum(), List.of(in), true).get(0);
        return gradF.sub(gradH).square().sum(1, false).mean();
    }

    /**
     * Trains on squared error (or logistic loss for 0/1 targets) plus {@code lambda} times the
     * robustness penalty, plus the reconstruction loss for autoencoder concepts.
     *
     * @param x inputs {@code [n, d]}
     * @param y targets {@code [n, 1]}
     * @param classification whether the targets are 0/1 labels (logistic loss on {@code f})
     * @param lambda the robustness weight
     * @param steps Adam steps
     * @param seed the batch seed
     */
    public void fit(Tensor x, Tensor y, boolean classification, float lambda, int steps, long seed) {
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        int n = x.shape().dim(0);
        int batch = Math.min(64, n);
        List<Var> params = parameters();
        for (int s = 0; s < steps; s++) {
            int from = r.nextInt(n - batch + 1);
            Tensor xb = x.slice(0, from, from + batch);
            Tensor yb = y.slice(0, from, from + batch);
            Var f = forward(Var.of(xb));
            Var loss = classification ? f.softplus().sub(f.mul(yb)).mean() : f.sub(yb).square().mean();
            if (lambda > 0) {
                loss = loss.add(robustness(xb).mul(lambda));
            }
            if (autoencoder) {
                Var h = concepts(Var.of(xb));
                loss = loss.add(h.matmul(dec).add(decB).sub(xb).square().mean().mul(0.5f));
            }
            loss.backward();
            opt.step(params);
        }
    }

    /** @return the number of concepts */
    public int conceptCount() {
        return concepts;
    }

    /** @return the number of inputs */
    public int inputs() {
        return inputs;
    }
}
