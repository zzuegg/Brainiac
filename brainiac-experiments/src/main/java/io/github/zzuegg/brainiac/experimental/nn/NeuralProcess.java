package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Neural processes: amortized function regression. An encoder turns every context pair
 * {@code (x_i, y_i)} into a representation {@code r_i}; a decoder takes a summary of them plus
 * a target input {@code x*} and outputs a Gaussian {@code N(mu(x*), sigma(x*)^2)}. Trained on
 * many functions by maximizing the targets' log-likelihood, it predicts a new function from a
 * handful of points in one forward pass, with error bars that shrink near the context.
 *
 * <ul>
 *   <li>{@link Kind#CONDITIONAL} is the <b>CNP</b>: the summary is the mean of {@code r_i},
 *       the same for every target, which tends to underfit the context. Garnelo et al.,
 *       "Conditional Neural Processes" (2018), <a href="https://arxiv.org/abs/1807.01613">arXiv:1807.01613</a>.</li>
 *   <li>{@link Kind#ATTENTIVE} is the deterministic path of the <b>ANP</b>: each target reads
 *       its own summary by multi-head cross-attention (queries from {@code x*}, keys from
 *       {@code x_i}, values {@code r_i}), so nearby context points dominate. Kim et al.,
 *       "Attentive Neural Processes" (2019), <a href="https://arxiv.org/abs/1901.05761">arXiv:1901.05761</a>.
 *       The ANP's latent path (a global stochastic code trained by an ELBO) is left out;
 *       the attention is what fixes the underfitting.</li>
 * </ul>
 */
public final class NeuralProcess {

    /** Which neural process. */
    public enum Kind {
        /** Mean-pooled context (CNP). */
        CONDITIONAL,
        /** Cross-attention over the context (deterministic ANP). */
        ATTENTIVE
    }

    private static final int HEADS = 4;

    private final Kind kind;
    private final int yDim;
    private final int hidden;
    private final List<Var[]> encoder;
    private final List<Var[]> decoder;
    private final List<Var[]> queryNet;

    /**
     * Creates the process.
     *
     * @param xDim input size
     * @param yDim output size
     * @param hidden the representation width, divisible by 4
     * @param kind CNP or ANP
     * @param random the initializer
     */
    public NeuralProcess(int xDim, int yDim, int hidden, Kind kind, RandomGenerator random) {
        if (hidden % HEADS != 0) {
            throw new IllegalArgumentException("hidden must be divisible by " + HEADS + ", got " + hidden);
        }
        this.kind = kind;
        this.yDim = yDim;
        this.hidden = hidden;
        this.encoder = mlp(new int[] {xDim + yDim, hidden, hidden, hidden}, random);
        int summary = kind == Kind.ATTENTIVE ? 2 * hidden : hidden;
        this.decoder = mlp(new int[] {summary + xDim, hidden, hidden, 2 * yDim}, random);
        this.queryNet = kind == Kind.ATTENTIVE ? mlp(new int[] {xDim, hidden, hidden}, random) : List.of();
    }

    private static List<Var[]> mlp(int[] sizes, RandomGenerator random) {
        List<Var[]> layers = new ArrayList<>();
        for (int i = 0; i + 1 < sizes.length; i++) {
            layers.add(new Var[] {
                Var.param(Tensor.normal(Shape.of(sizes[i], sizes[i + 1]), 0f, (float) Math.sqrt(2.0 / sizes[i]),
                        random)),
                Var.param(Tensor.zeros(Shape.of(sizes[i + 1])))});
        }
        return layers;
    }

    private static Var run(List<Var[]> layers, Var x) {
        for (int i = 0; i < layers.size(); i++) {
            x = x.matmul(layers.get(i)[0]).add(layers.get(i)[1]);
            if (i + 1 < layers.size()) {
                x = x.relu();
            }
        }
        return x;
    }

    /** @return the kind */
    public Kind kind() {
        return kind;
    }

    /**
     * The predictive Gaussians at the targets, for one task ({@code [n, xDim]} inputs) or a
     * batch of tasks with equal sizes ({@code [tasks, n, xDim]}).
     *
     * @param contextX {@code [n, xDim]} or {@code [tasks, n, xDim]}
     * @param contextY {@code [n, yDim]} or {@code [tasks, n, yDim]}
     * @param targetX {@code [m, xDim]} or {@code [tasks, m, xDim]}
     * @return {@code {mean, sigma}}, each {@code [m, yDim]} or {@code [tasks, m, yDim]}
     */
    public Var[] predict(Tensor contextX, Tensor contextY, Tensor targetX) {
        boolean single = contextX.rank() == 2;
        if (single) {
            contextX = contextX.reshape(1, contextX.shape().dim(0), contextX.shape().dim(1));
            contextY = contextY.reshape(1, contextY.shape().dim(0), contextY.shape().dim(1));
            targetX = targetX.reshape(1, targetX.shape().dim(0), targetX.shape().dim(1));
        }
        int tasks = contextX.shape().dim(0);
        int n = contextX.shape().dim(1);
        int m = targetX.shape().dim(1);
        int xDim = contextX.shape().dim(2);
        Tensor pairs = Tensor.concat(2, contextX, contextY);
        Var r = run(encoder, Var.of(pairs.reshape(tasks * n, pairs.shape().dim(2)))).reshape(tasks, n, hidden);
        Var summary = r.mean(1, true).expand(Shape.of(tasks, m, hidden));
        if (kind == Kind.ATTENTIVE) {
            int dk = hidden / HEADS;
            Var q = run(queryNet, Var.of(targetX.reshape(tasks * m, xDim))).reshape(tasks, m, HEADS, dk)
                    .permute(0, 2, 1, 3);
            Var k = run(queryNet, Var.of(contextX.reshape(tasks * n, xDim))).reshape(tasks, n, HEADS, dk)
                    .permute(0, 2, 1, 3);
            Var v = r.reshape(tasks, n, HEADS, dk).permute(0, 2, 1, 3);
            Var read = Functional.attention(q, k, v, false, (float) (1 / Math.sqrt(dk)));
            summary = Var.concat(2, summary, read.permute(0, 2, 1, 3).reshape(tasks, m, hidden));
        }
        Var in = Var.concat(2, summary, Var.of(targetX));
        Var out = run(decoder, in.reshape(tasks * m, in.shape().dim(2)));
        Var mean = out.slice(1, 0, yDim);
        Var sigma = out.slice(1, yDim, 2 * yDim).softplus().mul(0.9f).add(0.05f);
        return single ? new Var[] {mean, sigma}
            : new Var[] {mean.reshape(tasks, m, yDim), sigma.reshape(tasks, m, yDim)};
    }

    /**
     * The mean Gaussian negative log-likelihood of the targets, the training loss.
     *
     * @param contextX {@code [n, xDim]}, or batched as in {@link #predict}
     * @param contextY {@code [n, yDim]}, or batched
     * @param targetX {@code [m, xDim]}, or batched
     * @param targetY {@code [m, yDim]}, or batched
     * @return a scalar
     */
    public Var nll(Tensor contextX, Tensor contextY, Tensor targetX, Tensor targetY) {
        Var[] p = predict(contextX, contextY, targetX);
        return gaussianNll(p[0], p[1], targetY);
    }

    /**
     * The mean Gaussian negative log-likelihood.
     *
     * @param mean predicted means
     * @param sigma predicted standard deviations
     * @param y targets, same shape
     * @return a scalar
     */
    public static Var gaussianNll(Var mean, Var sigma, Tensor y) {
        Var z = Var.of(y).sub(mean).div(sigma);
        return sigma.log().add(z.square().mul(0.5f)).add(0.5f * (float) Math.log(2 * Math.PI)).mean();
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        List<Var> out = new ArrayList<>();
        for (List<Var[]> net : List.of(encoder, decoder, queryNet)) {
            for (Var[] l : net) {
                out.add(l[0]);
                out.add(l[1]);
            }
        }
        return out;
    }
}
