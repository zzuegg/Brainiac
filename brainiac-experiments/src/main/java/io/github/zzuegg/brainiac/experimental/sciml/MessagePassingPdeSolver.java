package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A message-passing neural PDE solver (Brandstetter, Worrall and Welling, ICLR 2022) for
 * periodic 1-D grids. The grid is a graph whose nodes are linked to their neighbours within
 * {@code radius}; an encoder-processor-decoder GNN maps the last {@code history} states (plus
 * PDE coefficients) to the next {@code bundle} states at once (temporal bundling):
 *
 * <pre>
 * h_i  = enc(u_i^{t-H+1..t}, coef)
 * m_ij = phi(h_i, h_j, u_i - u_j, (x_i - x_j)/r)           for |i - j| &lt;= r
 * h_i += psi(h_i, mean_j m_ij)                              (processor layers)
 * u_i^{t+k} = u_i^t + (k + 1) d_ik,  d_i = dec(h_i)          k = 0 .. K-1
 * </pre>
 *
 * <p>{@link #loss} implements the pushforward trick: the model first predicts a bundle from an
 * earlier window with no gradient, and is trained to predict the following bundle from its own
 * output, so it learns to correct the distribution shift of its rollouts.
 *
 * <p>As a {@link Layer}, the input is {@code [b, n, history + coefficients]} and the output
 * {@code [b, n, bundle]}.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2202.03376">arXiv:2202.03376</a>.
 */
public final class MessagePassingPdeSolver implements Layer {

    private final int history;
    private final int bundle;
    private final int coefficients;
    private final int width;
    private final int radius;
    private final Var[] encoder;
    private final List<Var[]> layers = new ArrayList<>();
    private final Var[] decoder;

    private MessagePassingPdeSolver(int history, int bundle, int coefficients, int width, int depth, int radius,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.history = history;
        this.bundle = bundle;
        this.coefficients = coefficients;
        this.width = width;
        this.radius = radius;
        this.encoder = mlp(r, history + coefficients, width, width);
        for (int i = 0; i < depth; i++) {
            Var[] msg = mlp(r, 2 * width + 2, width, width);
            Var[] upd = mlp(r, 2 * width, width, width);
            layers.add(new Var[] {msg[0], msg[1], msg[2], msg[3], upd[0], upd[1], upd[2], upd[3]});
        }
        this.decoder = mlp(r, width, width, bundle);
        decoder[2].assign(decoder[2].value().mul(0.1f));
    }

    private static Var[] mlp(SplittableRandom r, int in, int hidden, int out) {
        return new Var[] {
            Var.param(Tensor.normal(Shape.of(in, hidden), 0, (float) Math.sqrt(2.0 / in), r)),
            Var.param(Tensor.zeros(Shape.of(hidden))),
            Var.param(Tensor.normal(Shape.of(hidden, out), 0, (float) Math.sqrt(1.0 / hidden), r)),
            Var.param(Tensor.zeros(Shape.of(out)))};
    }

    private static Var apply(Var[] m, Var x) {
        Var h = x.matmul(m[0]).add(m[1]);
        h = h.mul(h.sigmoid());   // swish
        return h.matmul(m[2]).add(m[3]);
    }

    /**
     * Creates the solver.
     *
     * @param history input time steps {@code H}
     * @param bundle output time steps {@code K}
     * @param coefficients PDE coefficient channels per node
     * @param width hidden width
     * @param depth message-passing layers
     * @param radius neighbourhood radius in grid cells
     * @param seed the seed
     * @return the solver
     */
    public static MessagePassingPdeSolver of(int history, int bundle, int coefficients, int width, int depth,
            int radius, long seed) {
        return new MessagePassingPdeSolver(history, bundle, coefficients, width, depth, radius, seed);
    }

    /** @return input time steps */
    public int history() {
        return history;
    }

    /** @return output time steps */
    public int bundle() {
        return bundle;
    }

    /**
     * A periodic shift along the node axis: {@code out[:, i] = x[:, (i - s) mod n]}.
     *
     * @param x {@code [b, n, c]}
     * @param s the shift
     * @return the shifted tensor
     */
    static Var roll(Var x, int s) {
        int n = x.shape().dim(1);
        int k = ((s % n) + n) % n;
        if (k == 0) {
            return x;
        }
        return Var.concat(1, x.slice(1, n - k, n), x.slice(1, 0, n - k));
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        int b = s.dim(0);
        int n = s.dim(1);
        Var u = input.slice(2, history - 1, history);   // latest state [b, n, 1]
        Var h = apply(encoder, input);
        for (Var[] l : layers) {
            Var agg = null;
            for (int off = -radius; off <= radius; off++) {
                if (off == 0) {
                    continue;
                }
                Var hj = roll(h, off);
                Var du = u.sub(roll(u, off));
                Var rel = Var.of(Tensor.full(Shape.of(b, n, 1), off / (float) radius));
                Var m = apply(new Var[] {l[0], l[1], l[2], l[3]}, Var.concat(2, h, hj, du, rel));
                agg = agg == null ? m : agg.add(m);
            }
            agg = agg.div(2f * radius);
            h = h.add(apply(new Var[] {l[4], l[5], l[6], l[7]}, Var.concat(2, h, agg)));
        }
        float[] steps = new float[bundle];
        for (int k = 0; k < bundle; k++) {
            steps[k] = k + 1;
        }
        Var d = apply(decoder, h).mul(Var.of(Tensor.of(Shape.of(bundle), steps)));
        return u.add(d);
    }

    /**
     * Rolls the solver forward from an initial window.
     *
     * @param window the last {@code history} states {@code [b, n, history]}
     * @param conditions coefficient channels {@code [b, n, coefficients]}, or {@code null}
     * @param bundles how many bundles to predict
     * @return predicted states {@code [b, n, bundles * bundle]}
     */
    public Tensor rollout(Tensor window, Tensor conditions, int bundles) {
        return Var.noGrad(() -> {
            Var w = Var.of(window);
            List<Var> out = new ArrayList<>();
            for (int i = 0; i < bundles; i++) {
                Var in = conditions == null ? w : Var.concat(2, w, Var.of(conditions));
                Var next = forward(in);
                out.add(next);
                w = lastStates(Var.concat(2, w, next), history);
            }
            return Var.concat(2, out.toArray(new Var[0]));
        }).value();
    }

    private static Var lastStates(Var x, int h) {
        int t = x.shape().dim(2);
        return x.slice(2, t - h, t);
    }

    /**
     * The training loss on trajectory windows, with or without the pushforward trick.
     *
     * @param window states {@code [b, n, history + bundle]} without pushforward, or
     *     {@code [b, n, history + 2 * bundle]} with it
     * @param conditions coefficient channels {@code [b, n, coefficients]}, or {@code null}
     * @param pushforward whether to train on the model's own (stop-gradient) first prediction
     * @return the mean squared error on the last bundle
     */
    public Var loss(Tensor window, Tensor conditions, boolean pushforward) {
        int t = window.shape().dim(2);
        Tensor input;
        if (pushforward) {
            Tensor first = window.slice(2, 0, history);
            Tensor pred = Var.noGrad(() -> forward(cat(first, conditions))).value();
            Tensor joined = Tensor.concat(2, first, pred);
            input = joined.slice(2, joined.shape().dim(2) - history, joined.shape().dim(2));
        } else {
            input = window.slice(2, t - bundle - history, t - bundle);
        }
        Tensor target = window.slice(2, t - bundle, t);
        return forward(cat(input, conditions)).sub(Var.of(target)).square().mean();
    }

    private Var cat(Tensor states, Tensor conditions) {
        return conditions == null ? Var.of(states) : Var.of(Tensor.concat(2, states, conditions));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(encoder));
        layers.forEach(l -> p.addAll(List.of(l)));
        p.addAll(List.of(decoder));
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return bundle;
    }

    @Override
    public String toString() {
        return "MessagePassingPdeSolver(H=" + history + ", K=" + bundle + ", width=" + width + ", coef="
                + coefficients + ")";
    }
}
