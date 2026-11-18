package io.github.zzuegg.brainiac.experimental.circuits;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Squared probabilistic circuits with real or complex weights (Loconte et al., ICLR 2024,
 * <a href="https://arxiv.org/abs/2310.00724">arXiv:2310.00724</a>), and with {@code U > 1}
 * components the Inception circuits of Wang and Van den Broeck (AAAI 2025,
 * <a href="https://arxiv.org/abs/2408.00876">arXiv:2408.00876</a>), see
 * {@link InceptionCircuit}.
 *
 * <p>A circuit {@code f} with unconstrained (possibly negative or complex) weights on one
 * {@link RegionTree} defines {@code p(x) = |f(x)|^2 / Z}. Because the circuit is
 * structured-decomposable, {@code Z = sum_x |f(x)|^2} is computed exactly by propagating Gram
 * matrices {@code G_R[a][b] = sum over x_R of f_a(x_R) conj(f_b(x_R))} up the tree: a leaf's Gram
 * is {@code E^T conj(E)}, a product layer's is the Kronecker product of its children's, and a
 * sum layer {@code W} maps it to {@code W^T G conj(W)}, in {@code O(K^4)} per region. Negative
 * weights let the square cancel probability mass, which monotone circuits cannot do.
 *
 * <p>With {@code U} components sharing the leaves,
 * {@code p(x) proportional to sum_u a_u |f_u(x)|^2} with {@code a = softmax}: the outer
 * latent is summed outside the square and the units inside it, so monotone mixtures and squared
 * circuits are both special cases. Training is Adam on the exact negative log-likelihood through
 * autograd. Every node's amplitudes and Gram are divided by a detached scale (and its square),
 * which cancels exactly between {@code |f|^2} and {@code Z} and keeps float32 in range.
 */
public final class SquaredCircuit {

    private final int[] categories;
    private final int units;
    private final boolean complex;
    private final RegionTree tree;
    private final Var[][] leaves;          // [node][re, im] : [C][K]
    private final Var[][][] weights;       // [u][node][re, im] : [K*K][Kout]
    private final Var mixture;             // [U] logits
    private final List<Var> parameters = new ArrayList<>();

    /**
     * Creates a randomly initialised circuit.
     *
     * @param categories categories per variable (all categorical)
     * @param units {@code K}, units per region
     * @param complex complex instead of real parameters
     * @param components {@code U}, squared components summed outside the square (1 for a plain
     *     squared circuit)
     * @param seed the seed
     */
    public SquaredCircuit(int[] categories, int units, boolean complex, int components, long seed) {
        this.categories = categories.clone();
        this.units = units;
        this.complex = complex;
        this.tree = RegionTree.random(categories.length, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        int parts = complex ? 2 : 1;
        leaves = new Var[tree.size()][];
        weights = new Var[components][tree.size()][];
        for (int node = 0; node < tree.size(); node++) {
            if (tree.isLeaf(node)) {
                int c = categories[tree.variable(node)];
                leaves[node] = new Var[parts];
                for (int p = 0; p < parts; p++) {
                    leaves[node][p] = param(c, units, 1.0 / Math.sqrt(c * parts), p == 0 ? 0.5 : 0, r);
                }
            } else {
                int out = node == tree.root() ? 1 : units;
                for (int u = 0; u < components; u++) {
                    weights[u][node] = new Var[parts];
                    for (int p = 0; p < parts; p++) {
                        weights[u][node][p] = param(units * units, out, 1.0 / Math.sqrt(units * units * parts), 0,
                                r);
                    }
                }
            }
        }
        mixture = Var.param(Tensor.zeros(Shape.of(components)));
        if (components > 1) {
            parameters.add(mixture);
        }
    }

    private Var param(int rows, int cols, double scale, double shift, SplittableRandom r) {
        float[] v = new float[rows * cols];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (shift + scale * r.nextGaussian());
        }
        Var p = Var.param(Tensor.of(Shape.of(rows, cols), v));
        parameters.add(p);
        return p;
    }

    /** @return the number of real parameters */
    public int parameterCount() {
        int n = 0;
        for (Var p : parameters) {
            n += (int) p.shape().size();
        }
        return n;
    }

    /** A complex value as two real parts ({@code im == null} for real circuits). */
    private record Cx(Var re, Var im) {

        Cx mul(Cx o) {
            if (im == null) {
                return new Cx(re.mul(o.re), null);
            }
            return new Cx(re.mul(o.re).sub(im.mul(o.im)), re.mul(o.im).add(im.mul(o.re)));
        }

        Cx matmul(Cx o) {
            if (im == null) {
                return new Cx(re.matmul(o.re), null);
            }
            return new Cx(re.matmul(o.re).sub(im.matmul(o.im)), re.matmul(o.im).add(im.matmul(o.re)));
        }

        Cx transposed() {
            return new Cx(re.transpose(), im == null ? null : im.transpose());
        }

        Cx conj() {
            return new Cx(re, im == null ? null : im.neg());
        }

        Cx reshape(int... dims) {
            return new Cx(re.reshape(dims), im == null ? null : im.reshape(dims));
        }

        Cx div(float s) {
            return new Cx(re.div(s), im == null ? null : im.div(s));
        }

        Var abs2() {
            return im == null ? re.square() : re.square().add(im.square());
        }
    }

    private Cx leaf(int node) {
        Var[] e = leaves[node];
        return new Cx(e[0], complex ? e[1] : null);
    }

    private Cx weight(int u, int node) {
        Var[] w = weights[u][node];
        return new Cx(w[0], complex ? w[1] : null);
    }

    private static float trace(Cx g, int n) {
        float[] v = g.re().value().toArray();
        double s = 0;
        for (int i = 0; i < n; i++) {
            s += v[i * n + i];
        }
        return (float) Math.max(Math.sqrt(Math.max(s, 1e-30) / n), 1e-15);
    }

    /**
     * The per-sample log-probabilities as a differentiable {@code [B]} value.
     *
     * @param x samples, category indices {@code [B][d]}
     * @return {@code log p(x)}
     */
    public Var logProbability(int[][] x) {
        int b = x.length;
        int k = units;
        int comps = weights.length;
        Cx[] amp = new Cx[tree.size()];
        Cx[] gram = new Cx[tree.size()];
        Var logZ = null;
        Var logF = null;
        Var logA = mixture.logSoftmax(0);
        // leaves are shared by every component
        Cx[] leafAmp = new Cx[tree.size()];
        Cx[] leafGram = new Cx[tree.size()];
        for (int node = 0; node < tree.size(); node++) {
            if (!tree.isLeaf(node)) {
                continue;
            }
            int v = tree.variable(node);
            int c = categories[v];
            float[] onehot = new float[b * c];
            for (int i = 0; i < b; i++) {
                onehot[i * c + x[i][v]] = 1;
            }
            Cx e = leaf(node);
            Cx a = new Cx(Var.of(Tensor.of(Shape.of(b, c), onehot)).matmul(e.re()),
                    complex ? Var.of(Tensor.of(Shape.of(b, c), onehot)).matmul(e.im()) : null);
            Cx g = e.transposed().matmul(e.conj());
            float s = trace(g, k);
            leafAmp[node] = a.div(s);
            leafGram[node] = g.div(s * s);
        }
        Var[] perComp = new Var[comps];
        Var[] zComp = new Var[comps];
        for (int u = 0; u < comps; u++) {
            double logScale = 0;   // sum of 2 log s over inner nodes: restores the true scale across components
            for (int node = 0; node < tree.size(); node++) {
                if (tree.isLeaf(node)) {
                    amp[node] = leafAmp[node];
                    gram[node] = leafGram[node];
                    continue;
                }
                Cx la = amp[tree.left(node)];
                Cx ra = amp[tree.right(node)];
                Cx prod = la.reshape(b, k, 1).mul(ra.reshape(b, 1, k)).reshape(b, k * k);
                Cx lg = gram[tree.left(node)];
                Cx rg = gram[tree.right(node)];
                Cx kron = lg.reshape(k, 1, k, 1).mul(rg.reshape(1, k, 1, k)).reshape(k * k, k * k);
                Cx w = weight(u, node);
                Cx out = prod.matmul(w);
                Cx g = w.transposed().matmul(kron).matmul(w.conj());
                int kout = node == tree.root() ? 1 : k;
                float s = trace(g, kout);
                amp[node] = out.div(s);
                gram[node] = g.div(s * s);
                logScale += 2 * Math.log(s);
            }
            Cx root = amp[tree.root()];
            perComp[u] = root.abs2().reshape(b).add(1e-30f).log().add(logA.slice(0, u, u + 1)).add((float) logScale);
            zComp[u] = gram[tree.root()].re().reshape(1).log().add(logA.slice(0, u, u + 1)).add((float) logScale);
        }
        logF = comps == 1 ? perComp[0] : Var.concat(1, reshapeAll(perComp, b)).logsumexp(1, false);
        logZ = comps == 1 ? zComp[0] : Var.concat(0, zComp).logsumexp(0, false).reshape(1);
        return logF.sub(logZ);
    }

    private static Var[] reshapeAll(Var[] v, int b) {
        Var[] out = new Var[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = v[i].reshape(b, 1);
        }
        return out;
    }

    /**
     * Trains by Adam on the negative log-likelihood.
     *
     * @param x training samples
     * @param steps Adam steps
     * @param batch samples per step
     * @param lr the learning rate
     * @param seed the batch seed
     * @return this
     */
    public SquaredCircuit fit(int[][] x, int steps, int batch, float lr, long seed) {
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        int bs = Math.min(batch, x.length);
        for (int s = 0; s < steps; s++) {
            int[][] bx = new int[bs][];
            for (int i = 0; i < bs; i++) {
                bx[i] = x[r.nextInt(x.length)];
            }
            logProbability(bx).mean().neg().backward();
            opt.step(parameters);
        }
        return this;
    }

    /**
     * The mean log-likelihood.
     *
     * @param x samples
     * @return the average {@code log p(x)}
     */
    public double meanLogLikelihood(int[][] x) {
        float[] v = Var.noGrad(() -> logProbability(x)).value().toArray();
        double s = 0;
        for (float f : v) {
            s += f;
        }
        return s / v.length;
    }
}
