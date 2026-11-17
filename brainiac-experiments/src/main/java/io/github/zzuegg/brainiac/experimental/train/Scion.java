package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.Map;

/**
 * Scion (Pethick et al. 2025, "Training Deep Learning Models with Norm-Constrained LMOs",
 * <a href="https://arxiv.org/abs/2502.07529">arXiv:2502.07529</a>), built on modular duality
 * (Bernstein and Newhouse 2024, <a href="https://arxiv.org/abs/2410.21265">arXiv:2410.21265</a>):
 * each layer's update is the <em>linear minimisation oracle</em> (LMO) of the averaged
 * gradient {@code d} over the unit ball of a norm chosen for that layer's role, scaled by a
 * radius:
 *
 * <ul>
 *   <li>{@link Norm#SPECTRAL} for hidden matrices ({@code RMS -> RMS} operator norm):
 *       {@code lmo(d) = -sqrt(fanOut / fanIn) U V^T}, the orthogonalised gradient. Muon is
 *       this case alone.</li>
 *   <li>{@link Norm#ROW} for input layers and embeddings fed by one-hot or ℓ1-like inputs
 *       ({@code ℓ1 -> RMS}): each input's outgoing weights normalised to unit RMS.</li>
 *   <li>{@link Norm#SIGN} for output layers ({@code RMS -> ℓ∞}): {@code -sign(d) / fanIn}.</li>
 *   <li>{@link Norm#RMS} for biases and gains: {@code -d / RMS(d)}.</li>
 * </ul>
 * Because each update has a fixed, width-correct size in its layer's natural norm, the
 * learning rate transfers across widths without μP's per-layer multipliers.
 *
 * <p>Two update rules: unconstrained ({@code w -= lr * radius * lmo}, the default, like
 * Muon) and constrained ({@code w = (1 - lr) w + lr * radius * lmo}, conditional gradient,
 * which keeps every weight inside its norm ball). The averaged gradient is
 * {@code d = (1 - alpha) d + alpha g}.
 */
public final class Scion extends SlotOptimizer {

    /** The norm whose LMO a parameter's update uses. */
    public enum Norm {
        /** Spectral (RMS to RMS operator) norm, for hidden matrices. */
        SPECTRAL,
        /** Per-input-row RMS normalisation, for input layers and embeddings. */
        ROW,
        /** Sign scaled by {@code 1 / fanIn}, for output layers. */
        SIGN,
        /** Whole-tensor RMS normalisation, for biases and gains. */
        RMS
    }

    private static final float A = 3.4445f;
    private static final float B = -4.775f;
    private static final float C = 2.0315f;

    private final float alpha;
    private final boolean constrained;
    private final Map<Var, Norm> norms = new IdentityHashMap<>();
    private final Map<Var, Float> radii = new IdentityHashMap<>();

    /**
     * @param learningRate the step size
     * @param alpha the weight of the new gradient in the average, 0.1
     * @param constrained whether to use the constrained (Frank-Wolfe) update
     */
    public Scion(float learningRate, float alpha, boolean constrained) {
        super(learningRate, "d");
        if (!(alpha > 0) || alpha > 1) {
            throw new IllegalArgumentException("alpha must be in (0, 1]");
        }
        this.alpha = alpha;
        this.constrained = constrained;
    }

    /**
     * Sets the norm and radius of one parameter. Parameters without one get
     * {@link Norm#SPECTRAL} if they are matrices with both sides above 1, else
     * {@link Norm#RMS}, with radius 1.
     *
     * @param parameter the parameter
     * @param norm its norm
     * @param radius the scale of its updates
     * @return this optimizer
     */
    public Scion norm(Var parameter, Norm norm, float radius) {
        norms.put(parameter, norm);
        radii.put(parameter, radius);
        return this;
    }

    /**
     * Newton-Schulz orthogonalisation, as in Muon: the matrix {@code U S' V^T} nearest
     * {@code U S V^T} with {@code S'} close to the identity.
     *
     * @param g a matrix
     * @param steps iterations, 5
     * @return the orthogonalised matrix
     */
    public static Tensor orthogonalize(Tensor g, int steps) {
        boolean tall = g.shape().dim(0) > g.shape().dim(1);
        Tensor x = tall ? g.transpose() : g;
        Tensor norm = x.square().sum().sqrt();
        x = x.div(norm.maximum(Tensor.full(Shape.of(), 1e-7f)));
        for (int i = 0; i < steps; i++) {
            Tensor a = x.matmul(x.transpose());
            Tensor b = a.mul(B).add(a.matmul(a).mul(C));
            x = x.mul(A).add(b.matmul(x));
        }
        return tall ? x.transpose() : x;
    }

    /**
     * The LMO direction (the negative is the update) of {@code d} for a norm, unit radius.
     *
     * @param d the averaged gradient
     * @param norm the norm
     * @return the direction, same shape
     */
    public static Tensor direction(Tensor d, Norm norm) {
        Tensor eps = Tensor.full(Shape.of(), 1e-12f);
        return switch (norm) {
            case SPECTRAL -> {
                int fanIn = d.shape().dim(0);
                int fanOut = d.shape().dim(1);
                yield orthogonalize(d, 5).mul((float) Math.sqrt((double) fanOut / fanIn));
            }
            case ROW -> d.div(d.square().mean(1, true).sqrt().maximum(eps));
            case SIGN -> d.sign().div(d.shape().dim(0));
            case RMS -> d.div(d.square().mean().sqrt().maximum(eps));
        };
    }

    private Norm normOf(Var p, Tensor g) {
        Norm n = norms.get(p);
        if (n != null) {
            return n;
        }
        return g.rank() == 2 && g.shape().dim(0) > 1 && g.shape().dim(1) > 1 ? Norm.SPECTRAL : Norm.RMS;
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor d = s[0] == null ? g.add(0f) : s[0].mul(1 - alpha).add(g.mul(alpha));
        float radius = radii.getOrDefault(p, 1f);
        float lr = learningRate();
        Tensor dir = direction(d, normOf(p, g)).mul(radius);
        p.assign(constrained ? p.value().mul(1 - lr).sub(dir.mul(lr)) : p.value().sub(dir.mul(lr)));
        return new Tensor[] {d};
    }

    @Override
    public String toString() {
        return "Scion(lr=" + learningRate() + (constrained ? ", constrained" : "") + ")";
    }
}
