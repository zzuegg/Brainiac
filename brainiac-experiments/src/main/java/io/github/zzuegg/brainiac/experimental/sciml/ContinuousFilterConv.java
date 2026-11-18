package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * SchNet's continuous-filter convolution (Schutt et al., 2017): a convolution over atoms at
 * arbitrary positions whose filter is a function of the pair distance. Distances are expanded
 * in Gaussian radial basis functions {@code exp(-gamma (d - mu_k)^2)}, a filter network maps
 * them to per-pair filters {@code W(d_ij)} (softened to zero at the cutoff), and
 *
 * <pre>
 * x_i' = sum_{j != i} x_j * W(d_ij)
 * </pre>
 *
 * <p>Distances are invariant to rotations and translations, so the output is too, and it is
 * smooth in the positions, so forces can be taken by autograd.
 *
 * <p>As a {@link MultiInputLayer} the inputs are features {@code [b, n, F]} and distances
 * {@code [b, n, n]}; the output is {@code [b, n, F]}.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/1706.08566">arXiv:1706.08566</a>.
 */
public final class ContinuousFilterConv implements MultiInputLayer {

    private final int features;
    private final int basis;
    private final double cutoff;
    private final float gamma;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;

    private ContinuousFilterConv(int features, int basis, double cutoff, SplittableRandom random) {
        this.features = features;
        this.basis = basis;
        this.cutoff = cutoff;
        double spacing = cutoff / (basis - 1);
        this.gamma = (float) (0.5 / (spacing * spacing));
        this.w1 = Var.param(Tensor.normal(Shape.of(basis, features), 0, (float) Math.sqrt(1.0 / basis), random));
        this.b1 = Var.param(Tensor.zeros(Shape.of(features)));
        this.w2 = Var.param(Tensor.normal(Shape.of(features, features), 0, (float) Math.sqrt(1.0 / features), random));
        this.b2 = Var.param(Tensor.zeros(Shape.of(features)));
    }

    /**
     * Creates the convolution.
     *
     * @param features the feature width {@code F}
     * @param basis radial basis functions, for example 20
     * @param cutoff the interaction cutoff distance
     * @param seed the seed
     * @return the layer
     */
    public static ContinuousFilterConv of(int features, int basis, double cutoff, long seed) {
        return new ContinuousFilterConv(features, basis, cutoff, new SplittableRandom(seed));
    }

    /**
     * The shifted softplus {@code ln(1 + e^x) - ln 2} SchNet uses.
     *
     * @param x values
     * @return {@code ssp(x)}
     */
    public static Var shiftedSoftplus(Var x) {
        return x.softplus().sub((float) Math.log(2));
    }

    /**
     * The filters {@code W(d_ij)}, zero on the diagonal and beyond the cutoff.
     *
     * @param distances {@code [b, n, n]}
     * @return {@code [b, n, n, F]}
     */
    public Var filters(Var distances) {
        Shape s = distances.shape();
        int b = s.dim(0);
        int n = s.dim(1);
        float[] mu = new float[basis];
        for (int k = 0; k < basis; k++) {
            mu[k] = (float) (k * cutoff / (basis - 1));
        }
        Var d = distances.reshape(b, n, n, 1);
        Var rbf = d.sub(Var.of(Tensor.of(Shape.of(basis), mu))).square().mul(-gamma).exp();   // [b, n, n, K]
        Var w = shiftedSoftplus(rbf.matmul(w1).add(b1)).matmul(w2).add(b2);
        // cosine cutoff, and no self-interaction
        Tensor dv = distances.value();
        float[] dd = dv.toArray();
        float[] keep = new float[dd.length];
        for (int i = 0; i < dd.length; i++) {
            int row = (i / n) % n;
            int col = i % n;
            keep[i] = row == col || dd[i] >= cutoff ? 0f : 1f;
        }
        Var fc = d.mul((float) (Math.PI / cutoff)).cos().add(1f).mul(0.5f)
                .mul(Var.of(Tensor.of(Shape.of(b, n, n, 1), keep)));
        return w.mul(fc);
    }

    /**
     * The convolution.
     *
     * @param x features {@code [b, n, F]}
     * @param distances {@code [b, n, n]}
     * @return {@code [b, n, F]}
     */
    public Var forward(Var x, Var distances) {
        int b = x.shape().dim(0);
        int n = x.shape().dim(1);
        Var xj = x.reshape(b, 1, n, features).expand(Shape.of(b, n, n, features));
        return xj.mul(filters(distances)).sum(2, false);
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        return forward(inputs.get(0), inputs.get(1));
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return inputs.get(0);
    }

    @Override
    public List<Var> parameters() {
        return List.of(w1, b1, w2, b2);
    }

    @Override
    public String toString() {
        return "ContinuousFilterConv(F=" + features + ", K=" + basis + ", cutoff=" + cutoff + ")";
    }
}
