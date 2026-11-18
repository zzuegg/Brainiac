package io.github.zzuegg.brainiac.experimental.semisup;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * JEPAMatch (2026, <a href="https://arxiv.org/abs/2604.21046">arXiv:2604.21046</a>): FlexMatch
 * plus LeJEPA's sketched isotropic Gaussian regulariser (SIGReg) on the projected features of the
 * unlabelled batch. Pushing the embedding towards an isotropic Gaussian is argued to give clearer
 * class boundaries early, which limits the damage of noisy early pseudo-labels and the dominance
 * of majority classes. The paper's total loss is {@code L_sup + lambda_u L_unsup + lambda_rep
 * L_rep}, SIGReg applied to projection-head outputs of unlabelled views with an effective weight
 * of about 0.025 to 0.1; this block keeps the FlexMatch part ({@link FixMatch} with an
 * {@link AdaptiveThreshold} in {@code FLEX} mode) and adds {@link #sigreg}.
 *
 * <p>{@link #sigreg} is a local copy of LeJEPA's SIGReg (Balestriero and LeCun, Nov 2025): the
 * features are projected on random unit directions, and on each slice the Epps-Pulley statistic
 * compares the empirical characteristic function with the standard normal's,
 * {@code N * integral |phi_N(t) - exp(-t^2/2)|^2 exp(-t^2/2) dt}, by trapezoidal quadrature on
 * {@code [-tMax, tMax]}. It is to be replaced by {@code experimental.ssl.Sigreg} once that lands.
 */
public final class JepaMatch {

    private final FixMatch fixMatch;
    private final float lambdaRep;
    private final int slices;
    private final int points;

    /**
     * Creates the method.
     *
     * @param classes the number of classes
     * @param pool the size of the unlabelled pool
     * @param tau FlexMatch's upper threshold
     * @param lambdaU the unlabelled weight
     * @param lambdaRep the SIGReg weight
     * @param slices random directions per step
     * @param points quadrature points
     */
    public JepaMatch(int classes, int pool, float tau, float lambdaU, float lambdaRep, int slices, int points) {
        this.fixMatch = new FixMatch(new AdaptiveThreshold(AdaptiveThreshold.Mode.FLEX, classes, pool, tau, 0.99),
                lambdaU);
        this.lambdaRep = lambdaRep;
        this.slices = slices;
        this.points = points;
    }

    /** @return the FlexMatch part, for its loss and statistics */
    public FixMatch flexMatch() {
        return fixMatch;
    }

    /**
     * The JEPAMatch unlabelled loss.
     *
     * @param strongLogits logits on the strong views
     * @param weakLogits logits on the weak views
     * @param features the projected features of the unlabelled views, {@code [batch, dim]}
     * @param indices the rows' pool indices
     * @param truth hidden true classes for statistics, or null
     * @param seed seeds this step's random directions
     * @return FlexMatch's loss plus {@code lambdaRep * SIGReg(features)}
     */
    public Var loss(Var strongLogits, Tensor weakLogits, Var features, int[] indices, int[] truth, long seed) {
        Var l = fixMatch.loss(strongLogits, weakLogits, indices, truth);
        return lambdaRep == 0 ? l : l.add(sigreg(features, slices, points, 3f, seed).mul(lambdaRep));
    }

    /**
     * The sketched isotropic Gaussian regulariser.
     *
     * @param z {@code [batch, dim]} embeddings
     * @param slices random unit directions
     * @param points quadrature points on {@code [0, tMax]} (the integrand is even)
     * @param tMax the integration limit, such as 3
     * @param seed seeds the directions
     * @return the mean Epps-Pulley statistic over the slices, differentiable in {@code z}
     */
    public static Var sigreg(Var z, int slices, int points, float tMax, long seed) {
        int n = z.shape().dim(0);
        int d = z.shape().dim(1);
        SplittableRandom r = new SplittableRandom(seed);
        float[] a = new float[d * slices];
        for (int s = 0; s < slices; s++) {
            double norm = 0;
            for (int j = 0; j < d; j++) {
                a[j * slices + s] = (float) r.nextGaussian();
                norm += a[j * slices + s] * a[j * slices + s];
            }
            for (int j = 0; j < d; j++) {
                a[j * slices + s] /= (float) Math.sqrt(norm);
            }
        }
        Var proj = z.matmul(Var.of(Tensor.of(Shape.of(d, slices), a)));
        Var total = null;
        double h = tMax / (points - 1);
        for (int k = 0; k < points; k++) {
            float t = (float) (k * h);
            double gauss = Math.exp(-t * t / 2.0);
            double weight = (k == 0 || k == points - 1 ? 0.5 : 1) * h * gauss * 2;   // both half-lines
            Var tp = proj.mul(t);
            Var re = tp.cos().mean(0, false).sub((float) gauss);
            Var im = tp.sin().mean(0, false);
            Var term = re.square().add(im.square()).mul((float) weight);
            total = total == null ? term : total.add(term);
        }
        return total.mean().mul(n);
    }
}
