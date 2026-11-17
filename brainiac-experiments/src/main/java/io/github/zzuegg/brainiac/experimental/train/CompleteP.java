package io.github.zzuegg.brainiac.experimental.train;

/**
 * Depth parametrisations of residual networks, CompleteP among them (Dey et al. 2025,
 * "Don't be lazy: CompleteP enables compute-efficient deep transformers",
 * <a href="https://arxiv.org/abs/2505.01618">arXiv:2505.01618</a>). A residual stack of
 * {@code L} blocks {@code h <- h + m * f(h)} is scaled with depth relative to a tuned base
 * depth {@code L0} by one exponent {@code alpha}:
 *
 * <ul>
 *   <li>branch multiplier {@code m = (L0 / L)^alpha};</li>
 *   <li>Adam learning rate of the hidden (block) weights {@code lr (L / L0)^(alpha - 1)};</li>
 *   <li>Adam epsilon of the hidden weights {@code eps (L0 / L)^alpha} and decoupled weight
 *       decay {@code wd (L / L0)^(1 - alpha)}, so their effect is depth-independent;</li>
 *   <li>initialisation and the embedding/output layers unchanged.</li>
 * </ul>
 * {@code alpha = 0} is the standard parametrisation (SP), {@code alpha = 1/2} depth-μP, and
 * {@code alpha = 1} CompleteP, the only choice under which every block stays non-lazy
 * (its features keep learning as {@code L} grows) and both the learning rate and the
 * initialisation transfer across depth. Use it together with {@link MaximalUpdate} for
 * width.
 */
public final class CompleteP {

    private final int baseDepth;
    private final int depth;
    private final double alpha;

    /**
     * @param baseDepth the depth the hyperparameters were tuned at
     * @param depth the depth of the model
     * @param alpha the depth exponent: 0 (SP), 0.5 (depth-μP) or 1 (CompleteP)
     */
    public CompleteP(int baseDepth, int depth, double alpha) {
        if (baseDepth < 1 || depth < 1 || alpha < 0 || alpha > 1) {
            throw new IllegalArgumentException("need positive depths and alpha in [0, 1]");
        }
        this.baseDepth = baseDepth;
        this.depth = depth;
        this.alpha = alpha;
    }

    /**
     * @param baseDepth the tuned depth
     * @param depth the model's depth
     * @return CompleteP, {@code alpha = 1}
     */
    public static CompleteP completeP(int baseDepth, int depth) {
        return new CompleteP(baseDepth, depth, 1);
    }

    /**
     * @param baseDepth the tuned depth
     * @param depth the model's depth
     * @return depth-μP, {@code alpha = 1/2}
     */
    public static CompleteP depthMuP(int baseDepth, int depth) {
        return new CompleteP(baseDepth, depth, 0.5);
    }

    /**
     * @param baseDepth the tuned depth
     * @param depth the model's depth
     * @return the standard parametrisation, {@code alpha = 0}
     */
    public static CompleteP standard(int baseDepth, int depth) {
        return new CompleteP(baseDepth, depth, 0);
    }

    /** @return the depth exponent */
    public double alpha() {
        return alpha;
    }

    /** @return the depth ratio {@code L / L0} */
    public double depthRatio() {
        return depth / (double) baseDepth;
    }

    /** @return the residual branch multiplier {@code (L0 / L)^alpha} */
    public float branchMultiplier() {
        return (float) Math.pow(depthRatio(), -alpha);
    }

    /**
     * @param baseRate the rate tuned at the base depth
     * @return the Adam learning rate of the hidden (residual block) weights
     */
    public float hiddenLearningRate(float baseRate) {
        return (float) (baseRate * Math.pow(depthRatio(), alpha - 1));
    }

    /**
     * @param baseEpsilon the Adam epsilon at the base depth
     * @return the epsilon for the hidden weights
     */
    public float hiddenEpsilon(float baseEpsilon) {
        return (float) (baseEpsilon * Math.pow(depthRatio(), -alpha));
    }

    /**
     * @param baseDecay the decoupled weight decay at the base depth
     * @return the weight decay for the hidden weights
     */
    public float hiddenWeightDecay(float baseDecay) {
        return (float) (baseDecay * Math.pow(depthRatio(), 1 - alpha));
    }

    @Override
    public String toString() {
        return "CompleteP(alpha=" + alpha + ", depth " + baseDepth + " -> " + depth + ")";
    }
}
