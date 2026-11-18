package io.github.zzuegg.brainiac.diffusion;

/** How a {@link StableDiffusion} run moves from noise to an image. */
public enum Sampler {
    /** Euler steps on the probability-flow ODE: deterministic, the same as DDIM with eta 0. */
    EULER(false, false),
    /** Euler steps that add fresh noise each step: more varied, never converges to one image as steps grow. */
    EULER_ANCESTRAL(false, true),
    /** DPM-Solver++ (2M): second order, good images in 20 to 30 steps. */
    DPM_PP_2M(false, false),
    /** DPM-Solver++ (2M) with Karras-spaced noise levels: the common default of image tools. */
    DPM_PP_2M_KARRAS(true, false),
    /** Euler with Karras-spaced noise levels. */
    EULER_KARRAS(true, false);

    private final boolean karras;
    private final boolean ancestral;

    Sampler(boolean karras, boolean ancestral) {
        this.karras = karras;
        this.ancestral = ancestral;
    }

    /**
     * Returns whether the noise levels are Karras-spaced.
     *
     * @return whether the sampler uses Karras sigmas
     */
    public boolean karras() {
        return karras;
    }

    /**
     * Returns whether the sampler adds noise each step, so that its result depends on more
     * than the starting noise.
     *
     * @return whether the sampler is stochastic
     */
    public boolean ancestral() {
        return ancestral;
    }
}
