package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * The MatFormer feed-forward block: a SwiGLU FFN with {@code hidden} units trained so that
 * its <b>first</b> {@code m} units, for every granularity {@code m} (by default
 * {@code hidden/8, hidden/4, hidden/2, hidden}), form a working sub-network on their own.
 * The sub-networks are nested (the small one is a prefix of the large one), so one trained
 * model holds a family of models of different sizes, and any of them can be extracted
 * without retraining; Mix'n'Match picks a different width per layer at inference.
 *
 * <p>Training: every training forward pass uses one granularity drawn from a fixed-seed
 * sequence; layers created with the same {@code seed} draw the same sequence, so a whole
 * stack trains one consistent sub-model per step (as in the paper, which samples one
 * granularity per step). Outside training the layer uses {@link #activeWidth(int)} (the full
 * width by default).
 *
 * <p>Papers: Devvrit et al., "MatFormer: Nested Transformer for Elastic Inference",
 * <a href="https://arxiv.org/abs/2310.07707">arXiv:2310.07707</a>; Gemma 3n
 * (<a href="https://developers.googleblog.com/en/introducing-gemma-3n-developer-guide/">developer guide</a>).
 */
public final class NestedFeedForward implements Layer {

    private final int features;
    private final int hidden;
    private final int[] granularities;
    private final Var gate;
    private final Var up;
    private final Var down;
    private final SplittableRandom sampler;
    private int activeWidth;
    private int lastWidth;

    /**
     * Creates the block.
     *
     * @param features the input and output width
     * @param hidden the full hidden width
     * @param granularities the trained hidden widths, ascending, the last equal to {@code hidden}
     * @param seed the seed of the granularity sequence (equal in all layers of a stack)
     * @param random the random source for weights
     */
    public NestedFeedForward(int features, int hidden, int[] granularities, long seed, RandomGenerator random) {
        if (granularities.length == 0 || granularities[granularities.length - 1] != hidden) {
            throw new IllegalArgumentException("the largest granularity must be the hidden width");
        }
        for (int g : granularities) {
            if (g < 1 || g > hidden) {
                throw new IllegalArgumentException("granularity " + g + " outside [1, " + hidden + "]");
            }
        }
        this.features = features;
        this.hidden = hidden;
        this.granularities = granularities.clone();
        this.gate = MixerOps.weight(random, features, hidden);
        this.up = MixerOps.weight(random, features, hidden);
        this.down = MixerOps.weight(random, hidden, features);
        this.sampler = new SplittableRandom(seed);
        this.activeWidth = hidden;
        this.lastWidth = hidden;
    }

    /**
     * The usual granularities {@code hidden/8, hidden/4, hidden/2, hidden}.
     *
     * @param hidden the full hidden width, a multiple of 8
     * @return the granularities
     */
    public static int[] halvings(int hidden) {
        return new int[] {hidden / 8, hidden / 4, hidden / 2, hidden};
    }

    /**
     * Describes the block for {@code Model.sequential().add(...)} (or as a
     * {@code ResidualMixerBlock} FFN), with the halving granularities and seed 0.
     *
     * @param hiddenFactor the full hidden width as a multiple of the input width
     * @return the spec
     */
    public static LayerSpec spec(int hiddenFactor) {
        return MixerOps.sameShape((in, r) -> {
            int d = in.dim(-1);
            return new NestedFeedForward(d, hiddenFactor * d, halvings(hiddenFactor * d), 0, r);
        });
    }

    /**
     * Describes the block with explicit widths, for example {@code new int[] {hidden}} for a
     * plain (non-nested) FFN of the same shape.
     *
     * @param hidden the full hidden width
     * @param granularities the trained widths, ascending, the last equal to {@code hidden}
     * @return the spec
     */
    public static LayerSpec spec(int hidden, int[] granularities) {
        int[] g = granularities.clone();
        return MixerOps.sameShape((in, r) -> new NestedFeedForward(in.dim(-1), hidden, g, 0, r));
    }

    /**
     * Sets the hidden width used outside training (Mix'n'Match per layer).
     *
     * @param width the number of leading hidden units, between 1 and {@code hidden}
     * @return this block
     */
    public NestedFeedForward activeWidth(int width) {
        if (width < 1 || width > hidden) {
            throw new IllegalArgumentException("width " + width + " outside [1, " + hidden + "]");
        }
        this.activeWidth = width;
        return this;
    }

    /** @return the hidden width used outside training */
    public int activeWidth() {
        return activeWidth;
    }

    /** @return the hidden width used in the last forward pass */
    public int lastWidth() {
        return lastWidth;
    }

    /** @return the trained granularities */
    public int[] granularities() {
        return granularities.clone();
    }

    /**
     * The sub-network of the leading {@code width} hidden units.
     *
     * @param input {@code [..., features]}
     * @param width the hidden units to use
     * @return {@code [..., features]}
     */
    public Var forward(Var input, int width) {
        lastWidth = width;
        Var g = Activation.SILU.forward(input.matmul(gate.slice(1, 0, width)));
        return g.mul(input.matmul(up.slice(1, 0, width))).matmul(down.slice(0, 0, width));
    }

    @Override
    public Var forward(Var input) {
        return forward(input, activeWidth);
    }

    @Override
    public Var forward(Var input, boolean training) {
        return training ? forward(input, granularities[sampler.nextInt(granularities.length)])
                : forward(input, activeWidth);
    }

    /**
     * Parameters a sub-network of {@code width} hidden units uses.
     *
     * @param width the hidden units
     * @return the count
     */
    public long parameterCount(int width) {
        return 3L * features * width;
    }

    @Override
    public List<Var> parameters() {
        return List.of(gate, up, down);
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "NestedFeedForward(" + features + ", hidden=" + hidden + ", granularities="
                + Arrays.toString(granularities) + ")";
    }
}
