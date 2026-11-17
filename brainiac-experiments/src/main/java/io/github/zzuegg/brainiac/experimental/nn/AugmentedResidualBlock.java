package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * LAuReL, the learned augmented residual: a stack of {@code depth} pre-RMSNorm blocks
 * (mixer, then SwiGLU FFN) in which every sub-layer's residual {@code x + f(x)} is replaced
 * by a learned one, {@code alpha f(x) + g(x)}:
 *
 * <ul>
 *   <li>{@link Variant#RW} (residual weights): {@code alpha f(x) + beta x}, two learned
 *       scalars;</li>
 *   <li>{@link Variant#LR} (low rank): {@code f(x) + x + x A B}, a rank-{@code r} learned map
 *       next to the identity ({@code B} starts at zero);</li>
 *   <li>{@link Variant#RW_LR}: both;</li>
 *   <li>{@link Variant#PA} (previous activations): {@code f(x_i) + x_i + sum_{j<i} gamma_ij x_j},
 *       learned weights on every earlier sub-layer input;</li>
 *   <li>{@link Variant#NONE}: the plain residual, as the in-class baseline.</li>
 * </ul>
 *
 * <p>The extra parameters are tiny (two scalars, {@code 2 d r}, or one scalar per earlier
 * layer), but they let the network re-weight or reshape the skip path, which the paper finds
 * improves quality at equal size.
 *
 * <p>Paper: Menghani et al., "LAuReL: Learned Augmented Residual Layer",
 * <a href="https://arxiv.org/abs/2411.07501">arXiv:2411.07501</a>.
 */
public final class AugmentedResidualBlock implements Layer {

    /** The form of {@code g}. */
    public enum Variant {
        /** The plain residual {@code x + f(x)}. */
        NONE,
        /** Learned residual weights {@code alpha f(x) + beta x}. */
        RW,
        /** A low-rank map next to the identity. */
        LR,
        /** Residual weights and the low-rank map. */
        RW_LR,
        /** A learned mix of earlier sub-layer inputs. */
        PA
    }

    private final int features;
    private final Variant variant;
    private final List<Layer> sublayers = new ArrayList<>();
    private final List<Var> gains = new ArrayList<>();
    private final List<Var> alphas = new ArrayList<>();
    private final List<Var> betas = new ArrayList<>();
    private final List<Var> downs = new ArrayList<>();
    private final List<Var> ups = new ArrayList<>();
    private final List<Var> gammas = new ArrayList<>();

    /**
     * Creates the stack.
     *
     * @param features the width
     * @param mixer the token mixer of every block (for example causal attention), built for
     *     {@code [steps, features]}
     * @param depth the number of blocks
     * @param variant the residual form
     * @param rank the rank of the low-rank map ({@link Variant#LR}, {@link Variant#RW_LR})
     * @param steps the sequence length the mixer is built for
     * @param random the random source
     */
    public AugmentedResidualBlock(int features, LayerSpec mixer, int depth, Variant variant, int rank, int steps,
            SplittableRandom random) {
        this.features = features;
        this.variant = variant;
        for (int l = 0; l < depth; l++) {
            sublayers.add(mixer.create(Shape.of(steps, features), random));
            sublayers.add(new GluFeedForward(GluFeedForward.Kind.SWIGLU, features, 2 * features, random));
        }
        for (int i = 0; i < sublayers.size(); i++) {
            gains.add(RefinementOps.ones(features));
            alphas.add(MixerOps.filled(1f, 1));
            betas.add(MixerOps.filled(1f, 1));
            downs.add(MixerOps.uniform(random, (float) (1 / Math.sqrt(features)), features, rank));
            ups.add(MixerOps.filled(0f, rank, features));
            gammas.add(MixerOps.filled(0f, Math.max(1, i)));
        }
    }

    /**
     * Describes the stack for {@code Model.sequential().add(...)}, with rank 4.
     *
     * @param mixer the token mixer of every block
     * @param depth the number of blocks
     * @param variant the residual form
     * @return the spec
     */
    public static LayerSpec of(LayerSpec mixer, int depth, Variant variant) {
        return MixerOps.sameShape((in, r) -> new AugmentedResidualBlock(in.dim(-1), mixer, depth, variant, 4,
                in.dim(0), r));
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Var x = input;
        List<Var> earlier = new ArrayList<>();
        for (int i = 0; i < sublayers.size(); i++) {
            Var f = sublayers.get(i).forward(RefinementOps.rms(x, gains.get(i)), training);
            Var next = switch (variant) {
                case NONE -> x.add(f);
                case RW -> f.mul(alphas.get(i)).add(x.mul(betas.get(i)));
                case LR -> x.add(f).add(x.matmul(downs.get(i)).matmul(ups.get(i)));
                case RW_LR -> f.mul(alphas.get(i)).add(x.mul(betas.get(i)))
                        .add(x.matmul(downs.get(i)).matmul(ups.get(i)));
                case PA -> {
                    Var y = x.add(f);
                    for (int j = 0; j < earlier.size(); j++) {
                        y = y.add(earlier.get(j).mul(gammas.get(i).slice(0, j, j + 1)));
                    }
                    yield y;
                }
            };
            earlier.add(x);
            x = next;
        }
        return x;
    }

    /** @return the learned {@code alpha} of every sub-layer ({@link Variant#RW}, {@link Variant#RW_LR}) */
    public float[] alphas() {
        float[] out = new float[alphas.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = alphas.get(i).value().item();
        }
        return out;
    }

    /** @return the residual form */
    public Variant variant() {
        return variant;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (int i = 0; i < sublayers.size(); i++) {
            p.addAll(sublayers.get(i).parameters());
            p.add(gains.get(i));
            switch (variant) {
                case NONE -> {
                }
                case RW -> p.addAll(List.of(alphas.get(i), betas.get(i)));
                case LR -> p.addAll(List.of(downs.get(i), ups.get(i)));
                case RW_LR -> p.addAll(List.of(alphas.get(i), betas.get(i), downs.get(i), ups.get(i)));
                case PA -> {
                    if (i > 0) {
                        p.add(gammas.get(i));
                    }
                }
            }
        }
        return p;
    }

    @Override
    public Var auxiliaryLoss() {
        Var total = null;
        for (Layer l : sublayers) {
            Var a = l.auxiliaryLoss();
            if (a != null) {
                total = total == null ? a : total.add(a);
            }
        }
        return total;
    }

    @Override
    public Shape outputShape(Shape input) {
        return input;
    }

    @Override
    public String toString() {
        return "AugmentedResidualBlock(" + features + ", depth=" + sublayers.size() / 2 + ", " + variant + ")";
    }
}
