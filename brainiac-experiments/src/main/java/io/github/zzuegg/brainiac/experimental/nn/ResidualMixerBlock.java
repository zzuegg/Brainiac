package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A pre-norm transformer block around any sequence mixer and any feed-forward layer:
 *
 * <pre>
 * h = x + mixer(canonA(rms(x)))
 * y = h + ffn(canonC(rms(h)))
 * </pre>
 *
 * <p>The mixer is any {@link LayerSpec} that keeps {@code [steps, features]} (attention
 * variants, state-space layers); the feed-forward network defaults to SwiGLU with twice the
 * width and can be any other shape-keeping spec, such as a mixture of experts. The
 * optional {@link CanonLayer}s ("Canon-A" and "Canon-C") are off by default. With
 * {@code RefinedAttention.of(h).causal(true).rotary(10000)} as the mixer this is the core
 * {@code TransformerBlock.of(h).causal(true).rmsNorm(true).gated(true).rotary(10000)}, so a
 * single option can be compared against that baseline. Auxiliary losses of the mixer and the
 * feed-forward layer (such as MoE balance losses) are passed through.
 */
public final class ResidualMixerBlock implements Layer {

    private final int features;
    private final Layer mixer;
    private final Layer ffn;
    private final Layer canonA;
    private final Layer canonC;
    private final Var gain1;
    private final Var gain2;

    /**
     * Creates the block from finished layers.
     *
     * @param features the width
     * @param mixer the sequence mixer
     * @param ffn the feed-forward layer, or {@code null} for none
     * @param canonA a layer before the mixer (after the norm), or {@code null}
     * @param canonC a layer before the feed-forward layer, or {@code null}
     */
    public ResidualMixerBlock(int features, Layer mixer, Layer ffn, Layer canonA, Layer canonC) {
        this.features = features;
        this.mixer = mixer;
        this.ffn = ffn;
        this.canonA = canonA;
        this.canonC = canonC;
        this.gain1 = RefinementOps.ones(features);
        this.gain2 = RefinementOps.ones(features);
    }

    /**
     * Starts a spec around a mixer, with a SwiGLU feed-forward network of twice the width and
     * no canon layers.
     *
     * @param mixer the mixer spec
     * @return the spec
     */
    public static Spec of(LayerSpec mixer) {
        return new Spec(mixer, GluFeedForward.spec(GluFeedForward.Kind.SWIGLU, 2f), 0);
    }

    /** An immutable block description. */
    public static final class Spec implements LayerSpec {
        private final LayerSpec mixer;
        private final LayerSpec ffn;
        private final int canon;

        private Spec(LayerSpec mixer, LayerSpec ffn, int canon) {
            this.mixer = mixer;
            this.ffn = ffn;
            this.canon = canon;
        }

        /** @param f the feed-forward spec, or {@code null} for a mixer-only block @return the new spec */
        public Spec ffn(LayerSpec f) {
            return new Spec(mixer, f, canon);
        }

        /** @param kernel the canon kernel length, 0 for no canon layers @return the new spec */
        public Spec canon(int kernel) {
            return new Spec(mixer, ffn, kernel);
        }

        @Override
        public Shape outputShape(Shape input) {
            return input;
        }

        @Override
        public ResidualMixerBlock create(Shape input, SplittableRandom random) {
            int d = input.dim(input.rank() - 1);
            return new ResidualMixerBlock(d, mixer.create(input, random),
                    ffn == null ? null : ffn.create(input, random),
                    canon > 0 ? new CanonLayer(d, canon, random) : null,
                    canon > 0 && ffn != null ? new CanonLayer(d, canon, random) : null);
        }
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        MixerOps.requireSequence(input, features, "ResidualMixerBlock");
        Var a = RefinementOps.rms(input, gain1);
        if (canonA != null) {
            a = canonA.forward(a, training);
        }
        Var h = input.add(mixer.forward(a, training));
        if (ffn == null) {
            return h;
        }
        Var c = RefinementOps.rms(h, gain2);
        if (canonC != null) {
            c = canonC.forward(c, training);
        }
        return h.add(ffn.forward(c, training));
    }

    /** @return the mixer */
    public Layer mixer() {
        return mixer;
    }

    /** @return the feed-forward layer, or {@code null} */
    public Layer ffn() {
        return ffn;
    }

    @Override
    public Var auxiliaryLoss() {
        Var a = mixer.auxiliaryLoss();
        Var b = ffn == null ? null : ffn.auxiliaryLoss();
        return a == null ? b : b == null ? a : a.add(b);
    }

    @Override
    public void resetState() {
        mixer.resetState();
        if (ffn != null) {
            ffn.resetState();
        }
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(gain1, gain2));
        p.addAll(mixer.parameters());
        for (Layer l : new Layer[] {ffn, canonA, canonC}) {
            if (l != null) {
                p.addAll(l.parameters());
            }
        }
        return p;
    }

    @Override
    public String toString() {
        return "ResidualMixerBlock(" + mixer + (ffn == null ? "" : ", " + ffn) + (canonA != null ? ", canon" : "")
                + ")";
    }
}
