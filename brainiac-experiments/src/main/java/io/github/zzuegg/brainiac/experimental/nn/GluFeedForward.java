package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A transformer feed-forward network as a standalone layer: {@code down(f(up x))} or, gated,
 * {@code down(f(gate x) * up x)} (GLU family). The gated forms with SiLU (SwiGLU), GELU
 * (GeGLU) and ReLU (ReGLU) beat the plain two-layer MLP at equal parameter count and are the
 * feed-forward of Llama, PaLM, Gemma and most current LLMs; {@link Kind#SQUARED_RELU} is the
 * Primer variant. Works on any input whose last axis holds the features.
 *
 * <p>Papers: Shazeer, "GLU Variants Improve Transformer" (2020),
 * <a href="https://arxiv.org/abs/2002.05202">arXiv:2002.05202</a>; So et al., "Primer"
 * (2021), <a href="https://arxiv.org/abs/2109.08668">arXiv:2109.08668</a>.
 */
public final class GluFeedForward implements Layer {

    /** The feed-forward variants. */
    public enum Kind {
        /** {@code down(silu(gate x) * up x)}. */
        SWIGLU(true, Activation.SILU),
        /** {@code down(gelu(gate x) * up x)}. */
        GEGLU(true, Activation.GELU),
        /** {@code down(relu(gate x) * up x)}. */
        REGLU(true, Activation.RELU),
        /** {@code down(sigmoid(gate x) * up x)}: the original GLU. */
        GLU(true, Activation.SIGMOID),
        /** {@code down(gelu(up x))}: the classic transformer MLP. */
        GELU_MLP(false, Activation.GELU),
        /** {@code down(relu(up x)^2)}: Primer's squared ReLU. */
        SQUARED_RELU(false, ModernActivations.SQUARED_RELU);

        private final boolean gated;
        private final Activation activation;

        Kind(boolean gated, Activation activation) {
            this.gated = gated;
            this.activation = activation;
        }

        /** @return whether this variant multiplies by a gate */
        public boolean gated() {
            return gated;
        }

        /** @return the activation applied to the gate (or to the hidden layer) */
        public Activation activation() {
            return activation;
        }
    }

    private final Kind kind;
    private final int features;
    private final int hidden;
    private final Var up;
    private final Var gate;
    private final Var down;

    /**
     * Creates the layer.
     *
     * @param kind the variant
     * @param features the input and output width
     * @param hidden the hidden width; gated variants usually use {@code 8/3 * features} to
     *     match the plain MLP's {@code 4 * features} in parameters
     * @param random the random source
     */
    public GluFeedForward(Kind kind, int features, int hidden, RandomGenerator random) {
        this.kind = kind;
        this.features = features;
        this.hidden = hidden;
        this.up = TechniqueOps.weight(random, features, hidden);
        this.gate = kind.gated ? TechniqueOps.weight(random, features, hidden) : null;
        this.down = TechniqueOps.weight(random, hidden, features);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param kind the variant
     * @param hiddenFactor the hidden width as a multiple of the input width
     * @return the spec
     */
    public static LayerSpec spec(Kind kind, float hiddenFactor) {
        return TechniqueOps.sameShape((Shape in, SplittableRandom r) -> {
            int d = in.dim(in.rank() - 1);
            return new GluFeedForward(kind, d, Math.max(1, Math.round(d * hiddenFactor)), r);
        });
    }

    @Override
    public Var forward(Var input) {
        Var h;
        if (kind.gated) {
            h = kind.activation.forward(input.matmul(gate)).mul(input.matmul(up));
        } else {
            h = kind.activation.forward(input.matmul(up));
        }
        return h.matmul(down);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(up, down));
        if (gate != null) {
            p.add(gate);
        }
        return p;
    }

    /** @return the variant */
    public Kind kind() {
        return kind;
    }

    @Override
    public String toString() {
        return "GluFeedForward(" + kind + ", " + features + " -> " + hidden + ")";
    }
}
