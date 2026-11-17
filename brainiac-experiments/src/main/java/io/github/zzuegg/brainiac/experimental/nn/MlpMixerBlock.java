package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * An MLP-Mixer block over patch tokens {@code [batch, tokens, dim]}: a token-mixing MLP
 * applied across the tokens of each channel (it learns fixed spatial mixing, in place of
 * attention), then a channel-mixing MLP applied to each token, each with pre-LayerNorm and a
 * residual connection. Only matrix multiplications, so it is simple and fast; with enough
 * data it rivals attention, and it is the template of many "attention-free" designs.
 *
 * <p>Paper: Tolstikhin et al., "MLP-Mixer: An all-MLP Architecture for Vision" (2021),
 * <a href="https://arxiv.org/abs/2105.01601">arXiv:2105.01601</a>.
 */
public final class MlpMixerBlock implements Layer {

    private final int tokens;
    private final int dim;
    private final LayerNorm norm1;
    private final LayerNorm norm2;
    private final Var tokenUp;
    private final Var tokenDown;
    private final Var channelUp;
    private final Var channelDown;

    /**
     * Creates the block.
     *
     * @param tokens the tokens per sample
     * @param dim the channels per token
     * @param tokenHidden the token-mixing MLP's hidden width
     * @param channelHidden the channel-mixing MLP's hidden width
     * @param random the random source
     */
    public MlpMixerBlock(int tokens, int dim, int tokenHidden, int channelHidden, RandomGenerator random) {
        this.tokens = tokens;
        this.dim = dim;
        norm1 = LayerNorm.of(dim);
        norm2 = LayerNorm.of(dim);
        tokenUp = TechniqueOps.weight(random, tokens, tokenHidden);
        tokenDown = TechniqueOps.weight(random, tokenHidden, tokens);
        channelUp = TechniqueOps.weight(random, dim, channelHidden);
        channelDown = TechniqueOps.weight(random, channelHidden, dim);
    }

    /**
     * Describes the block for {@code Model.sequential().add(...)} on {@code [tokens, dim]}.
     *
     * @param tokenFactor the token-mixing hidden width as a multiple of the tokens
     * @param channelFactor the channel-mixing hidden width as a multiple of the channels
     * @return the spec
     */
    public static LayerSpec spec(float tokenFactor, float channelFactor) {
        return TechniqueOps.sameShape((Shape in, SplittableRandom r) -> new MlpMixerBlock(in.dim(0), in.dim(1),
                Math.max(1, Math.round(in.dim(0) * tokenFactor)), Math.max(1, Math.round(in.dim(1) * channelFactor)),
                r));
    }

    @Override
    public Var forward(Var input) {
        if (input.shape().rank() != 3 || input.shape().dim(1) != tokens || input.shape().dim(2) != dim) {
            throw new IllegalArgumentException(this + " expects [batch, " + tokens + ", " + dim + "], got "
                    + input.shape());
        }
        Var t = norm1.forward(input).permute(0, 2, 1);                    // [b, dim, tokens]
        t = Activation.GELU.forward(t.matmul(tokenUp)).matmul(tokenDown);
        Var x = input.add(t.permute(0, 2, 1));
        Var c = Activation.GELU.forward(norm2.forward(x).matmul(channelUp)).matmul(channelDown);
        return x.add(c);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(norm1.parameters());
        p.addAll(norm2.parameters());
        p.addAll(List.of(tokenUp, tokenDown, channelUp, channelDown));
        return p;
    }

    @Override
    public String toString() {
        return "MlpMixerBlock(" + tokens + " tokens, " + dim + ")";
    }
}
