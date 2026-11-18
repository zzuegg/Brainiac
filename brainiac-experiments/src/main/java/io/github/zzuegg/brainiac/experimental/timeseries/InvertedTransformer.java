package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * iTransformer (Liu et al., ICLR 2024, <a href="https://arxiv.org/abs/2310.06625">arXiv:2310.06625</a>):
 * the transformer forecaster "inverted". Each variate's whole (instance-normalised) lookback
 * window becomes one token through a linear embedding {@code lookback -> width}; core
 * {@link TransformerBlock}s attend across the variates, so attention captures cross-channel
 * correlation (including lagged copies) while the feed-forward network works on each
 * variate's temporal representation; a linear head maps each token to its horizon, and the
 * window statistics are restored. With few variates it is cheap, and misaligned channels pose
 * no problem since no token mixes time steps of different variates.
 *
 * <p>Input {@code [batch, lookback, variates]}, output {@code [batch, horizon, variates]}.
 */
public final class InvertedTransformer implements Layer {

    private final int lookback;
    private final int horizon;
    private final int variates;
    private final Var embed;
    private final Var embedBias;
    private final List<Layer> blocks = new ArrayList<>();
    private final Var head;
    private final Var headBias;

    /**
     * Creates the model.
     *
     * @param lookback input steps
     * @param horizon forecast steps
     * @param variates channels
     * @param width the token width
     * @param layers transformer blocks
     * @param heads attention heads
     * @param random the initialisation source
     */
    public InvertedTransformer(int lookback, int horizon, int variates, int width, int layers, int heads,
            SplittableRandom random) {
        this.lookback = lookback;
        this.horizon = horizon;
        this.variates = variates;
        embed = Var.param(Tensor.normal(Shape.of(lookback, width), 0, (float) Math.sqrt(1.0 / lookback), random));
        embedBias = Var.param(Tensor.zeros(Shape.of(width)));
        for (int l = 0; l < layers; l++) {
            blocks.add(TransformerBlock.of(heads).ffnSize(2 * width).create(Shape.of(variates, width), random));
        }
        head = Var.param(Tensor.normal(Shape.of(width, horizon), 0, (float) Math.sqrt(1.0 / width), random));
        headBias = Var.param(Tensor.zeros(Shape.of(horizon)));
    }

    /**
     * @param horizon forecast steps
     * @param width token width
     * @param layers transformer blocks
     * @param heads attention heads
     * @return a spec for {@code [lookback, variates]} inputs
     */
    public static LayerSpec spec(int horizon, int width, int layers, int heads) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(horizon, input.dim(-1));
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new InvertedTransformer(input.dim(-2), horizon, input.dim(-1), width, layers, heads, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (input.shape().rank() != 3 || input.shape().dim(1) != lookback || input.shape().dim(2) != variates) {
            throw new IllegalArgumentException("InvertedTransformer expects [batch, " + lookback + ", " + variates
                    + "], got " + input.shape());
        }
        Var tokens = input.permute(0, 2, 1);                          // [b, variates, lookback]
        Var mean = tokens.mean(2, true);
        Var std = tokens.variance(2, 0, true).add(1e-5f).sqrt();
        Var h = tokens.sub(mean).div(std).matmul(embed).add(embedBias);
        for (Layer block : blocks) {
            h = block.forward(h, training);
        }
        Var y = h.matmul(head).add(headBias).mul(std).add(mean);     // [b, variates, horizon]
        return y.permute(0, 2, 1);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embed, embedBias));
        for (Layer b : blocks) {
            p.addAll(b.parameters());
        }
        p.addAll(List.of(head, headBias));
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(horizon, input.dim(-1));
    }

    @Override
    public String toString() {
        return "InvertedTransformer(" + lookback + " -> " + horizon + ", variates=" + variates + ")";
    }
}
