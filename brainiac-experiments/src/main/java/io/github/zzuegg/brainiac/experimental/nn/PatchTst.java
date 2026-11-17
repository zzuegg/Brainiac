package io.github.zzuegg.brainiac.experimental.nn;

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
 * PatchTST (Nie et al., 2022, <a href="https://arxiv.org/abs/2211.14730">arXiv:2211.14730</a>):
 * a transformer for long-horizon forecasting that attends over <b>patches</b> of the
 * series instead of single steps. The lookback window is cut into (possibly overlapping)
 * patches of {@code patchLength} steps, each linearly embedded into one token with a
 * learned position; a non-causal transformer encoder mixes the tokens; a linear head maps
 * the flattened tokens to the whole horizon at once. Patching shortens the sequence
 * quadratically for attention and gives every token local semantics; channels are handled
 * independently (feed each univariate series as its own row), and each window is
 * instance-normalized (see {@link RevIn}).
 *
 * <p>Input {@code [batch, lookback]}, output {@code [batch, horizon]}.
 */
public final class PatchTst implements Layer {

    private final int lookback;
    private final int horizon;
    private final int patchLength;
    private final int stride;
    private final int patches;
    private final int width;
    private final Var embed;
    private final Var embedBias;
    private final Var position;
    private final Var head;
    private final Var headBias;
    private final List<Layer> blocks = new ArrayList<>();
    private final boolean normalize;

    /**
     * Creates the model.
     *
     * @param lookback input steps
     * @param horizon forecast steps
     * @param patchLength steps per patch
     * @param stride steps between patch starts
     * @param width the model width
     * @param layers transformer blocks
     * @param heads attention heads
     * @param normalize whether to instance-normalize each window (RevIN)
     * @param random the initialization source
     */
    public PatchTst(int lookback, int horizon, int patchLength, int stride, int width, int layers, int heads,
            boolean normalize, SplittableRandom random) {
        if (patchLength > lookback) {
            throw new IllegalArgumentException("patches must fit into the lookback window");
        }
        this.lookback = lookback;
        this.horizon = horizon;
        this.patchLength = patchLength;
        this.stride = stride;
        this.patches = (lookback - patchLength) / stride + 1;
        this.width = width;
        this.normalize = normalize;
        embed = Var.param(Tensor.normal(Shape.of(patchLength, width), 0, (float) Math.sqrt(1.0 / patchLength), random));
        embedBias = Var.param(Tensor.zeros(Shape.of(1, width)));
        position = Var.param(Tensor.normal(Shape.of(patches, width), 0, 0.02f, random));
        head = Var.param(Tensor.normal(Shape.of(patches * width, horizon), 0,
                (float) Math.sqrt(1.0 / (patches * width)), random));
        headBias = Var.param(Tensor.zeros(Shape.of(1, horizon)));
        for (int l = 0; l < layers; l++) {
            blocks.add(TransformerBlock.of(heads).ffnSize(2 * width).create(Shape.of(patches, width), random));
        }
    }

    /**
     * @param horizon forecast steps
     * @param patchLength steps per patch
     * @param stride patch stride
     * @param width model width
     * @param layers transformer blocks
     * @return a spec for {@code Model.sequential().input(lookback).add(...)}
     */
    public static LayerSpec spec(int horizon, int patchLength, int stride, int width, int layers) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(horizon);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new PatchTst(input.dim(-1), horizon, patchLength, stride, width, layers, 2, true, random);
            }
        };
    }

    /** @return the number of patch tokens */
    public int patches() {
        return patches;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (input.shape().rank() != 2 || input.shape().dim(1) != lookback) {
            throw new IllegalArgumentException("PatchTst expects [batch, " + lookback + "], got " + input.shape());
        }
        int b = input.shape().dim(0);
        RevIn.Stats stats = normalize ? RevIn.stats(input.value()) : null;
        Var x = normalize ? RevIn.normalize(input, stats) : input;
        Var[] parts = new Var[patches];
        for (int p = 0; p < patches; p++) {
            parts[p] = x.slice(1, p * stride, p * stride + patchLength).reshape(b, 1, patchLength);
        }
        Var tokens = (patches == 1 ? parts[0] : Var.concat(1, parts)).matmul(embed).add(embedBias).add(position);
        Var h = tokens;
        for (Layer block : blocks) {
            h = block.forward(h, training);
        }
        Var y = h.reshape(b, patches * width).matmul(head).add(headBias);
        return normalize ? RevIn.denormalize(y, stats) : y;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embed, embedBias, position, head, headBias));
        for (Layer block : blocks) {
            p.addAll(block.parameters());
        }
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return horizon;
    }

    @Override
    public String toString() {
        return "PatchTst(" + patches + " patches of " + patchLength + ", width " + width + ")";
    }
}
