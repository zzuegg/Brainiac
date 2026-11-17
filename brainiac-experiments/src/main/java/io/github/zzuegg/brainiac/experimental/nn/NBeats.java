package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * N-BEATS (Oreshkin et al., 2019, <a href="https://arxiv.org/abs/1905.10437">arXiv:1905.10437</a>):
 * univariate forecasting with nothing but fully connected layers and <b>doubly residual
 * stacking</b>. Each block reads the remaining lookback window, predicts coefficients
 * {@code theta}, and expands them through a basis into a backcast (subtracted from the
 * block's input, so the next block models what is left) and a partial forecast (summed
 * over blocks).
 *
 * <p>Blocks are {@link BlockType#GENERIC} (learned linear basis) or, for the
 * interpretable configuration, {@link BlockType#TREND} (low-order polynomials) and
 * {@link BlockType#SEASONALITY} (Fourier harmonics), whose partial forecasts can be read
 * as trend and seasonal components.
 *
 * <p>Input {@code [batch, lookback]}, output {@code [batch, horizon]}.
 */
public final class NBeats implements Layer {

    /** The basis a block expands its coefficients through. */
    public enum BlockType {
        /** a learned linear projection */
        GENERIC,
        /** polynomials {@code t^0 .. t^degree} */
        TREND,
        /** cosines and sines of integer frequencies */
        SEASONALITY
    }

    private final class Block {
        final BlockType type;
        final List<Var> fc = new ArrayList<>();
        final Var thetaBack;
        final Var thetaFore;
        final Tensor basisBack;
        final Tensor basisFore;

        Block(BlockType type, int hidden, int layers, int thetaSize, SplittableRandom r) {
            this.type = type;
            int in = lookback;
            for (int l = 0; l < layers; l++) {
                fc.add(Var.param(Tensor.normal(Shape.of(in, hidden), 0, (float) Math.sqrt(2.0 / in), r)));
                fc.add(Var.param(Tensor.zeros(Shape.of(1, hidden))));
                in = hidden;
            }
            if (type == BlockType.GENERIC) {
                thetaBack = Var.param(Tensor.normal(Shape.of(hidden, lookback), 0, (float) Math.sqrt(1.0 / hidden), r));
                thetaFore = Var.param(Tensor.normal(Shape.of(hidden, horizon), 0, (float) Math.sqrt(1.0 / hidden), r));
                basisBack = null;
                basisFore = null;
            } else {
                basisBack = basis(type, thetaSize, lookback, -lookback);
                basisFore = basis(type, thetaSize, horizon, 0);
                int p = basisBack.shape().dim(0);
                thetaBack = Var.param(Tensor.normal(Shape.of(hidden, p), 0, (float) Math.sqrt(1.0 / hidden), r));
                thetaFore = Var.param(Tensor.normal(Shape.of(hidden, p), 0, (float) Math.sqrt(1.0 / hidden), r));
            }
        }

        Var[] apply(Var x) {
            Var h = x;
            for (int l = 0; l < fc.size(); l += 2) {
                h = h.matmul(fc.get(l)).add(fc.get(l + 1)).relu();
            }
            if (type == BlockType.GENERIC) {
                return new Var[] {h.matmul(thetaBack), h.matmul(thetaFore)};
            }
            return new Var[] {h.matmul(thetaBack).matmul(basisBack), h.matmul(thetaFore).matmul(basisFore)};
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(fc);
            p.add(thetaBack);
            p.add(thetaFore);
            return p;
        }
    }

    private final int lookback;
    private final int horizon;
    private final List<Block> blocks = new ArrayList<>();

    /**
     * Creates the model.
     *
     * @param lookback input steps
     * @param horizon forecast steps
     * @param types the block types, in order
     * @param hidden fully connected width
     * @param layers fully connected layers per block
     * @param random the initialization source
     */
    public NBeats(int lookback, int horizon, List<BlockType> types, int hidden, int layers, SplittableRandom random) {
        this.lookback = lookback;
        this.horizon = horizon;
        for (BlockType t : types) {
            blocks.add(new Block(t, hidden, layers, t == BlockType.TREND ? 3 : Math.max(1, horizon / 2 - 1), random));
        }
    }

    /**
     * The generic configuration.
     *
     * @param lookback input steps
     * @param horizon forecast steps
     * @param blocks the number of generic blocks
     * @param hidden fully connected width
     * @param seed the seed
     * @return the model
     */
    public static NBeats generic(int lookback, int horizon, int blocks, int hidden, long seed) {
        return new NBeats(lookback, horizon, java.util.Collections.nCopies(blocks, BlockType.GENERIC), hidden, 2,
                new SplittableRandom(seed));
    }

    /**
     * The interpretable configuration: a trend block then a seasonality block.
     *
     * @param lookback input steps
     * @param horizon forecast steps
     * @param hidden fully connected width
     * @param seed the seed
     * @return the model
     */
    public static NBeats interpretable(int lookback, int horizon, int hidden, long seed) {
        return new NBeats(lookback, horizon, List.of(BlockType.TREND, BlockType.SEASONALITY), hidden, 2,
                new SplittableRandom(seed));
    }

    /**
     * @param horizon forecast steps
     * @param blocks generic blocks
     * @param hidden width
     * @return a spec for {@code Model.sequential().input(lookback).add(...)}
     */
    public static LayerSpec spec(int horizon, int blocks, int hidden) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(horizon);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new NBeats(input.dim(-1), horizon, java.util.Collections.nCopies(blocks, BlockType.GENERIC),
                        hidden, 2, random);
            }
        };
    }

    /** Rows: basis functions; columns: time steps (normalized to [0, 1) or [-1, 0)). */
    private static Tensor basis(BlockType type, int size, int steps, int offset) {
        int rows = type == BlockType.TREND ? size : 1 + 2 * size;
        float[] b = new float[rows * steps];
        for (int t = 0; t < steps; t++) {
            double tau = (t + offset) / (double) Math.max(1, steps);
            if (type == BlockType.TREND) {
                for (int p = 0; p < size; p++) {
                    b[p * steps + t] = (float) Math.pow(tau, p);
                }
            } else {
                double phase = 2 * Math.PI * (t + offset);
                b[t] = 1;
                for (int k = 1; k <= size; k++) {
                    b[(2 * k - 1) * steps + t] = (float) Math.cos(phase * k / (2.0 * size + 2));
                    b[(2 * k) * steps + t] = (float) Math.sin(phase * k / (2.0 * size + 2));
                }
            }
        }
        return Tensor.of(Shape.of(rows, steps), b);
    }

    /**
     * Forecasts with the partial forecast of every block, for interpretation.
     *
     * @param input {@code [batch, lookback]}
     * @return one {@code [batch, horizon]} tensor per block; they sum to the forecast
     */
    public List<Tensor> components(Tensor input) {
        List<Tensor> parts = new ArrayList<>();
        Var residual = Var.of(input);
        for (Block b : blocks) {
            Var[] bf = b.apply(residual);
            residual = residual.sub(bf[0]);
            parts.add(bf[1].value());
        }
        return parts;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (input.shape().rank() != 2 || input.shape().dim(1) != lookback) {
            throw new IllegalArgumentException("NBeats expects [batch, " + lookback + "], got " + input.shape());
        }
        Var residual = input;
        Var forecast = null;
        for (Block b : blocks) {
            Var[] bf = b.apply(residual);
            residual = residual.sub(bf[0]);
            forecast = forecast == null ? bf[1] : forecast.add(bf[1]);
        }
        return forecast;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Block b : blocks) {
            p.addAll(b.parameters());
        }
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return horizon;
    }

    @Override
    public String toString() {
        return "NBeats(" + lookback + " -> " + horizon + ", " + blocks.size() + " blocks)";
    }
}
