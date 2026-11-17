package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * DLinear (Zeng et al., 2022, <a href="https://arxiv.org/abs/2205.13504">arXiv:2205.13504</a>,
 * "Are Transformers Effective for Time Series Forecasting?"): split the lookback window
 * into a trend (a moving average) and a remainder, map each to the horizon with one
 * linear layer, and add. Embarrassingly simple, it matched or beat the transformer
 * forecasters of its day, so it is the baseline every new forecaster must beat.
 *
 * <p>Input {@code [batch, lookback]}, output {@code [batch, horizon]}.
 */
public final class DLinear implements Layer {

    private final int lookback;
    private final int horizon;
    private final Tensor movingAverage;
    private final Var trend;
    private final Var seasonal;
    private final Var bias;

    /**
     * Creates the model.
     *
     * @param lookback input steps
     * @param horizon forecast steps
     * @param kernel the moving-average width (odd; edges are replicate-padded)
     * @param random the initialization source
     */
    public DLinear(int lookback, int horizon, int kernel, SplittableRandom random) {
        this.lookback = lookback;
        this.horizon = horizon;
        float[] m = new float[lookback * lookback];
        int half = kernel / 2;
        for (int t = 0; t < lookback; t++) {
            for (int k = -half; k <= half; k++) {
                int src = Math.max(0, Math.min(lookback - 1, t + k));
                m[src * lookback + t] += 1f / (2 * half + 1);
            }
        }
        movingAverage = Tensor.of(Shape.of(lookback, lookback), m);
        float init = 1f / lookback;
        trend = Var.param(Tensor.full(Shape.of(lookback, horizon), init));
        seasonal = Var.param(Tensor.uniform(Shape.of(lookback, horizon), -init, init, random));
        bias = Var.param(Tensor.zeros(Shape.of(1, horizon)));
    }

    /**
     * @param horizon forecast steps
     * @param kernel moving-average width
     * @return a spec for {@code Model.sequential().input(lookback).add(...)}
     */
    public static LayerSpec spec(int horizon, int kernel) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(horizon);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new DLinear(input.dim(-1), horizon, kernel, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (input.shape().rank() != 2 || input.shape().dim(1) != lookback) {
            throw new IllegalArgumentException("DLinear expects [batch, " + lookback + "], got " + input.shape());
        }
        Var t = input.matmul(Var.of(movingAverage));
        Var s = input.sub(t);
        return t.matmul(trend).add(s.matmul(seasonal)).add(bias);
    }

    @Override
    public List<Var> parameters() {
        return List.of(trend, seasonal, bias);
    }

    @Override
    public int outputSize(int inputSize) {
        return horizon;
    }

    @Override
    public String toString() {
        return "DLinear(" + lookback + " -> " + horizon + ")";
    }
}
