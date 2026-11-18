package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * TiDE, the time-series dense encoder (Das et al., TMLR 2023,
 * <a href="https://arxiv.org/abs/2304.08424">arXiv:2304.08424</a>): an MLP encoder-decoder
 * that uses known future covariates. Every step's covariates are projected to a few features
 * by a residual block (the "feature projection"); the lookback, the projected past and
 * future covariates are flattened into one vector and encoded by residual MLP blocks; a dense
 * decoder emits a small vector per future step, which the <b>temporal decoder</b> (a residual
 * block shared over steps) combines with that step's projected covariates, so a known price
 * step or holiday acts directly on the step it belongs to; a linear skip from the lookback is
 * added. The window is instance-normalised. A residual block is
 * {@code LayerNorm(W2 relu(W1 x) + W3 x)} (without the norm when it has one output).
 *
 * <p>Inputs: {@code [batch, lookback]} values, {@code [batch, lookback, covariates]} past and
 * {@code [batch, horizon, covariates]} future covariates; output {@code [batch, horizon]}.
 */
public final class Tide implements MultiInputLayer {

    private final int lookback;
    private final int horizon;
    private final int covariates;
    private final int projected;
    private final int decoderWidth;
    private final Residual featureProjection;
    private final List<Residual> encoder = new ArrayList<>();
    private final Residual decoder;
    private final Residual temporal;
    private final Var skip;

    /** {@code LayerNorm(W2 relu(W1 x + b1) + b2 + W3 x)}. */
    private static final class Residual {
        final Var w1;
        final Var b1;
        final Var w2;
        final Var b2;
        final Var w3;
        final LayerNorm norm;

        Residual(int in, int hidden, int out, SplittableRandom r) {
            w1 = Var.param(Tensor.normal(Shape.of(in, hidden), 0, (float) Math.sqrt(2.0 / in), r));
            b1 = Var.param(Tensor.zeros(Shape.of(hidden)));
            w2 = Var.param(Tensor.normal(Shape.of(hidden, out), 0, (float) Math.sqrt(1.0 / hidden), r));
            b2 = Var.param(Tensor.zeros(Shape.of(out)));
            w3 = Var.param(Tensor.normal(Shape.of(in, out), 0, (float) Math.sqrt(1.0 / in), r));
            norm = out > 1 ? LayerNorm.of(out) : null;   // a norm over one feature would erase it
        }

        Var apply(Var x) {
            Var y = x.matmul(w1).add(b1).relu().matmul(w2).add(b2).add(x.matmul(w3));
            return norm == null ? y : norm.forward(y);
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(w1, b1, w2, b2, w3));
            if (norm != null) {
                p.addAll(norm.parameters());
            }
            return p;
        }
    }

    /**
     * Creates the model.
     *
     * @param lookback input steps
     * @param horizon forecast steps
     * @param covariates covariates per step
     * @param hidden the encoder width
     * @param projected projected covariate features per step
     * @param decoderWidth the per-step decoder output
     * @param encoderLayers residual encoder blocks
     * @param random the initialisation source
     */
    public Tide(int lookback, int horizon, int covariates, int hidden, int projected, int decoderWidth,
            int encoderLayers, SplittableRandom random) {
        this.lookback = lookback;
        this.horizon = horizon;
        this.covariates = covariates;
        this.projected = projected;
        this.decoderWidth = decoderWidth;
        featureProjection = new Residual(covariates, hidden / 2, projected, random);
        int in = lookback + (lookback + horizon) * projected;
        for (int l = 0; l < encoderLayers; l++) {
            encoder.add(new Residual(in, hidden, hidden, random));
            in = hidden;
        }
        decoder = new Residual(hidden, hidden, horizon * decoderWidth, random);
        temporal = new Residual(decoderWidth + projected, hidden / 2, 1, random);
        skip = Var.param(Tensor.zeros(Shape.of(lookback, horizon)));
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Var y = inputs.get(0);
        Var past = inputs.get(1);
        Var future = inputs.get(2);
        int b = y.shape().dim(0);
        Var mean = y.mean(1, true);
        Var std = y.std(1, 0, true).add(1e-4f);
        Var yn = y.sub(mean).div(std);
        Var pastP = featureProjection.apply(past);                         // [b, L, p]
        Var futureP = featureProjection.apply(future);                     // [b, H, p]
        Var flat = Var.concat(1, yn, pastP.reshape(b, lookback * projected), futureP.reshape(b, horizon * projected));
        Var h = flat;
        for (Residual e : encoder) {
            h = e.apply(h);
        }
        Var perStep = decoder.apply(h).reshape(b, horizon, decoderWidth);
        Var out = temporal.apply(Var.concat(2, perStep, futureP)).reshape(b, horizon);
        return out.add(yn.matmul(skip)).mul(std).add(mean);
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return Shape.of(horizon);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(featureProjection.parameters());
        for (Residual e : encoder) {
            p.addAll(e.parameters());
        }
        p.addAll(decoder.parameters());
        p.addAll(temporal.parameters());
        p.add(skip);
        return p;
    }

    @Override
    public String toString() {
        return "Tide(" + lookback + " -> " + horizon + ", covariates=" + covariates + ")";
    }
}
