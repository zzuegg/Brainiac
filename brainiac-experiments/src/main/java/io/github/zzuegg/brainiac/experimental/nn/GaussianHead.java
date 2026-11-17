package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A probabilistic output head: maps features {@code [..., in]} to the mean and standard
 * deviation of a Gaussian per position, {@code [..., 2]} (the deviation passed through
 * softplus, plus a floor). Trained with the Gaussian negative log-likelihood
 * ({@link #nll}), it predicts its own uncertainty, which is what DeepAR-style forecasters
 * ({@link DeepAr}) need to sample future paths and quantiles instead of a single point.
 */
public final class GaussianHead implements Layer {

    private static final float MIN_STD = 1e-3f;
    private static final float HALF_LOG_TWO_PI = (float) (0.5 * Math.log(2 * Math.PI));
    private final int in;
    private final Var weight;
    private final Var bias;

    /**
     * Creates the head.
     *
     * @param in input features
     * @param random the initialization source
     */
    public GaussianHead(int in, SplittableRandom random) {
        this.in = in;
        weight = Var.param(Tensor.normal(Shape.of(in, 2), 0, (float) Math.sqrt(1.0 / in), random));
        bias = Var.param(Tensor.zeros(Shape.of(1, 2)));
    }

    /** @return a spec for {@code Model.sequential().add(...)} */
    public static LayerSpec spec() {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] d = input.toArray();
                d[d.length - 1] = 2;
                return Shape.of(d);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new GaussianHead(input.dim(-1), random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        int last = input.shape().rank() - 1;
        if (input.shape().dim(last) != in) {
            throw new IllegalArgumentException("GaussianHead expects " + in + " features, got " + input.shape());
        }
        Var raw = input.matmul(weight).add(bias);
        Var mean = raw.slice(last, 0, 1);
        Var std = raw.slice(last, 1, 2).softplus().add(MIN_STD);
        return Var.concat(last, mean, std);
    }

    /**
     * The mean negative log-likelihood of targets.
     *
     * @param output the head's output {@code [..., 2]}
     * @param target the targets {@code [..., 1]}
     * @return the scalar loss
     */
    public static Var nll(Var output, Tensor target) {
        int last = output.shape().rank() - 1;
        Var mean = output.slice(last, 0, 1);
        Var std = output.slice(last, 1, 2);
        Var z = mean.sub(target).div(std);
        return z.square().mul(0.5f).add(std.log()).add(HALF_LOG_TWO_PI).mean();
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, bias);
    }

    @Override
    public int outputSize(int inputSize) {
        return 2;
    }

    @Override
    public String toString() {
        return "GaussianHead(" + in + ")";
    }
}
