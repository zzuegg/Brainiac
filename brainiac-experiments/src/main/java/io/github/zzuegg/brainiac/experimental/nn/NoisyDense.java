package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A dense layer with learned, factorized Gaussian parameter noise: Noisy Nets (Fortunato
 * et al., 2017, <a href="https://arxiv.org/abs/1706.10295">arXiv:1706.10295</a>), the
 * exploration part of Rainbow DQN.
 *
 * <p>The weight is {@code mu_W + sigma_W * (f(eps_in) f(eps_out)^T)} with
 * {@code f(x) = sign(x) sqrt|x|}; the bias is {@code mu_b + sigma_b * f(eps_out)}. Both
 * {@code mu} and {@code sigma} are trained, so the network learns how much to explore in
 * each state, replacing epsilon-greedy. Noise is used when the layer runs in training mode
 * and stays fixed until {@link #resample()}; in inference mode only the means are used.
 *
 * <pre>{@code
 * Model q = Model.sequential().input(4).dense(64, Activation.RELU)
 *         .add(NoisyDense.spec(64, 0.5f, 7)).activation(Activation.RELU)
 *         .add(NoisyDense.spec(2, 0.5f, 8)).build();
 * }</pre>
 */
public final class NoisyDense implements Layer {

    private final int inputSize;
    private final int units;
    private final Var weightMu;
    private final Var weightSigma;
    private final Var biasMu;
    private final Var biasSigma;
    private final SplittableRandom random;
    private Tensor weightNoise;
    private Tensor biasNoise;

    /**
     * Creates the layer with the paper's initialization.
     *
     * @param inputSize values in
     * @param units values out
     * @param sigma0 the initial noise scale (0.5 in the paper), divided by {@code sqrt(inputSize)}
     * @param random the initialization and noise source
     */
    public NoisyDense(int inputSize, int units, float sigma0, SplittableRandom random) {
        this.inputSize = inputSize;
        this.units = units;
        this.random = random;
        float bound = (float) (1 / Math.sqrt(inputSize));
        float sigma = sigma0 * bound;
        weightMu = Var.param(Tensor.uniform(Shape.of(inputSize, units), -bound, bound, random));
        weightSigma = Var.param(Tensor.full(Shape.of(inputSize, units), sigma));
        biasMu = Var.param(Tensor.uniform(Shape.of(1, units), -bound, bound, random));
        biasSigma = Var.param(Tensor.full(Shape.of(1, units), sigma));
        drawNoise();
    }

    /**
     * Describes a noisy layer for {@code Model.sequential().add(...)}.
     *
     * @param units values out
     * @param sigma0 the initial noise scale
     * @param seed the noise seed
     * @return the spec
     */
    public static LayerSpec spec(int units, float sigma0, long seed) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] d = input.toArray();
                d[d.length - 1] = units;
                return Shape.of(d);
            }

            @Override
            public Layer create(Shape input, SplittableRandom r) {
                return new NoisyDense(input.dim(-1), units, sigma0, new SplittableRandom(seed ^ r.nextLong()));
            }
        };
    }

    private static float[] factor(SplittableRandom r, int n) {
        float[] f = new float[n];
        for (int i = 0; i < n; i++) {
            double x = r.nextGaussian();
            f[i] = (float) (Math.signum(x) * Math.sqrt(Math.abs(x)));
        }
        return f;
    }

    private void drawNoise() {
        float[] in = factor(random, inputSize);
        float[] out = factor(random, units);
        float[] w = new float[inputSize * units];
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < units; j++) {
                w[i * units + j] = in[i] * out[j];
            }
        }
        weightNoise = Tensor.of(Shape.of(inputSize, units), w).to(weightMu.value().device());
        biasNoise = Tensor.of(Shape.of(1, units), out).to(weightMu.value().device());
    }

    /** Draws fresh noise, used by every training-mode forward until the next call. */
    public void resample() {
        drawNoise();
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (input.shape().dim(-1) != inputSize) {
            throw new IllegalArgumentException("NoisyDense expects " + inputSize + " inputs, got " + input.shape());
        }
        if (!training) {
            return input.matmul(weightMu).add(biasMu);
        }
        Var w = weightMu.add(weightSigma.mul(weightNoise));
        Var b = biasMu.add(biasSigma.mul(biasNoise));
        return input.matmul(w).add(b);
    }

    @Override
    public List<Var> parameters() {
        return List.of(weightMu, weightSigma, biasMu, biasSigma);
    }

    @Override
    public int outputSize(int in) {
        return units;
    }

    /** @return the mean absolute noise scale of the weights, which shrinks as the agent grows confident */
    public float meanSigma() {
        return weightSigma.value().abs().mean().item();
    }

    @Override
    public String toString() {
        return "NoisyDense(" + inputSize + " -> " + units + ")";
    }
}
