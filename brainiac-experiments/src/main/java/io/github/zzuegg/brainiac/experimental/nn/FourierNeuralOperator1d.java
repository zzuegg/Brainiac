package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A one-dimensional Fourier Neural Operator: learns a map between functions sampled on a
 * uniform grid, such as from an initial condition to the solution of a PDE. The input
 * function (plus the grid coordinate) is lifted to {@code width} channels, passed through
 * Fourier layers
 *
 * <pre>
 * v = gelu(F^-1(R . truncate(F v)) + v W)
 * </pre>
 *
 * <p>and projected to the output channels. {@code R} holds complex weights for only the
 * lowest {@code modes} frequencies, so the parameters do not depend on the grid: a model
 * trained at one resolution can be evaluated at another (zero-shot super-resolution).
 * The transforms use the core FFT ({@code Var.rfft}/{@code irfft}).
 *
 * <p>Input {@code [batch, points, inChannels]}, output {@code [batch, points, outChannels]};
 * the points cover {@code [0, 1)} periodically.
 *
 * <p>Paper: Li et al., "Fourier Neural Operator for Parametric Partial Differential
 * Equations", <a href="https://arxiv.org/abs/2010.08895">arXiv:2010.08895</a>.
 */
public final class FourierNeuralOperator1d implements Layer {

    private final int inChannels;
    private final int outChannels;
    private final int width;
    private final Var lift;
    private final Var liftBias;
    private final List<Var[]> layers = new ArrayList<>();
    private final Var project1;
    private final Var project2;

    /**
     * Creates the operator.
     *
     * @param inChannels channels of the input function
     * @param outChannels channels of the output function
     * @param width the hidden channels
     * @param modes the Fourier modes kept per layer
     * @param depth the number of Fourier layers
     * @param random the random source
     */
    public FourierNeuralOperator1d(int inChannels, int outChannels, int width, int modes, int depth,
            RandomGenerator random) {
        this.inChannels = inChannels;
        this.outChannels = outChannels;
        this.width = width;
        this.lift = MixerOps.weight(random, inChannels + 1, width);
        this.liftBias = MixerOps.filled(0f, width);
        float scale = 1f / (width * width);
        for (int i = 0; i < depth; i++) {
            layers.add(new Var[] {
                Var.param(Tensor.uniform(Shape.of(modes, width, width), 0, scale, random)),
                Var.param(Tensor.uniform(Shape.of(modes, width, width), 0, scale, random)),
                MixerOps.weight(random, width, width),
                MixerOps.filled(0f, width)});
        }
        this.project1 = MixerOps.weight(random, width, 2 * width);
        this.project2 = MixerOps.weight(random, 2 * width, outChannels);
    }

    /**
     * Describes the operator for {@code Model.sequential().add(...)}; the input is
     * {@code [points, inChannels]}.
     *
     * @param outChannels channels of the output function
     * @param width hidden channels
     * @param modes Fourier modes kept
     * @param depth Fourier layers
     * @return the spec
     */
    public static LayerSpec of(int outChannels, int width, int modes, int depth) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(0), outChannels);
            }

            @Override
            public Layer create(Shape input, java.util.SplittableRandom random) {
                return new FourierNeuralOperator1d(input.dim(-1), outChannels, width, modes, depth, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.rank() != 3 || s.dim(2) != inChannels) {
            throw new IllegalArgumentException("FourierNeuralOperator1d expects [batch, points, " + inChannels
                    + "], got " + s);
        }
        int b = s.dim(0);
        int n = s.dim(1);
        float[] grid = new float[b * n];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < n; j++) {
                grid[i * n + j] = j / (float) n;
            }
        }
        Var x = Var.concat(2, input, Var.of(Tensor.of(Shape.of(b, n, 1), grid)));
        Var v = x.matmul(lift).add(liftBias);   // [b, n, width]
        for (int i = 0; i < layers.size(); i++) {
            Var[] l = layers.get(i);
            Var spectral = MixerOps.spectralConvolution(v.permute(0, 2, 1), l[0], l[1]).permute(0, 2, 1);
            v = spectral.add(v.matmul(l[2])).add(l[3]);
            if (i < layers.size() - 1) {
                v = Activation.GELU.forward(v);
            }
        }
        return Activation.GELU.forward(v.matmul(project1)).matmul(project2);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(lift, liftBias, project1, project2));
        layers.forEach(l -> p.addAll(List.of(l)));
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return outChannels;
    }

    @Override
    public String toString() {
        return "FourierNeuralOperator1d(" + inChannels + " -> " + outChannels + ", width=" + width + ")";
    }
}
