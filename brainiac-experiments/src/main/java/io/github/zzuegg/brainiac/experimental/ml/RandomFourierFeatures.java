package io.github.zzuegg.brainiac.experimental.ml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * Random Fourier features (Rahimi and Recht 2007,
 * <a href="https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines">Random
 * features for large-scale kernel machines</a>): by Bochner's theorem the RBF kernel
 * {@code exp(-gamma |a - b|^2)} is the expectation of {@code z(a) . z(b)} for the map
 * {@code z(x) = sqrt(2/D) cos(W x + b)}, {@code W ~ N(0, 2 gamma I)}, {@code b ~ U(0, 2 pi)}.
 * A linear model on {@code D} such features approximates the kernel machine at linear cost.
 *
 * <p>It is also a fixed (untrained) {@link Layer}, so it drops into {@code Model.sequential()}
 * through {@link #spec}: an SNGP-style Gaussian-process output head, or a random-feature
 * baseline for a learned encoder.
 */
public final class RandomFourierFeatures implements Layer {

    private final double[][] w;
    private final double[] b;
    private final double scale;
    private final Tensor wTensor;
    private final Tensor bTensor;

    /**
     * Draws the features.
     *
     * @param inputs the input dimension
     * @param features the number of features {@code D}
     * @param gamma the RBF inverse width; the lengthscale {@code l} is {@code gamma = 1 / (2 l^2)}
     * @param random the source of the random projection
     */
    public RandomFourierFeatures(int inputs, int features, double gamma, SplittableRandom random) {
        this.w = new double[inputs][features];
        this.b = new double[features];
        double sd = Math.sqrt(2 * gamma);
        float[] wf = new float[inputs * features];
        float[] bf = new float[features];
        for (int i = 0; i < inputs; i++) {
            for (int j = 0; j < features; j++) {
                w[i][j] = sd * random.nextGaussian();
                wf[i * features + j] = (float) w[i][j];
            }
        }
        for (int j = 0; j < features; j++) {
            b[j] = random.nextDouble() * 2 * Math.PI;
            bf[j] = (float) b[j];
        }
        this.scale = Math.sqrt(2.0 / features);
        this.wTensor = Tensor.of(Shape.of(inputs, features), wf);
        this.bTensor = Tensor.of(Shape.of(features), bf);
    }

    /**
     * A layer spec drawing the features for the input width it is built with.
     *
     * @param features the number of features
     * @param gamma the RBF inverse width
     * @return the spec
     */
    public static LayerSpec spec(int features, double gamma) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] dims = input.toArray();
                dims[dims.length - 1] = features;
                return Shape.of(dims);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new RandomFourierFeatures(input.dim(-1), features, gamma, random);
            }
        };
    }

    /**
     * Maps one point.
     *
     * @param x the point {@code [inputs]}
     * @return its features {@code [D]}
     */
    public double[] transform(double[] x) {
        double[] z = new double[b.length];
        for (int j = 0; j < z.length; j++) {
            double s = b[j];
            for (int i = 0; i < x.length; i++) {
                s += x[i] * w[i][j];
            }
            z[j] = scale * Math.cos(s);
        }
        return z;
    }

    /**
     * Maps many points.
     *
     * @param x points {@code [n][inputs]}
     * @return their features {@code [n][D]}
     */
    public double[][] transform(double[][] x) {
        double[][] z = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            z[i] = transform(x[i]);
        }
        return z;
    }

    /**
     * The kernel value the features approximate: {@code z(a) . z(b)}.
     *
     * @param a one point
     * @param c another point
     * @return the approximate RBF kernel value
     */
    public double approximateKernel(double[] a, double[] c) {
        return MlMatrices.dot(transform(a), transform(c));
    }

    /** @return the number of features */
    public int features() {
        return b.length;
    }

    @Override
    public Var forward(Var input) {
        return input.matmul(wTensor).add(bTensor).cos().mul((float) scale);
    }

    @Override
    public int outputSize(int inputSize) {
        return b.length;
    }
}
