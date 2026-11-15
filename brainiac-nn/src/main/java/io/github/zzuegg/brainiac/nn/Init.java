package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/** How a layer's weights are filled before training. */
@FunctionalInterface
public interface Init {

    /**
     * Glorot/Xavier uniform: {@code U(-a, a)} with {@code a = sqrt(6 / (fanIn + fanOut))}.
     * Good for tanh and sigmoid.
     */
    Init XAVIER = (shape, fanIn, fanOut, random) -> {
        float a = (float) Math.sqrt(6.0 / (fanIn + fanOut));
        return Tensor.uniform(shape, -a, a, random);
    };

    /** He/Kaiming normal: {@code N(0, sqrt(2 / fanIn))}. Good for ReLU. */
    Init HE = (shape, fanIn, fanOut, random) ->
            Tensor.normal(shape, 0, (float) Math.sqrt(2.0 / fanIn), random);

    /** LeCun normal: {@code N(0, sqrt(1 / fanIn))}. Good for SELU. */
    Init LECUN = (shape, fanIn, fanOut, random) ->
            Tensor.normal(shape, 0, (float) Math.sqrt(1.0 / fanIn), random);

    /**
     * A random orthogonal matrix (rows or columns orthonormal, whichever fits), the usual
     * choice for recurrent weights because it keeps gradients from exploding or vanishing.
     * Tensors with more than two dimensions are treated as {@code [dim0, rest]}.
     */
    Init ORTHOGONAL = (shape, fanIn, fanOut, random) -> {
        int rows = shape.dim(0);
        int cols = Math.toIntExact(shape.size() / rows);
        boolean transpose = rows < cols;
        int r = transpose ? cols : rows;
        int c = transpose ? rows : cols;
        double[][] q = new double[c][r];   // c orthonormal vectors of length r
        for (int j = 0; j < c; j++) {
            for (int i = 0; i < r; i++) {
                q[j][i] = random.nextGaussian();
            }
            for (int pass = 0; pass < 2; pass++) {
                for (int k = 0; k < j; k++) {
                    double dot = 0;
                    for (int i = 0; i < r; i++) {
                        dot += q[j][i] * q[k][i];
                    }
                    for (int i = 0; i < r; i++) {
                        q[j][i] -= dot * q[k][i];
                    }
                }
            }
            double norm = 0;
            for (int i = 0; i < r; i++) {
                norm += q[j][i] * q[j][i];
            }
            norm = Math.sqrt(norm);
            for (int i = 0; i < r; i++) {
                q[j][i] /= norm;
            }
        }
        float[] v = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                v[i * cols + j] = (float) (transpose ? q[i][j] : q[j][i]);
            }
        }
        return Tensor.of(shape, v);
    };

    /** All zeros. */
    Init ZEROS = (shape, fanIn, fanOut, random) -> Tensor.zeros(shape);

    /**
     * Creates initial values.
     *
     * @param shape the shape to fill
     * @param fanIn the number of inputs per unit
     * @param fanOut the number of units
     * @param random the random source
     * @return the values, on the default device
     */
    Tensor create(Shape shape, int fanIn, int fanOut, RandomGenerator random);
}
