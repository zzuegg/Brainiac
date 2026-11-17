package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.EquivariantGraphLayer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Rotation-invariant shape classification with EGNN (Satorras et al.,
 * <a href="https://arxiv.org/abs/2102.09844">2102.09844</a>): six-point clouds are either an
 * octahedron or a triangular prism, scaled and jittered. Training sees them only in one
 * orientation; testing rotates them at random. The EGNN, invariant by construction, keeps its
 * accuracy, while an MLP on the raw coordinates trained the same way does not.
 */
@SuppressWarnings("exports")   // experimental.* packages are exported by the lead at integration
public final class EquivariantPointClouds {
    /** Points per cloud. */
    public static final int POINTS = 6;
    private static final int FEATURES = 4;

    private final List<Layer> layers = new ArrayList<>();
    private final boolean graph;
    private final Optimizer optimizer;
    private final SplittableRandom random;

    private EquivariantPointClouds(boolean graph, float learningRate, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.graph = graph;
        if (graph) {
            layers.add(EquivariantGraphLayer.of(3, FEATURES, 16, seed + 1));
            layers.add(EquivariantGraphLayer.of(3, FEATURES, 16, seed + 2));
            layers.add(new Dense(FEATURES, 2, Activation.IDENTITY, Init.XAVIER, r));
        } else {
            layers.add(new Dense(3 * POINTS, 32, Activation.TANH, Init.XAVIER, r));
            layers.add(new Dense(32, 2, Activation.IDENTITY, Init.XAVIER, r));
        }
        this.optimizer = Optimizer.adam(learningRate);
        this.random = r;
    }

    /**
     * Creates an EGNN classifier: two equivariant layers, mean pooling of node features, a linear head.
     *
     * @param learningRate the Adam learning rate
     * @param seed the seed
     * @return the experiment
     */
    public static EquivariantPointClouds egnn(float learningRate, long seed) {
        return new EquivariantPointClouds(true, learningRate, seed);
    }

    /**
     * Creates the baseline: an MLP on the flattened coordinates.
     *
     * @param learningRate the Adam learning rate
     * @param seed the seed
     * @return the experiment
     */
    public static EquivariantPointClouds mlp(float learningRate, long seed) {
        return new EquivariantPointClouds(false, learningRate, seed);
    }

    /**
     * Draws a point cloud.
     *
     * @param label 0 for the octahedron, 1 for the triangular prism
     * @param rotate whether to apply a random rotation
     * @param random the random source
     * @return {@code [POINTS][3]}
     */
    public static double[][] cloud(int label, boolean rotate, SplittableRandom random) {
        double[][] p = new double[POINTS][];
        if (label == 0) {
            for (int i = 0; i < 3; i++) {
                p[2 * i] = new double[3];
                p[2 * i][i] = 1;
                p[2 * i + 1] = new double[3];
                p[2 * i + 1][i] = -1;
            }
        } else {
            for (int i = 0; i < 3; i++) {
                double a = 2 * Math.PI * i / 3;
                p[i] = new double[] {0.8 * Math.cos(a), 0.8 * Math.sin(a), 0.7};
                p[i + 3] = new double[] {0.8 * Math.cos(a), 0.8 * Math.sin(a), -0.7};
            }
        }
        double scale = random.nextDouble(0.8, 1.2);
        double[][] rot = rotate ? randomRotation(random) : new double[][] {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        for (double[] q : p) {
            double[] r = apply(rot, q);
            for (int j = 0; j < 3; j++) {
                q[j] = scale * r[j] + 0.05 * random.nextGaussian();
            }
        }
        return p;
    }

    /**
     * A uniformly random rotation from a random unit quaternion.
     *
     * @param random the random source
     * @return a 3x3 rotation matrix
     */
    public static double[][] randomRotation(SplittableRandom random) {
        double w = random.nextGaussian();
        double x = random.nextGaussian();
        double y = random.nextGaussian();
        double z = random.nextGaussian();
        double n = Math.sqrt(w * w + x * x + y * y + z * z);
        w /= n;
        x /= n;
        y /= n;
        z /= n;
        return new double[][] {
            {1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
            {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
            {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
    }

    /**
     * Multiplies a matrix and a vector.
     *
     * @param m the matrix
     * @param v the vector
     * @return {@code m v}
     */
    public static double[] apply(double[][] m, double[] v) {
        double[] r = new double[m.length];
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < v.length; j++) {
                r[i] += m[i][j] * v[j];
            }
        }
        return r;
    }

    private Var logits(List<double[][]> clouds) {
        int b = clouds.size();
        if (graph) {
            float[] in = new float[b * POINTS * (3 + FEATURES)];
            for (int c = 0; c < b; c++) {
                for (int i = 0; i < POINTS; i++) {
                    int o = (c * POINTS + i) * (3 + FEATURES);
                    for (int j = 0; j < 3; j++) {
                        in[o + j] = (float) clouds.get(c)[i][j];
                    }
                    in[o + 3] = 1;   // all nodes start alike
                }
            }
            Var x = Var.of(Tensor.of(Shape.of(b, POINTS, 3 + FEATURES), in));
            x = layers.get(1).forward(layers.get(0).forward(x));
            Var pooled = x.slice(2, 3, 3 + FEATURES).mean(1, false);
            return layers.get(2).forward(pooled);
        }
        float[] in = new float[b * POINTS * 3];
        for (int c = 0; c < b; c++) {
            for (int i = 0; i < POINTS; i++) {
                for (int j = 0; j < 3; j++) {
                    in[(c * POINTS + i) * 3 + j] = (float) clouds.get(c)[i][j];
                }
            }
        }
        return layers.get(1).forward(layers.get(0).forward(Var.of(Tensor.of(Shape.of(b, 3 * POINTS), in))));
    }

    /**
     * Trains on unrotated clouds.
     *
     * @param steps optimizer steps
     * @param batch the batch size
     * @return the last loss
     */
    public float train(int steps, int batch) {
        float last = Float.NaN;
        for (int s = 0; s < steps; s++) {
            List<double[][]> clouds = new ArrayList<>();
            float[] labels = new float[batch];
            for (int i = 0; i < batch; i++) {
                labels[i] = i % 2;
                clouds.add(cloud(i % 2, false, random));
            }
            Var loss = Functional.crossEntropy(logits(clouds), Tensor.of(Shape.of(batch), labels));
            last = loss.value().item();
            loss.backward();
            List<Var> params = new ArrayList<>();
            for (Layer l : layers) {
                params.addAll(l.parameters());
            }
            optimizer.step(params);
        }
        return last;
    }

    /**
     * Accuracy on fresh clouds.
     *
     * @param rotated whether test clouds are randomly rotated
     * @param n the number of clouds
     * @param seed the test seed
     * @return the share correct
     */
    public double accuracy(boolean rotated, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<double[][]> clouds = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            clouds.add(cloud(i % 2, rotated, r));
        }
        float[] p = logits(clouds).value().toArray();
        int correct = 0;
        for (int i = 0; i < n; i++) {
            int guess = p[2 * i + 1] > p[2 * i] ? 1 : 0;
            if (guess == i % 2) {
                correct++;
            }
        }
        return (double) correct / n;
    }
}
