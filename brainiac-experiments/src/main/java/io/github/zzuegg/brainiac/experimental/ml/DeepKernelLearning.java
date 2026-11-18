package io.github.zzuegg.brainiac.experimental.ml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Deep kernel learning (Wilson, Hu, Salakhutdinov and Xing 2016,
 * <a href="https://arxiv.org/abs/1511.02222">arXiv:1511.02222</a>): a Gaussian process whose
 * kernel acts on features {@code g(x)} of a neural network, {@code k(g(x), g(x'))}, with the
 * network and the kernel hyperparameters trained jointly by the GP's marginal likelihood.
 *
 * <p>Training runs on minibatches: the exact negative log marginal likelihood of a batch is
 * computed on the host from a Cholesky factor, its gradient with respect to the kernel matrix
 * {@code 1/2 (K^-1 - alpha alpha^T)} is pushed through the kernel to the batch features, and
 * from there through the network by core autograd ({@code features.backward(gradient)} as a
 * surrogate {@code sum(g * dL/dg)}). Predictions condition an exact GP on the features of the
 * training set (or a subset of it).
 *
 * <p>Why it matters: the network learns where the data's structure is (periodicities, relevant
 * directions); the GP head keeps calibrated uncertainty that a network with a point output
 * lacks.
 */
public final class DeepKernelLearning {

    private final Model network;
    private final GaussianProcess head;

    private DeepKernelLearning(Model network, GaussianProcess head) {
        this.network = network;
        this.head = head;
    }

    /**
     * Trains the network and the kernel jointly, then conditions the GP head.
     *
     * @param network maps inputs {@code [batch, d]} to features {@code [batch, p]}; trained in place
     * @param x inputs {@code [n][d]}
     * @param y targets {@code [n]}
     * @param kernel the initial kernel on the features (its family is kept)
     * @param noise the initial noise variance
     * @param optimizer updates the network
     * @param steps minibatch steps
     * @param batch the minibatch size
     * @param conditionOn how many training points the final GP head conditions on
     * @param seed draws the minibatches
     * @return the trained model
     */
    public static DeepKernelLearning train(Model network, double[][] x, double[] y, StationaryKernel kernel,
            double noise, Optimizer optimizer, int steps, int batch, int conditionOn, long seed) {
        Objects.requireNonNull(network, "network");
        SplittableRandom random = new SplittableRandom(seed);
        double[] theta = {Math.log(kernel.lengthscale()), Math.log(kernel.variance()), Math.log(noise)};
        double mean = 0;
        for (double v : y) {
            mean += v;
        }
        mean /= y.length;
        int b = Math.min(batch, x.length);
        for (int step = 0; step < steps; step++) {
            double[][] xb = new double[b][];
            double[] yb = new double[b];
            for (int i = 0; i < b; i++) {
                int j = random.nextInt(x.length);
                xb[i] = x[j];
                yb[i] = y[j] - mean;
            }
            Var g = network.forward(Var.of(Tensor.of(MlMatrices.toFloat(xb))), true);
            float[] gf = g.value().toArray();
            int width = g.shape().dim(1);
            double[][] phi = new double[b][width];
            for (int i = 0; i < b; i++) {
                for (int t = 0; t < width; t++) {
                    phi[i][t] = gf[i * width + t];
                }
            }
            StationaryKernel k = kernel.with(Math.exp(theta[0]), Math.exp(theta[1]));
            double sn = Math.exp(theta[2]);
            GaussianProcess gp = GaussianProcess.fit(phi, yb, k, sn);
            // hyperparameters: plain gradient ascent on the per-point log marginal likelihood
            double[] grad = gp.logMarginalGradient();
            for (int p = 0; p < 3; p++) {
                theta[p] += 0.01 * Math.clamp(grad[p] / b, -5, 5);
            }
            theta[2] = Math.max(theta[2], Math.log(1e-4));
            // features: dNLL/dphi_i = sum_j (K^-1 - alpha alpha^T)_ij dk(phi_i, phi_j)/dphi_i
            double[][] w = gp.nllKernelGradient();
            int p = phi[0].length;
            float[] dphi = new float[b * p];
            for (int i = 0; i < b; i++) {
                for (int j = 0; j < b; j++) {
                    if (i == j || w[i][j] == 0) {
                        continue;
                    }
                    double[] dk = k.gradient(phi[i], phi[j]);
                    for (int t = 0; t < p; t++) {
                        dphi[i * p + t] += (float) (w[i][j] * dk[t] / b);
                    }
                }
            }
            g.mul(Tensor.of(Shape.of(b, p), dphi)).sum().backward();
            optimizer.step(network.parameters());
        }
        int c = Math.min(conditionOn, x.length);
        double[][] xc = new double[c][];
        double[] yc = new double[c];
        for (int i = 0; i < c; i++) {
            int j = (int) ((long) i * x.length / c);
            xc[i] = x[j];
            yc[i] = y[j];
        }
        double[][] phi = features(network, xc);
        GaussianProcess head = GaussianProcess.fit(phi, yc, kernel.with(Math.exp(theta[0]), Math.exp(theta[1])),
                Math.exp(theta[2]));
        return new DeepKernelLearning(network, head);
    }

    private static double[][] features(Model network, double[][] x) {
        float[][] f = network.predict(MlMatrices.toFloat(x));
        return MlMatrices.toDouble(f);
    }

    /**
     * Predicts at new inputs.
     *
     * @param xs test inputs
     * @return the GP head's predictive mean and variances
     */
    public GaussianProcess.Prediction predict(double[][] xs) {
        return head.predict(features(network, xs));
    }

    /** @return the fitted GP head on the network features */
    public GaussianProcess head() {
        return head;
    }
}
