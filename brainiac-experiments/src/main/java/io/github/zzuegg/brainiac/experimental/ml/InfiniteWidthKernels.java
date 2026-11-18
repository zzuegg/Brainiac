package io.github.zzuegg.brainiac.experimental.ml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.function.Function;

/**
 * The kernels of infinitely wide ReLU networks.
 *
 * <p>As the width of an MLP goes to infinity, its randomly initialised function is a Gaussian
 * process whose kernel, the NNGP, follows a layer-wise recursion: {@code Sigma^1 = sw^2 x.x'/d
 * + sb^2} and {@code Sigma^(l+1) = sw^2 E[relu(u) relu(v)] + sb^2} with the expectation in
 * closed form (the arc-cosine kernel; Cho and Saul 2009; Lee et al. 2018,
 * <a href="https://arxiv.org/abs/1711.00165">arXiv:1711.00165</a>). Gradient-descent training
 * of the same network in the NTK parameterisation follows kernel regression with the neural
 * tangent kernel {@code Theta^(l+1) = Sigma^(l+1) + Theta^l sw^2 E[relu'(u) relu'(v)]} (Jacot et
 * al. 2018, <a href="https://arxiv.org/abs/1806.07572">arXiv:1806.07572</a>; Lee et al. 2019,
 * <a href="https://arxiv.org/abs/1902.06720">arXiv:1902.06720</a>). Both are
 * {@link KernelFunction}s, so they plug into {@link KernelGaussianProcess}.
 * {@link #empiricalNtk} computes {@code J J^T} of a finite network with autograd, to see how far
 * a real network is from the limit and how much its kernel moves while it trains (feature
 * learning).
 */
public final class InfiniteWidthKernels {

    private InfiniteWidthKernels() {
    }

    /**
     * The NNGP kernel of a ReLU MLP.
     *
     * @param hiddenLayers the number of hidden ReLU layers
     * @param weightVariance {@code sw^2}, for example 2
     * @param biasVariance {@code sb^2}, for example 0.01
     * @return the kernel of the network's output
     */
    public static KernelFunction nngp(int hiddenLayers, double weightVariance, double biasVariance) {
        return (a, b) -> recursion(a, b, hiddenLayers, weightVariance, biasVariance)[0];
    }

    /**
     * The neural tangent kernel of a ReLU MLP in the NTK parameterisation.
     *
     * @param hiddenLayers the number of hidden ReLU layers
     * @param weightVariance {@code sw^2}
     * @param biasVariance {@code sb^2}
     * @return the kernel
     */
    public static KernelFunction ntk(int hiddenLayers, double weightVariance, double biasVariance) {
        return (a, b) -> recursion(a, b, hiddenLayers, weightVariance, biasVariance)[1];
    }

    /** Returns {NNGP, NTK} at the output. */
    private static double[] recursion(double[] a, double[] b, int layers, double sw, double sb) {
        int d = a.length;
        double kab = sw * MlMatrices.dot(a, b) / d + sb;
        double kaa = sw * MlMatrices.dot(a, a) / d + sb;
        double kbb = sw * MlMatrices.dot(b, b) / d + sb;
        double theta = kab;
        for (int l = 0; l < layers; l++) {
            double norm = Math.sqrt(kaa * kbb);
            double c = Math.max(-1, Math.min(1, kab / norm));
            double angle = Math.acos(c);
            double expRelu = norm / (2 * Math.PI) * (Math.sin(angle) + (Math.PI - angle) * c);
            double expDeriv = (Math.PI - angle) / (2 * Math.PI);
            double next = sw * expRelu + sb;
            theta = next + theta * sw * expDeriv;
            kab = next;
            kaa = sw * kaa / 2 + sb;
            kbb = sw * kbb / 2 + sb;
        }
        return new double[] {kab, theta};
    }

    /**
     * The empirical NTK {@code Theta(x_i, x_j) = <df(x_i)/dw, df(x_j)/dw>} of a scalar-output
     * network, from one backward pass per input.
     *
     * @param parameters the network's parameters
     * @param model maps an input batch {@code [n, d]} to outputs {@code [n, 1]}
     * @param x the inputs {@code [n][d]}
     * @return {@code [n][n]}
     */
    public static double[][] empiricalNtk(List<Var> parameters, Function<Var, Var> model, double[][] x) {
        int n = x.length;
        int d = x[0].length;
        double[][] jac = new double[n][];
        for (int i = 0; i < n; i++) {
            float[] row = new float[d];
            for (int j = 0; j < d; j++) {
                row[j] = (float) x[i][j];
            }
            for (Var p : parameters) {
                p.zeroGrad();
            }
            model.apply(Var.of(Tensor.of(Shape.of(1, d), row))).sum().backward();
            int total = 0;
            for (Var p : parameters) {
                total += (int) p.value().shape().size();
            }
            double[] g = new double[total];
            int o = 0;
            for (Var p : parameters) {
                int size = (int) p.value().shape().size();
                if (p.grad() != null) {
                    double[] v = p.grad().toDoubleArray();
                    System.arraycopy(v, 0, g, o, size);
                }
                o += size;
                p.zeroGrad();
            }
            jac[i] = g;
        }
        double[][] k = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                k[i][j] = MlMatrices.dot(jac[i], jac[j]);
                k[j][i] = k[i][j];
            }
        }
        return k;
    }
}
