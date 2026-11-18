package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;

/**
 * A frozen double-precision copy of a tanh {@link SimpleMlp}, for fast evaluation of the
 * network and its input gradient inside integrators and rollouts, where building an autograd
 * graph for every one of thousands of single-state calls would dominate the run time.
 */
final class MlpSnapshot {

    private final double[][][] w;
    private final double[][] b;

    MlpSnapshot(SimpleMlp net) {
        int layers = net.depth();
        w = new double[layers][][];
        b = new double[layers][];
        int[] sizes = net.sizes();
        for (int l = 0; l < layers; l++) {
            float[] wf = net.weight(l).value().toArray();
            float[] bf = net.bias(l).value().toArray();
            int in = sizes[l];
            int out = sizes[l + 1];
            w[l] = new double[in][out];
            b[l] = new double[out];
            for (int i = 0; i < in; i++) {
                for (int j = 0; j < out; j++) {
                    w[l][i][j] = wf[i * out + j];
                }
            }
            for (int j = 0; j < out; j++) {
                b[l][j] = bf[j];
            }
        }
    }

    double[] forward(double[] x) {
        double[] h = x;
        for (int l = 0; l < w.length; l++) {
            double[] z = b[l].clone();
            for (int i = 0; i < h.length; i++) {
                double hi = h[i];
                double[] row = w[l][i];
                for (int j = 0; j < z.length; j++) {
                    z[j] += hi * row[j];
                }
            }
            if (l < w.length - 1) {
                for (int j = 0; j < z.length; j++) {
                    z[j] = Math.tanh(z[j]);
                }
            }
            h = z;
        }
        return h;
    }

    /** The gradient of output {@code k} with respect to the input. */
    double[] inputGradient(double[] x, int k) {
        int layers = w.length;
        double[][] acts = new double[layers][];
        double[] h = x;
        for (int l = 0; l < layers; l++) {
            acts[l] = h;
            double[] z = b[l].clone();
            for (int i = 0; i < h.length; i++) {
                double hi = h[i];
                double[] row = w[l][i];
                for (int j = 0; j < z.length; j++) {
                    z[j] += hi * row[j];
                }
            }
            if (l < layers - 1) {
                for (int j = 0; j < z.length; j++) {
                    z[j] = Math.tanh(z[j]);
                }
            }
            h = z;
        }
        double[] g = new double[h.length];
        g[k] = 1;
        for (int l = layers - 1; l >= 0; l--) {
            if (l < layers - 1) {
                // h was tanh(z): dz = dh * (1 - h^2), with h = acts[l + 1]
                double[] a = acts[l + 1];
                for (int j = 0; j < g.length; j++) {
                    g[j] *= 1 - a[j] * a[j];
                }
            }
            double[] prev = new double[acts[l].length];
            for (int i = 0; i < prev.length; i++) {
                double s = 0;
                double[] row = w[l][i];
                for (int j = 0; j < g.length; j++) {
                    s += row[j] * g[j];
                }
                prev[i] = s;
            }
            g = prev;
        }
        return g;
    }
}
