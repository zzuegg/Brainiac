package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/**
 * Drifting models (<a href="https://arxiv.org/abs/2602.04770">arXiv:2602.04770</a>): a one-step
 * generator {@code x = G(z)} trained so that a <b>drift field</b> on its samples vanishes. The
 * drift at a generated point is attraction minus repulsion,
 * {@code V(x) = sum_j k(x, y_j) y_j / sum_j k(x, y_j) - sum_j k(x, x_j) x_j / sum_j k(x, x_j)},
 * the kernel-weighted mean of nearby data minus the kernel-weighted mean of nearby generated
 * samples (Laplace kernel {@code exp(-|x - y| / tau)}, summed over several temperatures). It
 * is zero when the generated and data distributions agree, so training pushes each sample a
 * small way along its drift, {@code |G(z) - stopgrad(G(z) + V)|^2}, and the generator settles
 * at that equilibrium. No ODE or SDE is integrated, at training or at inference: one network
 * evaluation per sample. Repulsion is what spreads samples over every mode instead of
 * collapsing onto one.
 */
public final class DriftingGenerator {
    private final Function<Var, Var> generator;
    private final List<Var> parameters;
    private final Optimizer optimizer;
    private final float[] temperatures;

    private DriftingGenerator(Function<Var, Var> generator, List<Var> parameters, Optimizer optimizer,
            float[] temperatures) {
        this.generator = Objects.requireNonNull(generator);
        this.parameters = List.copyOf(parameters);
        this.optimizer = Objects.requireNonNull(optimizer);
        this.temperatures = temperatures.clone();
    }

    /**
     * Creates the trainer.
     *
     * @param generator maps noise to samples in one step
     * @param parameters its parameters
     * @param optimizer updates them
     * @param temperatures kernel temperatures {@code tau}, in data units, such as {@code {0.05, 0.2, 1}}
     * @return the trainer
     */
    public static DriftingGenerator of(Function<Var, Var> generator, List<Var> parameters, Optimizer optimizer,
            float... temperatures) {
        if (temperatures.length == 0) {
            throw new IllegalArgumentException("at least one temperature");
        }
        return new DriftingGenerator(generator, parameters, optimizer, temperatures);
    }

    /**
     * The drift field at generated points.
     *
     * @param x generated points {@code [n, dim]}
     * @param data data points {@code [m, dim]}
     * @param temperatures kernel temperatures
     * @return {@code V(x)}, {@code [n, dim]}
     */
    public static Tensor drift(Tensor x, Tensor data, float... temperatures) {
        int n = x.shape().dim(0);
        int m = data.shape().dim(0);
        int d = x.shape().dim(1);
        float[] xv = x.toArray();
        float[] yv = data.toArray();
        float[] out = new float[n * d];
        double[] dy = new double[m];
        double[] dx = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                dy[j] = distance(xv, i, yv, j, d);
            }
            for (int j = 0; j < n; j++) {
                dx[j] = distance(xv, i, xv, j, d);
            }
            for (float tau : temperatures) {
                double[] attract = mean(dy, yv, d, tau, -1);
                double[] repel = mean(dx, xv, d, tau, i);
                for (int k = 0; k < d; k++) {
                    out[i * d + k] += (float) ((attract[k] - repel[k]) / temperatures.length);
                }
            }
        }
        return Tensor.of(Shape.of(n, d), out);
    }

    private static double distance(float[] a, int i, float[] b, int j, int d) {
        double s = 0;
        for (int k = 0; k < d; k++) {
            double v = a[i * d + k] - b[j * d + k];
            s += v * v;
        }
        return Math.sqrt(s);
    }

    /** The kernel-weighted mean of the rows of {@code points}, skipping row {@code skip}. */
    private static double[] mean(double[] dist, float[] points, int d, float tau, int skip) {
        double min = Double.MAX_VALUE;
        for (int j = 0; j < dist.length; j++) {
            if (j != skip) {
                min = Math.min(min, dist[j]);
            }
        }
        double[] acc = new double[d];
        double total = 0;
        for (int j = 0; j < dist.length; j++) {
            if (j == skip) {
                continue;
            }
            double w = Math.exp(-(dist[j] - min) / tau);
            total += w;
            for (int k = 0; k < d; k++) {
                acc[k] += w * points[j * d + k];
            }
        }
        for (int k = 0; k < d; k++) {
            acc[k] /= total;
        }
        return acc;
    }

    /**
     * One update: generate, compute the drift against a data batch, and regress the samples
     * onto their drifted (stop-gradient) positions.
     *
     * @param noise generator inputs {@code [n, dim]}
     * @param data a data batch {@code [m, dim]}
     * @return the mean drift norm before the update (0 at equilibrium)
     */
    public float step(Tensor noise, Tensor data) {
        Var x = generator.apply(Var.of(noise));
        Tensor v = drift(x.value(), data, temperatures);
        Var loss = x.sub(Var.of(x.value().add(v))).square().sum(1, false).mean();
        loss.backward();
        optimizer.step(parameters);
        return (float) Math.sqrt(v.square().sum(1, false).mean().item());
    }

    /**
     * Generates in one step.
     *
     * @param noise standard normal noise
     * @return the samples
     */
    public Tensor generate(Tensor noise) {
        return generator.apply(Var.of(noise)).value();
    }
}
