package io.github.zzuegg.brainiac.experiments.probabilistic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.ComponentAnalysis;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.ml.PersistentHomology;
import io.github.zzuegg.brainiac.experimental.nn.TopologicalLoss;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Topological autoencoders on nested spheres (Moor et al. 2020).
 *
 * <p>Data: four small 9-spheres in 10-D, each of radius 1 and well separated, inside a large
 * enclosing 9-sphere of radius 5. A 10-2-10 autoencoder is trained with and without
 * {@link TopologicalLoss#autoencoder} (weight {@code lambda}); PCA is the linear baseline.
 * Scores on the 2-D embedding: trustworthiness (k = 10), the leave-one-out 5-NN accuracy of the
 * five sphere labels, Moor et al.'s density divergence {@code KL_0.1}, the topological error
 * (the topological loss on a 150-point subsample) and the bottleneck distance between the
 * {@code H0} diagrams of the input and the embedding, each scaled to unit largest death.
 */
public final class TopologicalAutoencoders {

    private TopologicalAutoencoders() {
    }

    private record Data(double[][] x, int[] labels) {
    }

    private static double[] sphere(double[] centre, double radius, SplittableRandom r) {
        double[] v = new double[centre.length];
        double norm = 0;
        for (int k = 0; k < v.length; k++) {
            v[k] = r.nextGaussian();
            norm += v[k] * v[k];
        }
        norm = Math.sqrt(norm);
        for (int k = 0; k < v.length; k++) {
            v[k] = centre[k] + radius * v[k] / norm;
        }
        return v;
    }

    private static Data nestedSpheres(int perSphere, SplittableRandom r) {
        int d = 10;
        int inner = 4;
        int n = perSphere * inner + 2 * perSphere;
        double[][] x = new double[n][];
        int[] labels = new int[n];
        double[][] centres = new double[inner][d];
        for (int s = 0; s < inner; s++) {
            centres[s][s] = 2.5;
            centres[s][(s + 1) % d] = s % 2 == 0 ? 1 : -1;
        }
        int i = 0;
        for (int s = 0; s < inner; s++) {
            for (int p = 0; p < perSphere; p++, i++) {
                x[i] = sphere(centres[s], 1, r);
                labels[i] = s;
            }
        }
        for (; i < n; i++) {
            x[i] = sphere(new double[d], 5, r);
            labels[i] = inner;
        }
        return new Data(x, labels);
    }

    private static Tensor matrix(double[][] rows, int[] index) {
        int d = rows[0].length;
        float[] v = new float[index.length * d];
        for (int i = 0; i < index.length; i++) {
            for (int k = 0; k < d; k++) {
                v[i * d + k] = (float) rows[index[i]][k];
            }
        }
        return Tensor.of(Shape.of(index.length, d), v);
    }

    private static int[] range(int n) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        }
        return a;
    }

    private static double[][] rows(Tensor t) {
        int n = t.shape().dim(0);
        int d = t.shape().dim(1);
        float[] v = t.toArray();
        double[][] out = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < d; k++) {
                out[i][k] = v[i * d + k];
            }
        }
        return out;
    }

    private static double[][] autoencoder(Data data, double lambda, int steps, long seed) {
        int n = data.x().length;
        Model enc = Model.sequential().input(10).seed(seed).dense(64, Activation.RELU).dense(2, Activation.IDENTITY)
                .build();
        Model dec = Model.sequential().input(2).seed(seed + 1).dense(64, Activation.RELU)
                .dense(10, Activation.IDENTITY).build();
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        int batch = 64;
        for (int s = 0; s < steps; s++) {
            int[] idx = new int[batch];
            for (int b = 0; b < batch; b++) {
                idx[b] = r.nextInt(n);
            }
            Tensor x = matrix(data.x(), idx);
            Var z = enc.forward(Var.of(x), true);
            Var loss = dec.forward(z, true).sub(Var.of(x)).square().mean();
            if (lambda > 0) {
                loss = loss.add(TopologicalLoss.autoencoder(x, z).mul((float) lambda));
            }
            loss.backward();
            opt.step(enc.parameters());
            opt.step(dec.parameters());
        }
        return rows(enc.forward(Var.of(matrix(data.x(), range(n))), false).value());
    }

    private static List<PersistentHomology.Pair> scaledH0(double[][] x) {
        List<PersistentHomology.Pair> pairs = PersistentHomology.h0(PersistentHomology.distances(x));
        double max = pairs.getLast().death();
        return pairs.stream().map(p -> new PersistentHomology.Pair(0, 0, p.death() / max, null, null)).toList();
    }

    private static double[] scores(Data data, double[][] z, int[] sub, List<PersistentHomology.Pair> inputH0) {
        double[][] zs = new double[sub.length][];
        for (int i = 0; i < sub.length; i++) {
            zs[i] = z[sub[i]];
        }
        double[][] xs = new double[sub.length][];
        for (int i = 0; i < sub.length; i++) {
            xs[i] = data.x()[sub[i]];
        }
        double topo = TopologicalLoss.autoencoder(matrix(xs, range(sub.length)), Var.of(matrix(zs, range(sub.length))))
                .value().item();
        return new double[] {MlMetrics.trustworthiness(data.x(), z, 10), MlMetrics.knnAccuracy(z, data.labels(), 5),
            densityKl(data.x(), z, 0.1), topo, PersistentHomology.bottleneckDistance(inputH0, scaledH0(zs))};
    }

    /** KL between Gaussian-kernel densities of the max-scaled distances (Moor et al.'s KL_sigma). */
    private static double densityKl(double[][] x, double[][] z, double sigma) {
        double[] px = density(PersistentHomology.distances(x), sigma);
        double[] pz = density(PersistentHomology.distances(z), sigma);
        double kl = 0;
        for (int i = 0; i < px.length; i++) {
            kl += px[i] * Math.log(px[i] / pz[i]);
        }
        return kl;
    }

    private static double[] density(double[][] d, double sigma) {
        double max = 0;
        for (double[] row : d) {
            for (double v : row) {
                max = Math.max(max, v);
            }
        }
        double[] p = new double[d.length];
        double total = 0;
        for (int i = 0; i < d.length; i++) {
            for (int j = 0; j < d.length; j++) {
                double v = d[i][j] / max;
                p[i] += Math.exp(-v * v / sigma);
            }
            total += p[i];
        }
        for (int i = 0; i < p.length; i++) {
            p[i] /= total;
        }
        return p;
    }

    /**
     * Runs the comparison.
     *
     * @param perSphere points per inner sphere (the outer one has twice as many)
     * @param steps autoencoder steps of 64 points
     * @param lambda the topological loss weight
     * @param seed the seed
     * @return method to {@code {trustworthiness, 5-NN label accuracy, KL_0.1, topological error,
     *     H0 bottleneck}}
     */
    public static Map<String, double[]> nestedSpheres(int perSphere, int steps, double lambda, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Data data = nestedSpheres(perSphere, r);
        int[] sub = new int[Math.min(150, data.x().length)];
        int stride = data.x().length / sub.length;
        double[][] xs = new double[sub.length][];
        for (int i = 0; i < sub.length; i++) {
            sub[i] = i * stride;
            xs[i] = data.x()[sub[i]];
        }
        List<PersistentHomology.Pair> inputH0 = scaledH0(xs);
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("topological AE", scores(data, autoencoder(data, lambda, steps, seed), sub, inputH0));
        out.put("plain AE", scores(data, autoencoder(data, 0, steps, seed), sub, inputH0));
        out.put("PCA", scores(data, ComponentAnalysis.pca(data.x(), 2).transform(data.x()), sub, inputH0));
        return out;
    }
}
