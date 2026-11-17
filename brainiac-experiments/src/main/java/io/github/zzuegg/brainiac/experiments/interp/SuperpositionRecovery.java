package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.experimental.nn.SparseAutoencoder;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Sparse dictionary learning on a toy model of superposition. There are more ground-truth
 * features than dimensions, each rarely active, packed into a small activation space as
 * random directions. A TopK sparse autoencoder trained on the activations should find the
 * original directions again, which is the premise of SAE-based interpretability. The
 * outcome counts the true features that some learned feature matches with cosine
 * similarity above 0.9.
 *
 * <p>Elhage et al. 2022, "Toy Models of Superposition",
 * <a href="https://arxiv.org/abs/2209.10652">arXiv:2209.10652</a>; Gao et al. 2024,
 * "Scaling and evaluating sparse autoencoders", <a href="https://arxiv.org/abs/2406.04093">arXiv:2406.04093</a>.
 */
public final class SuperpositionRecovery {

    /**
     * The outcome.
     *
     * @param trueFeatures the number of ground-truth features
     * @param dimension the activation width they are packed into
     * @param recovered the true features matched by a learned one (cosine above 0.9)
     * @param meanBestCosine the mean, over true features, of the best match's cosine
     * @param explainedVariance the SAE's explained variance on the activations
     */
    public record Outcome(int trueFeatures, int dimension, int recovered, double meanBestCosine,
            double explainedVariance) {
    }

    private SuperpositionRecovery() {
    }

    /**
     * Runs the experiment.
     *
     * @param trueFeatures ground-truth features
     * @param dimension activation width (smaller than {@code trueFeatures})
     * @param steps SAE training steps
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome run(int trueFeatures, int dimension, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] dirs = new double[trueFeatures][dimension];
        for (double[] d : dirs) {
            double n = 0;
            for (int i = 0; i < dimension; i++) {
                d[i] = r.nextGaussian();
                n += d[i] * d[i];
            }
            for (int i = 0; i < dimension; i++) {
                d[i] /= Math.sqrt(n);
            }
        }
        int samples = 4000;
        float[][] x = new float[samples][dimension];
        double p = 1.5 / trueFeatures;
        for (float[] row : x) {
            for (double[] d : dirs) {
                if (r.nextDouble() < p) {
                    double a = 0.5 + r.nextDouble();
                    for (int i = 0; i < dimension; i++) {
                        row[i] += (float) (a * d[i]);
                    }
                }
            }
        }
        SparseAutoencoder sae = SparseAutoencoder.topK(dimension, 2 * trueFeatures, 2, seed + 1);
        sae.train(x, steps, 128, 3e-3f, seed + 2);
        int recovered = 0;
        double mean = 0;
        for (double[] d : dirs) {
            double best = 0;
            for (int f = 0; f < sae.features(); f++) {
                float[] e = sae.direction(f);
                double dot = 0;
                double n = 0;
                for (int i = 0; i < dimension; i++) {
                    dot += d[i] * e[i];
                    n += e[i] * e[i];
                }
                best = Math.max(best, n == 0 ? 0 : dot / Math.sqrt(n));
            }
            recovered += best > 0.9 ? 1 : 0;
            mean += best / trueFeatures;
        }
        return new Outcome(trueFeatures, dimension, recovered, mean, sae.explainedVariance(x));
    }

    /**
     * The outcome of one sparsity mode on hierarchical features.
     *
     * @param mode the sparse autoencoder's sparsity mode
     * @param trueFeatures parents plus children
     * @param recovered true features matched by a decoder row with cosine above 0.9
     * @param absorption the share of samples with a parent and one of its children active in
     *     which the parent's best-matching latent stays silent (the child latent absorbed it)
     * @param dead the share of latents that never fire
     * @param l0 the mean number of active latents per sample
     * @param explainedVariance the reconstruction's explained variance
     */
    public record HierarchyOutcome(SparseAutoencoder.Sparsity mode, int trueFeatures, int recovered,
            double absorption, double dead, double l0, double explainedVariance) {
    }

    /**
     * Sparse autoencoders of every {@link SparseAutoencoder.Sparsity} mode on a toy with
     * <em>hierarchical</em> ground truth: each parent feature has children that only fire
     * together with it (like "animal" and "dog"). Plain sparse codes save a latent by letting
     * the child latent absorb the parent direction, so the parent latent goes silent exactly
     * when a child is present ("feature absorption"); Matryoshka training is meant to stop
     * that (Chanin et al. 2024, <a href="https://arxiv.org/abs/2409.14507">arXiv:2409.14507</a>;
     * Bussmann et al. 2025, <a href="https://arxiv.org/abs/2503.17547">arXiv:2503.17547</a>).
     *
     * @param parents parent features
     * @param children children per parent
     * @param dimension the activation width
     * @param steps training steps per autoencoder
     * @param seed the seed
     * @return one outcome per mode, in {@code Sparsity} order
     */
    public static List<HierarchyOutcome> hierarchical(int parents, int children, int dimension, int steps,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int total = parents * (1 + children);
        double[][] dirs = new double[total][dimension];
        for (double[] d : dirs) {
            double n = 0;
            for (int i = 0; i < dimension; i++) {
                d[i] = r.nextGaussian();
                n += d[i] * d[i];
            }
            for (int i = 0; i < dimension; i++) {
                d[i] /= Math.sqrt(n);
            }
        }
        int samples = 4000;
        float[][] x = new float[samples][dimension];
        int[][] active = new int[samples][];   // {parent, child or -1} per active parent
        for (int s = 0; s < samples; s++) {
            List<Integer> on = new ArrayList<>();
            for (int p = 0; p < parents; p++) {
                if (r.nextDouble() < 1.5 / parents) {
                    int child = r.nextDouble() < 0.6 ? r.nextInt(children) : -1;
                    add(x[s], dirs[p], 0.8 + 0.4 * r.nextDouble());
                    if (child >= 0) {
                        add(x[s], dirs[parents + p * children + child], 0.8 + 0.4 * r.nextDouble());
                    }
                    on.add(p);
                    on.add(child);
                }
            }
            active[s] = on.stream().mapToInt(Integer::intValue).toArray();
        }
        int features = 2 * total;
        List<SparseAutoencoder> saes = List.of(
                SparseAutoencoder.topK(dimension, features, 2, seed + 1),
                SparseAutoencoder.l1(dimension, features, 0.05f, seed + 1),
                SparseAutoencoder.batchTopK(dimension, features, 2, seed + 1),
                SparseAutoencoder.jumpRelu(dimension, features, 0.02f, seed + 1),
                SparseAutoencoder.matryoshka(dimension, features, 2, new int[] {parents, 2 * parents, total},
                        seed + 1));
        List<HierarchyOutcome> out = new ArrayList<>();
        for (SparseAutoencoder sae : saes) {
            sae.train(x, steps, 128, 3e-3f, seed + 2);
            float[][] codes = sae.codes(x);
            int recovered = 0;
            int[] parentLatent = new int[parents];
            for (int t = 0; t < total; t++) {
                int best = -1;
                double bestCos = -1;
                for (int f = 0; f < features; f++) {
                    double c = cosine(dirs[t], sae.direction(f));
                    if (c > bestCos) {
                        bestCos = c;
                        best = f;
                    }
                }
                recovered += bestCos > 0.9 ? 1 : 0;
                if (t < parents) {
                    parentLatent[t] = best;
                }
            }
            int events = 0;
            int absorbed = 0;
            double l0 = 0;
            for (int s = 0; s < samples; s++) {
                for (float z : codes[s]) {
                    l0 += z > 0 ? 1.0 / samples : 0;
                }
                for (int i = 0; i < active[s].length; i += 2) {
                    if (active[s][i + 1] >= 0) {
                        events++;
                        absorbed += codes[s][parentLatent[active[s][i]]] > 0 ? 0 : 1;
                    }
                }
            }
            out.add(new HierarchyOutcome(sae.sparsity(), total, recovered, absorbed / (double) Math.max(1, events),
                    sae.deadFraction(x), l0, sae.explainedVariance(x)));
        }
        return out;
    }

    private static void add(float[] row, double[] dir, double a) {
        for (int i = 0; i < row.length; i++) {
            row[i] += (float) (a * dir[i]);
        }
    }

    private static double cosine(double[] a, float[] b) {
        double dot = 0;
        double nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            nb += b[i] * b[i];
        }
        return nb == 0 ? 0 : dot / Math.sqrt(nb);
    }
}
