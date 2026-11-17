package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.experimental.nn.SparseAutoencoder;

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
}
