package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.HierarchicalVae;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Hierarchical and importance-weighted VAEs (wave 6, item 4.12) on the 73 binary
 * {@link ToyImages}: a plain VAE (ELBO), a beta-VAE ({@code beta = 4}), an IWAE trained with
 * {@code k = 5} samples, and a two-layer ladder VAE, all through {@link HierarchicalVae} with
 * the same widths and steps. Scores: the IWAE log-likelihood estimate with many samples (nats
 * per image), the active top-layer units, and a disentanglement gap: for each generative factor
 * (class, centroid row, centroid column) the gap between the best and second-best latent's
 * absolute correlation with it, averaged (a correlation form of the mutual information gap).
 */
@SuppressWarnings("exports")
public final class HierarchicalVaeExperiment {
    private HierarchicalVaeExperiment() {
    }

    /** One model's scores. */
    public record Score(double nll, int activeUnits, double disentanglement) {
    }

    /** @return every toy image as {0, 1} pixels, {@code [73, 64]} */
    static float[][] data() {
        List<float[]> all = new ArrayList<>();
        for (int c = 0; c < ToyImages.CLASSES; c++) {
            for (float[] img : ToyImages.all(c)) {
                float[] b = new float[ToyImages.PIXELS];
                for (int p = 0; p < b.length; p++) {
                    b[p] = img[p] > 0 ? 1 : 0;
                }
                all.add(b);
            }
        }
        return all.toArray(new float[0][]);
    }

    /** {class, centroid row, centroid column} of every image. */
    private static double[][] factors(float[][] x) {
        double[][] f = new double[3][x.length];
        int i = 0;
        for (int c = 0; c < ToyImages.CLASSES; c++) {
            for (int k = 0; k < ToyImages.all(c).size(); k++, i++) {
                double r = 0;
                double col = 0;
                double mass = 0;
                for (int p = 0; p < ToyImages.PIXELS; p++) {
                    r += x[i][p] * (p / ToyImages.SIDE);
                    col += x[i][p] * (p % ToyImages.SIDE);
                    mass += x[i][p];
                }
                f[0][i] = c;
                f[1][i] = r / mass;
                f[2][i] = col / mass;
            }
        }
        return f;
    }

    /**
     * Trains one model.
     *
     * @param model the model
     * @param beta the KL weight (ELBO training)
     * @param k importance samples; above 1 trains the IWAE bound instead
     * @param iterations steps of 64 images
     * @param seed the seed
     * @return its scores
     */
    public static Score run(HierarchicalVae model, float beta, int k, int iterations, long seed) {
        float[][] x = data();
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < iterations; i++) {
            float[][] batch = new float[64][];
            for (int b = 0; b < batch.length; b++) {
                batch[b] = x[r.nextInt(x.length)];
            }
            Tensor t = Nets.matrix(batch);
            // warm up the KL weight over the first third (standard for VAEs, avoids early collapse)
            float warm = Math.min(1f, 3f * (i + 1) / iterations);
            Var loss = k > 1 ? model.iwaeLoss(t, k, r) : model.elboLoss(t, beta * warm, r);
            loss.backward();
            Nets.clip(model.parameters(), 5f);
            opt.step(model.parameters());
        }
        Tensor all = Nets.matrix(x);
        double[] ll = model.logLikelihood(all, 500, new SplittableRandom(seed + 7));
        double nll = 0;
        for (double v : ll) {
            nll -= v / ll.length;
        }
        return new Score(nll, model.activeUnits(all), gap(model.encode(all).toArray(), factors(x), x.length));
    }

    private static double gap(float[] codes, double[][] factors, int n) {
        int d = codes.length / n;
        double total = 0;
        for (double[] f : factors) {
            double best = 0;
            double second = 0;
            for (int j = 0; j < d; j++) {
                double[] z = new double[n];
                for (int i = 0; i < n; i++) {
                    z[i] = codes[i * d + j];
                }
                double c = Math.abs(correlation(z, f));
                if (c > best) {
                    second = best;
                    best = c;
                } else if (c > second) {
                    second = c;
                }
            }
            total += best - second;
        }
        return total / factors.length;
    }

    private static double correlation(double[] a, double[] b) {
        int n = a.length;
        double ma = 0;
        double mb = 0;
        for (int i = 0; i < n; i++) {
            ma += a[i] / n;
            mb += b[i] / n;
        }
        double sab = 0;
        double saa = 0;
        double sbb = 0;
        for (int i = 0; i < n; i++) {
            sab += (a[i] - ma) * (b[i] - mb);
            saa += (a[i] - ma) * (a[i] - ma);
            sbb += (b[i] - mb) * (b[i] - mb);
        }
        return saa < 1e-12 || sbb < 1e-12 ? 0 : sab / Math.sqrt(saa * sbb);
    }
}
