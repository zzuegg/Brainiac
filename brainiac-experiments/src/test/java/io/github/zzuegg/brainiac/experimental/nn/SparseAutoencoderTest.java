package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class SparseAutoencoderTest {

    static final int DIM = 12;
    static final int TRUE = 8;

    /** Random unit directions; each sample is a sparse positive mix of one or two of them. */
    static float[][] directions(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] dirs = new float[TRUE][DIM];
        for (float[] d : dirs) {
            double n = 0;
            for (int i = 0; i < DIM; i++) {
                d[i] = (float) r.nextGaussian();
                n += d[i] * d[i];
            }
            for (int i = 0; i < DIM; i++) {
                d[i] /= (float) Math.sqrt(n);
            }
        }
        return dirs;
    }

    static float[][] data(float[][] dirs, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][DIM];
        for (int s = 0; s < n; s++) {
            int active = 1 + r.nextInt(2);
            for (int a = 0; a < active; a++) {
                float[] d = dirs[r.nextInt(TRUE)];
                float m = (float) r.nextDouble(1, 2);
                for (int i = 0; i < DIM; i++) {
                    x[s][i] += m * d[i];
                }
            }
        }
        return x;
    }

    static int recovered(SparseAutoencoder sae, float[][] dirs) {
        int found = 0;
        for (float[] d : dirs) {
            double best = 0;
            for (int f = 0; f < sae.features(); f++) {
                float[] u = sae.direction(f);
                double dot = 0;
                double n = 0;
                for (int i = 0; i < DIM; i++) {
                    dot += u[i] * d[i];
                    n += u[i] * u[i];
                }
                best = Math.max(best, dot / Math.sqrt(n + 1e-12));
            }
            found += best > 0.9 ? 1 : 0;
        }
        return found;
    }

    @Test
    void topKRecoversTheTrueFeatures() {
        float[][] dirs = directions(1);
        float[][] x = data(dirs, 2000, 2);
        SparseAutoencoder sae = SparseAutoencoder.topK(DIM, 16, 2, 3);
        List<Float> losses = sae.train(x, 1500, 64, 0.01f, 4);
        assertTrue(losses.getLast() < 0.2 * losses.getFirst());
        int found = recovered(sae, dirs);
        double ev = sae.explainedVariance(x);
        assertTrue(found >= 7, "recovered " + found + "/8");
        assertTrue(ev > 0.9, "explained variance " + ev);
        // the feature browser: a live feature's top examples all fire it
        List<SparseAutoencoder.FeatureStats> stats = sae.stats(x);
        SparseAutoencoder.FeatureStats busiest = stats.stream()
                .max(java.util.Comparator.comparingDouble(SparseAutoencoder.FeatureStats::frequency)).orElseThrow();
        int[] top = sae.topExamples(busiest.feature(), x, 5);
        float[][] codes = sae.codes(x);
        for (int i : top) {
            assertTrue(codes[i][busiest.feature()] > 0);
        }
        assertEquals(16, stats.size());
    }

    @Test
    void l1VariantLearnsSparseCodes() {
        float[][] dirs = directions(5);
        float[][] x = data(dirs, 2000, 6);
        SparseAutoencoder sae = SparseAutoencoder.l1(DIM, 16, 0.05f, 7);
        sae.train(x, 1500, 64, 0.01f, 8);
        double meanActive = 0;
        for (float[] z : sae.codes(x)) {
            for (float v : z) {
                meanActive += v > 1e-3 ? 1 : 0;
            }
        }
        meanActive /= x.length;
        assertTrue(meanActive < 5, "active features per sample " + meanActive);
        assertTrue(recovered(sae, dirs) >= 6, "recovered " + recovered(sae, dirs));
        assertTrue(sae.explainedVariance(x) > 0.85);
    }
}
