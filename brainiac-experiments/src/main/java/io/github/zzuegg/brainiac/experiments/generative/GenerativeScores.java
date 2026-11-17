package io.github.zzuegg.brainiac.experiments.generative;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Sample-quality scores shared by the wave-6 generative experiments: the sliced Wasserstein
 * distance between two sample sets (any dimension), and for {@link ToyImages} the mean
 * distance per pixel to the nearest real image, the share of samples that are (nearly) real
 * images and how many distinct real images the samples land on.
 */
public final class GenerativeScores {
    private GenerativeScores() {
    }

    /**
     * Quality of generated {@link ToyImages}.
     *
     * @param distance the mean squared distance per pixel to the nearest real image
     * @param realShare the share of samples within 0.1 per pixel of a real image
     * @param distinct how many different real images are the nearest neighbour of a sample
     * @param classes how many of the four classes hold at least 5% of the samples
     */
    public record ImageScore(double distance, double realShare, int distinct, int classes) {
    }

    /**
     * The sliced Wasserstein-1 distance: the mean over random unit directions of the 1-D
     * Wasserstein distance between the two projected sample sets (compared at matching
     * quantiles, so the sets may differ in size).
     *
     * @param a one sample set {@code [n][d]}
     * @param b the other {@code [m][d]}
     * @param projections how many random directions
     * @param seed the seed of the directions
     * @return the distance
     */
    public static double slicedWasserstein(float[][] a, float[][] b, int projections, long seed) {
        int d = a[0].length;
        SplittableRandom r = new SplittableRandom(seed);
        int n = Math.min(a.length, b.length);
        double total = 0;
        for (int p = 0; p < projections; p++) {
            double[] dir = new double[d];
            double norm = 0;
            for (int k = 0; k < d; k++) {
                dir[k] = r.nextGaussian();
                norm += dir[k] * dir[k];
            }
            norm = Math.sqrt(norm);
            double[] pa = project(a, dir, norm);
            double[] pb = project(b, dir, norm);
            double s = 0;
            for (int q = 0; q < n; q++) {
                double u = (q + 0.5) / n;
                s += Math.abs(pa[(int) (u * pa.length)] - pb[(int) (u * pb.length)]);
            }
            total += s / n;
        }
        return total / projections;
    }

    private static double[] project(float[][] x, double[] dir, double norm) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double s = 0;
            for (int k = 0; k < dir.length; k++) {
                s += x[i][k] * dir[k];
            }
            out[i] = s / norm;
        }
        Arrays.sort(out);
        return out;
    }

    /**
     * Scores generated {@link ToyImages}.
     *
     * @param samples images {@code [n][64]}
     * @return the score
     */
    public static ImageScore images(float[][] samples) {
        double dist = 0;
        int real = 0;
        int[] perClass = new int[ToyImages.CLASSES];
        Set<String> seen = new HashSet<>();
        for (float[] s : samples) {
            double[] n = ToyImages.nearest(s);
            double perPixel = n[1] / ToyImages.PIXELS;
            dist += perPixel;
            real += perPixel < 0.1 ? 1 : 0;
            perClass[(int) n[0]]++;
            seen.add(key(s));
        }
        int classes = 0;
        for (int c : perClass) {
            classes += c >= 0.05 * samples.length ? 1 : 0;
        }
        return new ImageScore(dist / samples.length, (double) real / samples.length, seen.size(), classes);
    }

    /** The nearest real image's identity, as its sign pattern. */
    private static String key(float[] image) {
        double best = Double.MAX_VALUE;
        float[] pick = null;
        for (int c = 0; c < ToyImages.CLASSES; c++) {
            for (float[] real : ToyImages.all(c)) {
                double d = Nets.distance2(image, real);
                if (d < best) {
                    best = d;
                    pick = real;
                }
            }
        }
        StringBuilder sb = new StringBuilder();
        for (float v : pick) {
            sb.append(v > 0 ? '1' : '0');
        }
        return sb.toString();
    }

    /**
     * The mean squared distance between two sets of paired vectors.
     *
     * @param a vectors
     * @param b vectors, the same count
     * @return the mean of {@code |a_i - b_i|^2}
     */
    public static double pairedCost(float[][] a, float[][] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += Nets.distance2(a[i], b[i]);
        }
        return s / a.length;
    }
}
