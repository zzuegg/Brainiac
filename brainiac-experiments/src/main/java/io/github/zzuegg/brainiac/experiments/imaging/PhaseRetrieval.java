package io.github.zzuegg.brainiac.experiments.imaging;

import io.github.zzuegg.brainiac.experimental.data.InverseProblems;
import io.github.zzuegg.brainiac.experimental.imaging.ImageMath;
import io.github.zzuegg.brainiac.experimental.imaging.PhaseRetrieval.CodedDiffraction;
import io.github.zzuegg.brainiac.experimental.imaging.PhaseRetrieval.Method;
import io.github.zzuegg.brainiac.experimental.restore.Denoisers;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Phase retrieval on piecewise-smooth scenes (wave 13, item 13.4.3): error reduction against
 * Fienup's HIO on oversampled Fourier magnitudes ({@link #fourier}); Wirtinger flow with spectral
 * initialisation, truncated WF and the denoiser-regularised prDeep against gradient descent from
 * a random start and plug-and-play HQS on coded diffraction patterns ({@link #coded}).
 */
public final class PhaseRetrieval {

    private PhaseRetrieval() {
    }

    /**
     * Oversampled Fourier magnitudes.
     *
     * @param side the image side
     * @param images the number of images
     * @param iterations iterations per run
     * @param photons photons per unit intensity for Poisson noise on the squared magnitudes, 0 for
     *     none
     * @param seed the seed
     * @return method to {success rate (twin-aligned relative error below 0.05), mean PSNR}
     */
    public static Map<String, double[]> fourier(int side, int images, int iterations, double photons, long seed) {
        List<double[]> imgs = InverseProblems.scenes(images, side, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        Map<String, double[]> out = new LinkedHashMap<>();
        Method[] methods = {Method.ER, Method.HIO};
        for (Method m : methods) {
            out.put(m.name(), new double[2]);
        }
        for (double[] x : imgs) {
            double[] mag = io.github.zzuegg.brainiac.experimental.imaging.PhaseRetrieval.fourierMagnitudes(x, side);
            if (photons > 0) {
                double scale = 0;
                for (double v : mag) {
                    scale = Math.max(scale, v * v);
                }
                double[] intens = new double[mag.length];
                for (int i = 0; i < mag.length; i++) {
                    intens[i] = mag[i] * mag[i] / scale;
                }
                intens = InverseProblems.poisson(intens, photons, r);
                for (int i = 0; i < mag.length; i++) {
                    mag[i] = Math.sqrt(intens[i] * scale);
                }
            }
            for (Method m : methods) {
                double[] est = io.github.zzuegg.brainiac.experimental.imaging.PhaseRetrieval.fourier(m, mag, side,
                    iterations, 0.9, new SplittableRandom(r.nextLong()));
                double err = io.github.zzuegg.brainiac.experimental.imaging.PhaseRetrieval.twinError(est, x, side);
                double[] row = out.get(m.name());
                row[0] += (err < 0.05 ? 1.0 : 0.0) / images;
                row[1] += ImageMath.psnr(io.github.zzuegg.brainiac.experimental.imaging.PhaseRetrieval.alignTwin(est, x),
                    x) / images;
            }
        }
        return out;
    }

    /**
     * Coded diffraction patterns.
     *
     * @param side the image side
     * @param patternCounts the numbers of patterns
     * @param images images per count
     * @param iterations gradient iterations
     * @param photons photons per unit intensity (Poisson), 0 for none
     * @param seed the seed
     * @return method to {success rate (relative error below 0.01) per pattern count..., mean PSNR per pattern
     *     count...}
     */
    public static Map<String, double[]> coded(int side, int[] patternCounts, int images, int iterations,
            double photons, long seed) {
        Method[] methods = {Method.RANDOM_START_GD, Method.WIRTINGER_FLOW, Method.TRUNCATED_WF, Method.PNP_HQS,
            Method.PR_DEEP};
        int c = patternCounts.length;
        Map<String, double[]> out = new LinkedHashMap<>();
        for (Method m : methods) {
            out.put(m.name(), new double[2 * c]);
        }
        List<double[]> imgs = InverseProblems.scenes(images, side, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        io.github.zzuegg.brainiac.experimental.imaging.PhaseRetrieval.Denoiser nlm = (x, s) -> ImageMath.toDouble(
            Denoisers.nonLocalMeans(ImageMath.toFloat(x), side, s, 3, 0.9 * s));
        for (int pc = 0; pc < c; pc++) {
            CodedDiffraction a = new CodedDiffraction(side,
                InverseProblems.codedMasks(patternCounts[pc], side * side, seed + 10 + pc));
            for (double[] x : imgs) {
                double[] y = a.intensities(x);
                if (photons > 0) {
                    double max = 0;
                    for (double v : y) {
                        max = Math.max(max, v);
                    }
                    double[] scaled = new double[y.length];
                    for (int i = 0; i < y.length; i++) {
                        scaled[i] = y[i] / max;
                    }
                    scaled = InverseProblems.poisson(scaled, photons, r);
                    for (int i = 0; i < y.length; i++) {
                        y[i] = scaled[i] * max;
                    }
                }
                for (Method m : methods) {
                    double[][] z = io.github.zzuegg.brainiac.experimental.imaging.PhaseRetrieval.coded(m, a, y,
                        iterations, nlm, 0.5, 0.05, new SplittableRandom(r.nextLong()));
                    double err = io.github.zzuegg.brainiac.experimental.imaging.PhaseRetrieval.phaseError(z, x);
                    double[] row = out.get(m.name());
                    row[pc] += (err < 0.01 ? 1.0 : 0.0) / images;
                    row[c + pc] += ImageMath.psnr(
                        io.github.zzuegg.brainiac.experimental.imaging.PhaseRetrieval.alignPhase(z, x), x) / images;
                }
            }
        }
        return out;
    }
}
