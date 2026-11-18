package io.github.zzuegg.brainiac.experiments.imaging;

import io.github.zzuegg.brainiac.experimental.data.DegradedImages;
import io.github.zzuegg.brainiac.experimental.data.InverseProblems;
import io.github.zzuegg.brainiac.experimental.imaging.BlindDeconvolution;
import io.github.zzuegg.brainiac.experimental.imaging.EdgePreservingFilters;
import io.github.zzuegg.brainiac.experimental.imaging.ImageMath;
import io.github.zzuegg.brainiac.experimental.imaging.PrimalDual;
import io.github.zzuegg.brainiac.experimental.imaging.Wavelets;
import io.github.zzuegg.brainiac.experimental.restore.Denoisers;
import io.github.zzuegg.brainiac.experimental.restore.PlugAndPlay;
import io.github.zzuegg.brainiac.experimental.signal.DictionaryLearning;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Image inverse problems on synthetic scenes (wave 13, items 13.4.4-13.4.6 and 13.4.9-13.4.12),
 * one table per technique, each against its classical or learned baselines:
 * {@link #blind} (13.4.4), {@link #variational}, {@link #impulse}, {@link #ramps} and
 * {@link #tvSpeed} (13.4.9), {@link #wavelets} and {@link #waveletSignals} (13.4.10).
 */
public final class ImageInverseProblems {

    private ImageInverseProblems() {
    }

    static double psnrFromMse(double mse) {
        return 10 * Math.log10(1 / Math.max(mse, 1e-12));
    }

    // ---------------------------------------------------------------- 13.4.4 blind deconvolution

    /**
     * Blind deconvolution of motion- and defocus-blurred scenes.
     *
     * @param side the image side
     * @param images the number of images
     * @param kernelSize the kernel side
     * @param noise the noise level
     * @param seed the seed
     * @return method to {kernel similarity, PSNR (best over shifts of up to 3 pixels), share of images with an
     *     error ratio to the oracle kernel of at most 3, median error ratio}
     */
    public static Map<String, double[]> blind(int side, int images, int kernelSize, double noise, long seed) {
        String[] names = {"oracle kernel (PnP-HQS TV)", "L0 gradient blind", "normalised sparsity blind",
            "Gaussian guess (PnP-HQS TV)", "Richardson-Lucy, Gaussian guess", "Wiener, true kernel"};
        List<double[]> imgs = InverseProblems.scenes(images, side, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        double[][] sim = new double[names.length][images];
        double[][] mse = new double[names.length][images];
        int k = kernelSize;
        double[] guess = ImageMath.toDouble(DegradedImages.gaussianKernel(k, k / 5.0));
        for (int t = 0; t < images; t++) {
            double[] x = imgs.get(t);
            double[] kernel = InverseProblems.blurKernel(k, t % 2 == 0, r);
            double[] y = InverseProblems.gaussian(ImageMath.convolve(x, side, kernel, k), noise, r);
            double[] k0 = BlindDeconvolution.estimateKernel(BlindDeconvolution.Prior.L0_GRADIENT, y, side, k, 3, 6,
                4e-3);
            double[] k1 = BlindDeconvolution.estimateKernel(BlindDeconvolution.Prior.NORMALISED_SPARSITY, y, side, k, 3,
                6, 4e-3);
            double[][] kernels = {kernel, k0, k1, guess, guess, kernel};
            double[][] est = new double[names.length][];
            for (int m = 0; m < 4; m++) {
                est[m] = BlindDeconvolution.deconvolve(y, side, kernels[m], k, noise);
            }
            est[4] = ImageMath.toDouble(PlugAndPlay.richardsonLucy(ImageMath.toFloat(y), side,
                ImageMath.toFloat(guess), k, 30));
            est[5] = ImageMath.toDouble(PlugAndPlay.wiener(io.github.zzuegg.brainiac.tensor.Tensor.of(
                io.github.zzuegg.brainiac.tensor.Shape.of(1, 1, side, side), ImageMath.toFloat(y)),
                PlugAndPlay.stackTransfers(List.of(PlugAndPlay.transfer(ImageMath.toFloat(kernel), k, side, side))),
                (float) Math.max(1e-3, 50 * noise * noise)).toArray());
            for (int m = 0; m < names.length; m++) {
                sim[m][t] = BlindDeconvolution.kernelSimilarity(kernels[m], kernel, k);
                mse[m][t] = ImageMath.shiftedMse(est[m], x, side, 3);
            }
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int m = 0; m < names.length; m++) {
            double[] ratios = new double[images];
            int ok = 0;
            double p = 0;
            double s = 0;
            for (int t = 0; t < images; t++) {
                ratios[t] = mse[m][t] / mse[0][t];
                ok += ratios[t] <= 3 ? 1 : 0;
                p += psnrFromMse(mse[m][t]) / images;
                s += sim[m][t] / images;
            }
            java.util.Arrays.sort(ratios);
            out.put(names[m], new double[] {s, p, ok / (double) images, ratios[images / 2]});
        }
        return out;
    }

    // ---------------------------------------------------------------- 13.4.9 variational and filters

    private interface Method {
        double[] run(double[] noisy, int side, double sigma);
    }

    private static Map<String, double[]> table(Map<String, Method> methods, List<double[]> clean, int side,
            java.util.function.UnaryOperator<double[]> degrade, double sigma) {
        Map<String, double[]> out = new LinkedHashMap<>();
        double[] noisyRow = new double[3];
        List<double[]> noisy = new java.util.ArrayList<>();
        for (double[] x : clean) {
            double[] y = degrade.apply(x);
            noisy.add(y);
            noisyRow[0] += ImageMath.psnr(y, x) / clean.size();
            noisyRow[1] += ImageMath.ssim(y, x, side) / clean.size();
        }
        out.put("noisy", noisyRow);
        for (Map.Entry<String, Method> e : methods.entrySet()) {
            double[] row = new double[3];
            for (int i = 0; i < clean.size(); i++) {
                long t0 = System.nanoTime();
                double[] est = e.getValue().run(noisy.get(i), side, sigma);
                row[2] += (System.nanoTime() - t0) / 1e6 / clean.size();
                row[0] += ImageMath.psnr(est, clean.get(i)) / clean.size();
                row[1] += ImageMath.ssim(est, clean.get(i), side) / clean.size();
            }
            out.put(e.getKey(), row);
        }
        return out;
    }

    static double[] nlm(double[] x, int side, double sigma) {
        return ImageMath.toDouble(Denoisers.nonLocalMeans(ImageMath.toFloat(x), side, sigma, 5, 0.8 * sigma));
    }

    /**
     * Gaussian denoising by variational methods and edge-preserving filters.
     *
     * @param side the image side
     * @param images the number of scenes
     * @param sigma the noise level
     * @param seed the seed
     * @return method to {PSNR, SSIM, milliseconds per image}
     */
    public static Map<String, double[]> variational(int side, int images, double sigma, long seed) {
        SplittableRandom r = new SplittableRandom(seed + 1);
        Map<String, Method> m = new LinkedHashMap<>();
        m.put("TV-L2 (PDHG)", (y, s, sg) -> PrimalDual.denoise(PrimalDual.Model.TV_L2, y, s, 0.8 * sg, 150));
        m.put("TGV (PDHG)", (y, s, sg) -> PrimalDual.denoise(PrimalDual.Model.TGV, y, s, 0.8 * sg, 150));
        m.put("TV gradient descent", (y, s, sg) -> ImageMath.toDouble(PlugAndPlay.totalVariation(ImageMath.toFloat(y),
            s, new float[] {1}, 1, (float) (0.8 * sg), 300, 0.01f)));
        m.put("Gaussian filter", (y, s, sg) -> ImageMath.toDouble(Denoisers.gaussianFilter(ImageMath.toFloat(y), s,
            1.0)));
        m.put("median 3x3", (y, s, sg) -> ImageMath.toDouble(Denoisers.medianFilter(ImageMath.toFloat(y), s, 1)));
        m.put("non-local means", ImageInverseProblems::nlm);
        for (EdgePreservingFilters.Filter f : EdgePreservingFilters.Filter.values()) {
            m.put(f.name().toLowerCase(java.util.Locale.ROOT), (y, s, sg) -> EdgePreservingFilters.apply(f, y, s, sg));
        }
        return table(m, InverseProblems.scenes(images, side, seed), side, x -> InverseProblems.gaussian(x, sigma, r),
            sigma);
    }

    /**
     * Salt-and-pepper denoising: TV-L1 against TV-L2 and the median filter.
     *
     * @param side the image side
     * @param images the number of scenes
     * @param fraction the corrupted share
     * @param seed the seed
     * @return method to {PSNR, SSIM, milliseconds per image}
     */
    public static Map<String, double[]> impulse(int side, int images, double fraction, long seed) {
        SplittableRandom r = new SplittableRandom(seed + 1);
        Map<String, Method> m = new LinkedHashMap<>();
        m.put("TV-L1 (PDHG)", (y, s, sg) -> PrimalDual.denoise(PrimalDual.Model.TV_L1, y, s, 0.7, 300));
        m.put("TV-L2 (PDHG)", (y, s, sg) -> PrimalDual.denoise(PrimalDual.Model.TV_L2, y, s, 0.15, 300));
        m.put("median 3x3", (y, s, sg) -> ImageMath.toDouble(Denoisers.medianFilter(ImageMath.toFloat(y), s, 1)));
        return table(m, InverseProblems.scenes(images, side, seed), side, x -> ImageMath.toDouble(
            DegradedImages.saltAndPepper(ImageMath.toFloat(x), fraction, r)), 0);
    }

    /**
     * Piecewise-linear images: random half-planes, each with its own linear ramp.
     *
     * @param n the number of images
     * @param side the side
     * @param seed the seed
     * @return the images
     */
    public static List<double[]> rampImages(int n, int side, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<double[]> out = new java.util.ArrayList<>();
        for (int i = 0; i < n; i++) {
            double[] img = new double[side * side];
            double a = 0.3 + 0.3 * r.nextDouble();
            double gx = (r.nextDouble() - 0.5) * 0.6;
            double gy = (r.nextDouble() - 0.5) * 0.6;
            for (int y = 0; y < side; y++) {
                for (int x = 0; x < side; x++) {
                    img[y * side + x] = a + gx * x / side + gy * y / side;
                }
            }
            for (int k = 0; k < 2; k++) {
                double th = r.nextDouble() * 2 * Math.PI;
                double off = (r.nextDouble() - 0.5) * side * 0.5;
                double b = 0.2 + 0.5 * r.nextDouble();
                double hx = (r.nextDouble() - 0.5) * 0.8;
                double hy = (r.nextDouble() - 0.5) * 0.8;
                for (int y = 0; y < side; y++) {
                    for (int x = 0; x < side; x++) {
                        double d = (x - side / 2.0) * Math.cos(th) + (y - side / 2.0) * Math.sin(th) - off;
                        if (d > 0) {
                            img[y * side + x] = b + hx * x / side + hy * y / side;
                        }
                    }
                }
            }
            for (int p = 0; p < img.length; p++) {
                img[p] = Math.max(0, Math.min(1, img[p]));
            }
            out.add(img);
        }
        return out;
    }

    /**
     * TV against TGV on piecewise-linear ramps (where TV staircases).
     *
     * @param side the image side
     * @param images the number of images
     * @param sigma the noise level
     * @param seed the seed
     * @return method to {PSNR, SSIM, milliseconds per image}
     */
    public static Map<String, double[]> ramps(int side, int images, double sigma, long seed) {
        SplittableRandom r = new SplittableRandom(seed + 1);
        Map<String, Method> m = new LinkedHashMap<>();
        m.put("TV-L2 (PDHG)", (y, s, sg) -> PrimalDual.denoise(PrimalDual.Model.TV_L2, y, s, 0.8 * sg, 200));
        m.put("TGV (PDHG)", (y, s, sg) -> PrimalDual.denoise(PrimalDual.Model.TGV, y, s, 0.8 * sg, 200));
        return table(m, rampImages(images, side, seed), side, x -> InverseProblems.gaussian(x, sigma, r), sigma);
    }

    /**
     * Time to the TV-L2 optimum: iterations and milliseconds for PDHG and for the gradient-descent
     * baseline (smoothed TV, {@link PlugAndPlay#totalVariation}) to come within a relative gap of
     * the optimum objective (found by 3000 PDHG iterations).
     *
     * @param side the image side
     * @param sigma the noise level
     * @param gap the relative objective gap
     * @param seed the seed
     * @return {PDHG iterations, PDHG ms, gradient-descent iterations (the largest budget tried if never),
     *     gradient-descent ms, best relative gap reached by gradient descent}
     */
    public static double[] tvSpeed(int side, double sigma, double gap, long seed) {
        double[] x = InverseProblems.scenes(1, side, seed).get(0);
        double[] y = InverseProblems.gaussian(x, sigma, new SplittableRandom(seed + 1));
        double lambda = 0.8 * sigma;
        PrimalDual.DataTerm g = PrimalDual.l2(y);
        double opt = PrimalDual.objective(g, PrimalDual.tv(g, y, side, lambda, 3000), side, lambda);
        double[] out = new double[5];
        int[] budgets = {10, 20, 40, 80, 160, 320, 640, 1280, 2560};
        out[0] = Double.NaN;
        for (int b : budgets) {
            long t0 = System.nanoTime();
            double[] e = PrimalDual.tv(g, y, side, lambda, b);
            double ms = (System.nanoTime() - t0) / 1e6;
            if (PrimalDual.objective(g, e, side, lambda) <= opt * (1 + gap)) {
                out[0] = b;
                out[1] = ms;
                break;
            }
        }
        out[4] = Double.MAX_VALUE;
        for (int b : budgets) {
            long t0 = System.nanoTime();
            double[] e = ImageMath.toDouble(PlugAndPlay.totalVariation(ImageMath.toFloat(y), side, new float[] {1}, 1,
                (float) lambda, b, 0.01f));
            double ms = (System.nanoTime() - t0) / 1e6;
            double rel = PrimalDual.objective(g, e, side, lambda) / opt - 1;
            out[4] = Math.min(out[4], rel);
            out[2] = b;
            out[3] = ms;
            if (rel <= gap) {
                break;
            }
        }
        return out;
    }

    // ---------------------------------------------------------------- 13.4.10 wavelets

    /**
     * Wavelet shrinkage on images against Gaussian filtering, non-local means and overcomplete-DCT
     * patch sparse coding.
     *
     * @param side the image side
     * @param images the number of scenes
     * @param sigmas the noise levels
     * @param seed the seed
     * @return method to {PSNR per noise level}
     */
    public static Map<String, double[]> wavelets(int side, int images, double[] sigmas, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        List<double[]> clean = InverseProblems.scenes(images, side, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        double[][] dct = DictionaryLearning.overcompleteDct(8, 11);
        for (int si = 0; si < sigmas.length; si++) {
            double sigma = sigmas[si];
            Map<String, java.util.function.UnaryOperator<double[]>> m = new LinkedHashMap<>();
            m.put("noisy", y -> y);
            for (Wavelets.Rule rule : Wavelets.Rule.values()) {
                m.put(rule.name() + " (DB4, soft)", y -> Wavelets.denoise2d(y, side, Wavelets.Wavelet.DB4, 4, rule, true,
                    Double.NaN, 1));
            }
            m.put("VISU_SHRINK hard (DB4)", y -> Wavelets.denoise2d(y, side, Wavelets.Wavelet.DB4, 4,
                Wavelets.Rule.VISU_SHRINK, false, Double.NaN, 1));
            m.put("BAYES_SHRINK (Haar)", y -> Wavelets.denoise2d(y, side, Wavelets.Wavelet.HAAR, 4,
                Wavelets.Rule.BAYES_SHRINK, true, Double.NaN, 1));
            m.put("BAYES_SHRINK (CDF 9/7)", y -> Wavelets.denoise2d(y, side, Wavelets.Wavelet.CDF97, 4,
                Wavelets.Rule.BAYES_SHRINK, true, Double.NaN, 1));
            m.put("BAYES_SHRINK + cycle spinning 4x4", y -> Wavelets.cycleSpin(y, side, 4, z -> Wavelets.denoise2d(z,
                side, Wavelets.Wavelet.DB4, 4, Wavelets.Rule.BAYES_SHRINK, true, Double.NaN, 1)));
            m.put("SURE_SHRINK + cycle spinning 4x4", y -> Wavelets.cycleSpin(y, side, 4, z -> Wavelets.denoise2d(z,
                side, Wavelets.Wavelet.DB4, 4, Wavelets.Rule.SURE_SHRINK, true, Double.NaN, 1)));
            m.put("Gaussian filter", y -> ImageMath.toDouble(Denoisers.gaussianFilter(ImageMath.toFloat(y), side,
                sigma > 0.1 ? 1.5 : 1.0)));
            m.put("non-local means", y -> nlm(y, side, sigma));
            m.put("overcomplete DCT (OMP)", y -> {
                double[][] img = new double[side][side];
                for (int i = 0; i < side; i++) {
                    System.arraycopy(y, i * side, img[i], 0, side);
                }
                double[][] d = DictionaryLearning.denoise(img, dct, 8, sigma, 10);
                double[] flat = new double[side * side];
                for (int i = 0; i < side; i++) {
                    System.arraycopy(d[i], 0, flat, i * side, side);
                }
                return flat;
            });
            for (double[] x : clean) {
                double[] y = InverseProblems.gaussian(x, sigma, r);
                for (Map.Entry<String, java.util.function.UnaryOperator<double[]>> e : m.entrySet()) {
                    out.computeIfAbsent(e.getKey(), k -> new double[sigmas.length])[si] += ImageMath.psnr(
                        e.getValue().apply(y), x) / images;
                }
            }
        }
        return out;
    }

    /**
     * Wavelet shrinkage on the Donoho-Johnstone test signals.
     *
     * @param n the signal length
     * @param sigma the noise level (signals have unit standard deviation)
     * @param trials noise draws per signal
     * @param seed the seed
     * @return rule to {mean squared error on blocks, bumps, heavisine, doppler}, with the per-subband
     *     oracle threshold as a row
     */
    public static Map<String, double[]> waveletSignals(int n, double sigma, int trials, long seed) {
        String[] signals = {"blocks", "bumps", "heavisine", "doppler"};
        Map<String, double[]> out = new LinkedHashMap<>();
        SplittableRandom r = new SplittableRandom(seed);
        int levels = 6;
        for (int s = 0; s < signals.length; s++) {
            double[] f = InverseProblems.testSignal(signals[s], n);
            for (int t = 0; t < trials; t++) {
                double[] y = InverseProblems.gaussian(f, sigma, r);
                Map<String, double[]> est = new LinkedHashMap<>();
                est.put("noisy", y);
                for (Wavelets.Rule rule : Wavelets.Rule.values()) {
                    est.put(rule.name(), Wavelets.denoise1d(y, Wavelets.Wavelet.DB4, levels, rule, true, Double.NaN, 1));
                }
                est.put("oracle threshold", Wavelets.oracle1d(y, f, Wavelets.Wavelet.DB4, levels));
                for (Map.Entry<String, double[]> e : est.entrySet()) {
                    out.computeIfAbsent(e.getKey(), k -> new double[signals.length])[s] += ImageMath.mse(e.getValue(),
                        f) / trials;
                }
            }
        }
        return out;
    }
}
