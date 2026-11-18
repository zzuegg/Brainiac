package io.github.zzuegg.brainiac.experiments.imaging;

import io.github.zzuegg.brainiac.experimental.data.DegradedImages;
import io.github.zzuegg.brainiac.experimental.data.InverseProblems;
import io.github.zzuegg.brainiac.experimental.imaging.BlindDeconvolution;
import io.github.zzuegg.brainiac.experimental.imaging.ImageMath;
import io.github.zzuegg.brainiac.experimental.restore.PlugAndPlay;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Image inverse problems on synthetic scenes (wave 13, items 13.4.4-13.4.6 and 13.4.9-13.4.12),
 * one table per technique, each against its classical or learned baselines:
 * {@link #blind} (13.4.4).
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
}
