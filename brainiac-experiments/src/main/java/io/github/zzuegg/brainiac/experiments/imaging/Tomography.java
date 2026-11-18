package io.github.zzuegg.brainiac.experiments.imaging;

import io.github.zzuegg.brainiac.experimental.data.InverseProblems;
import io.github.zzuegg.brainiac.experimental.imaging.AlgebraicReconstruction;
import io.github.zzuegg.brainiac.experimental.imaging.ImageMath;
import io.github.zzuegg.brainiac.experimental.imaging.LearnedPrimalDual;
import io.github.zzuegg.brainiac.experimental.imaging.LinearOperator;
import io.github.zzuegg.brainiac.experimental.imaging.OperatorPlugAndPlay;
import io.github.zzuegg.brainiac.experimental.imaging.PrimalDual;
import io.github.zzuegg.brainiac.experimental.imaging.RadonTransform;
import io.github.zzuegg.brainiac.experimental.restore.Denoisers;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Parallel-beam CT on ellipse phantoms (wave 13, item 13.4.1): filtered back-projection (Ram-Lak
 * and Shepp-Logan), ART, SART and SART-TV against TV by diagonally preconditioned primal-dual on
 * the full operator (standing in for a FISTA-TV solver, which the sparse-recovery package has only
 * for explicit matrices) and plug-and-play ADMM with a non-local-means prior ({@link #compare});
 * and learned primal-dual against FBPConvNet (FBP plus a U-Net) at one sparse-view setting
 * ({@link #learned}).
 */
public final class Tomography {

    private Tomography() {
    }

    private static double[] noisySinogram(RadonTransform radon, double[] img, double photons, SplittableRandom r) {
        double[] s = radon.apply(img);
        return photons > 0 ? InverseProblems.poisson(s, photons, r) : s;
    }

    /**
     * Reconstruction quality against the number of angles.
     *
     * @param side the image side
     * @param angleCounts the numbers of angles
     * @param testImages test phantoms per setting
     * @param photons photons per unit line integral (Poisson noise), 0 for none
     * @param seed the seed
     * @return method to {PSNR per angle count..., mean SSIM at the last count, milliseconds per image at the
     *     last count}; NaN where a method was not run
     */
    public static Map<String, double[]> compare(int side, int[] angleCounts, int testImages, double photons,
            long seed) {
        String[] names = {"FBP (Ram-Lak)", "FBP (Shepp-Logan)", "ART", "SART", "SART-TV", "TV (preconditioned PDHG)",
            "PnP-ADMM (NLM)"};
        int a = angleCounts.length;
        Map<String, double[]> out = new LinkedHashMap<>();
        for (String n : names) {
            double[] row = new double[a + 2];
            java.util.Arrays.fill(row, Double.NaN);
            out.put(n, row);
        }
        List<double[]> test = InverseProblems.phantoms(testImages, side, seed);
        SplittableRandom noise = new SplittableRandom(seed + 1);
        for (int ai = 0; ai < a; ai++) {
            RadonTransform radon = new RadonTransform(side, angleCounts[ai]);
            double scale = 1 / Math.sqrt(radon.normSquared(15));
            LinearOperator scaled = radon.scaled(scale);
            double[][] psnr = new double[names.length][testImages];
            double[][] ssim = new double[names.length][testImages];
            double[] ms = new double[names.length];
            for (int t = 0; t < testImages; t++) {
                double[] img = test.get(t);
                double[] sino = noisySinogram(radon, img, photons, noise);
                double[][] est = new double[names.length][];
                long t0 = System.nanoTime();
                est[0] = radon.fbp(sino, RadonTransform.Filter.RAM_LAK);
                long t1 = System.nanoTime();
                est[1] = radon.fbp(sino, RadonTransform.Filter.SHEPP_LOGAN);
                long t2 = System.nanoTime();
                est[2] = AlgebraicReconstruction.reconstruct(AlgebraicReconstruction.Method.ART, radon, sino, 5, 0.5, 0);
                long t3 = System.nanoTime();
                est[3] = AlgebraicReconstruction.reconstruct(AlgebraicReconstruction.Method.SART, radon, sino, 40, 1.9,
                    0);
                long t4 = System.nanoTime();
                est[4] = AlgebraicReconstruction.reconstruct(AlgebraicReconstruction.Method.SART_TV, radon, sino, 60,
                    1.9, 0.005);
                long t5 = System.nanoTime();
                double[] ys = sino.clone();
                for (int i = 0; i < ys.length; i++) {
                    ys[i] *= scale;
                }
                est[5] = PrimalDual.linearPreconditioned(radon, sino, side, 0.0025 * angleCounts[ai], true, 300, null);
                long t6 = System.nanoTime();
                double sigmaImage = photons > 0 ? 0.01 : 0.003;
                est[6] = OperatorPlugAndPlay.admm(scaled, ys, sigmaImage,
                    (x, s) -> ImageMath.toDouble(Denoisers.nonLocalMeans(ImageMath.toFloat(x), side, s, 4, 0.9 * s)),
                    8, 0.3, est[1], 15);
                long t7 = System.nanoTime();
                long[] times = {t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t6 - t5, t7 - t6};
                for (int m = 0; m < 7; m++) {
                    psnr[m][t] = ImageMath.psnr(est[m], img);
                    ssim[m][t] = ImageMath.ssim(est[m], img, side);
                    ms[m] += times[m] / 1e6 / testImages;
                }
            }
            for (int m = 0; m < 7; m++) {
                double[] row = out.get(names[m]);
                row[ai] = mean(psnr[m]);
                if (ai == a - 1) {
                    row[a] = mean(ssim[m]);
                    row[a + 1] = ms[m];
                }
            }
        }
        return out;
    }

    /**
     * Learned reconstruction at one sparse-view setting: FBPConvNet and learned primal-dual
     * trained on phantom-sinogram pairs, against FBP and preconditioned PDHG-TV on the same test
     * phantoms.
     *
     * @param side the image side
     * @param angles the number of angles
     * @param testImages test phantoms
     * @param trainImages training phantoms
     * @param trainSteps gradient steps per network
     * @param photons photons per unit line integral, 0 for none
     * @param seed the seed
     * @return method to {PSNR, SSIM, milliseconds per image, parameters}
     */
    public static Map<String, double[]> learned(int side, int angles, int testImages, int trainImages, int trainSteps,
            double photons, long seed) {
        RadonTransform radon = new RadonTransform(side, angles);
        List<double[]> test = InverseProblems.phantoms(testImages, side, seed);
        List<double[]> train = InverseProblems.phantoms(trainImages, side, seed + 100);
        SplittableRandom tr = new SplittableRandom(seed + 2);
        List<double[]> sinos = new ArrayList<>();
        for (double[] img : train) {
            sinos.add(noisySinogram(radon, img, photons, tr));
        }
        List<double[]> testSinos = new ArrayList<>();
        SplittableRandom tn = new SplittableRandom(seed + 3);
        for (double[] img : test) {
            testSinos.add(noisySinogram(radon, img, photons, tn));
        }
        Tensor x = stack(train, side, side);
        Tensor y = stack(sinos, angles, side);
        Tensor ty = stack(testSinos, angles, side);
        Map<String, double[]> out = new LinkedHashMap<>();
        double[] fbp = new double[4];
        double[] tv = new double[4];
        for (int t = 0; t < testImages; t++) {
            long t0 = System.nanoTime();
            double[] f = radon.fbp(testSinos.get(t), RadonTransform.Filter.SHEPP_LOGAN);
            long t1 = System.nanoTime();
            double[] v = PrimalDual.linearPreconditioned(radon, testSinos.get(t), side, 0.0025 * angles, true, 300,
                null);
            long t2 = System.nanoTime();
            fbp[0] += ImageMath.psnr(f, test.get(t)) / testImages;
            fbp[1] += ImageMath.ssim(f, test.get(t), side) / testImages;
            fbp[2] += (t1 - t0) / 1e6 / testImages;
            tv[0] += ImageMath.psnr(v, test.get(t)) / testImages;
            tv[1] += ImageMath.ssim(v, test.get(t), side) / testImages;
            tv[2] += (t2 - t1) / 1e6 / testImages;
        }
        out.put("FBP (Shepp-Logan)", fbp);
        out.put("TV (preconditioned PDHG)", tv);
        LearnedPrimalDual[] nets = {
            new LearnedPrimalDual(LearnedPrimalDual.Variant.FBP_CONVNET, radon, 0, 0, 8, seed + 4),
            new LearnedPrimalDual(LearnedPrimalDual.Variant.LEARNED_PRIMAL_DUAL, radon, 4, 3, 12, seed + 5)};
        String[] names = {"FBPConvNet", "learned primal-dual"};
        for (int k = 0; k < 2; k++) {
            nets[k].fit(x, y, trainSteps, 4, 3e-3f, seed + 6);
            long s0 = System.nanoTime();
            float[] rec = nets[k].reconstruct(ty).toArray();
            double elapsed = (System.nanoTime() - s0) / 1e6 / testImages;
            double[] row = {0, 0, elapsed, nets[k].parameterCount()};
            for (int t = 0; t < testImages; t++) {
                double[] e = new double[side * side];
                for (int i = 0; i < e.length; i++) {
                    e[i] = rec[t * e.length + i];
                }
                row[0] += ImageMath.psnr(e, test.get(t)) / testImages;
                row[1] += ImageMath.ssim(e, test.get(t), side) / testImages;
            }
            out.put(names[k], row);
        }
        return out;
    }

    static Tensor stack(List<double[]> images, int h, int w) {
        float[] v = new float[images.size() * h * w];
        for (int i = 0; i < images.size(); i++) {
            double[] img = images.get(i);
            for (int k = 0; k < h * w; k++) {
                v[i * h * w + k] = (float) img[k];
            }
        }
        return Tensor.of(Shape.of(images.size(), 1, h, w), v);
    }

    static double mean(double[] v) {
        double s = 0;
        for (double x : v) {
            s += x;
        }
        return s / v.length;
    }
}
