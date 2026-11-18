package io.github.zzuegg.brainiac.experiments.imaging;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.InverseProblems;
import io.github.zzuegg.brainiac.experimental.imaging.CompressedSensingMri;
import io.github.zzuegg.brainiac.experimental.imaging.ImageMath;
import io.github.zzuegg.brainiac.experimental.imaging.MriOperator;
import io.github.zzuegg.brainiac.experimental.imaging.OperatorPlugAndPlay;
import io.github.zzuegg.brainiac.experimental.imaging.UnrolledMri;
import io.github.zzuegg.brainiac.experimental.nn.UNet2d;
import io.github.zzuegg.brainiac.experimental.restore.Denoisers;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Undersampled single-coil MRI on phantoms and piecewise-smooth scenes (wave 13, item 13.4.2):
 * zero filling, compressed sensing with TV and wavelet sparsity, and plug-and-play HQS with a
 * non-local-means prior ({@link #classical}); and MoDL and an end-to-end variational network
 * against a U-Net on the zero-filled image, also under a sampling mask not seen in training
 * ({@link #learned}).
 */
public final class MriReconstruction {

    private MriReconstruction() {
    }

    private static List<double[]> images(int n, int side, long seed) {
        List<double[]> p = InverseProblems.phantoms((n + 1) / 2, side, seed);
        List<double[]> s = InverseProblems.scenes(n / 2, side, seed + 7);
        List<double[]> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            out.add(i % 2 == 0 ? p.get(i / 2) : s.get(i / 2));
        }
        return out;
    }

    private static double[] measure(MriOperator op, double[] img, double noise, SplittableRandom r) {
        double[] y = op.apply(img);
        boolean[] mask = op.mask();
        int n = mask.length;
        for (int i = 0; i < n; i++) {
            if (mask[i]) {
                y[i] += noise * r.nextGaussian();
                y[n + i] += noise * r.nextGaussian();
            }
        }
        return y;
    }

    /**
     * Classical reconstructions against acceleration.
     *
     * @param side the image side
     * @param accelerations the undersampling factors
     * @param testImages the test images
     * @param noise the k-space noise level
     * @param seed the seed
     * @return method to {PSNR per acceleration..., SSIM at the last}
     */
    public static Map<String, double[]> classical(int side, int[] accelerations, int testImages, double noise,
            long seed) {
        String[] names = {"zero filled", "CS-TV", "CS-wavelet (DB4, FISTA)", "PnP-HQS (NLM)"};
        Map<String, double[]> out = new LinkedHashMap<>();
        for (String n : names) {
            out.put(n, new double[accelerations.length + 1]);
        }
        List<double[]> test = images(testImages, side, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int a = 0; a < accelerations.length; a++) {
            MriOperator op = new MriOperator(side, InverseProblems.kSpaceMask(side, accelerations[a], true, seed + a));
            for (double[] img : test) {
                double[] y = measure(op, img, noise, r);
                double[][] est = {
                    op.adjoint(y),
                    CompressedSensingMri.reconstruct(CompressedSensingMri.Prior.TV, op, y, 0.02, 300),
                    CompressedSensingMri.reconstruct(CompressedSensingMri.Prior.WAVELET, op, y, 0.003, 300),
                    OperatorPlugAndPlay.hqs(op, y, Math.max(noise, 0.01),
                        (x, s) -> ImageMath.toDouble(Denoisers.nonLocalMeans(ImageMath.toFloat(x), side, s, 4,
                            0.9 * s)), 8, 0.25, op.adjoint(y), 10)};
                for (int m = 0; m < names.length; m++) {
                    out.get(names[m])[a] += ImageMath.psnr(est[m], img) / testImages;
                    if (a == accelerations.length - 1) {
                        out.get(names[m])[a + 1] += ImageMath.ssim(est[m], img, side) / testImages;
                    }
                }
            }
        }
        return out;
    }

    /**
     * Learned reconstructions at one acceleration, on the training mask and on a new mask.
     *
     * @param side the image side
     * @param acceleration the undersampling factor
     * @param trainImages training images
     * @param testImages test images
     * @param steps gradient steps per network
     * @param noise the k-space noise level
     * @param seed the seed
     * @return method to {PSNR on the training mask, PSNR on a new mask, parameters}
     */
    public static Map<String, double[]> learned(int side, int acceleration, int trainImages, int testImages,
            int steps, double noise, long seed) {
        MriOperator train = new MriOperator(side, InverseProblems.kSpaceMask(side, acceleration, true, seed));
        MriOperator other = new MriOperator(side, InverseProblems.kSpaceMask(side, acceleration, true, seed + 99));
        List<double[]> trainImgs = images(trainImages, side, seed + 10);
        List<double[]> testImgs = images(testImages, side, seed + 20);
        SplittableRandom r = new SplittableRandom(seed + 2);
        List<double[]> trainK = new ArrayList<>();
        for (double[] img : trainImgs) {
            trainK.add(measure(train, img, noise, r));
        }
        MriOperator[] ops = {train, other};
        List<List<double[]>> testK = new ArrayList<>();
        for (MriOperator op : ops) {
            List<double[]> k = new ArrayList<>();
            for (double[] img : testImgs) {
                k.add(measure(op, img, noise, r));
            }
            testK.add(k);
        }
        Tensor x = Tomography.stack(trainImgs, side, side);
        Tensor k = UnrolledMri.stack(train, trainK);
        Map<String, double[]> out = new LinkedHashMap<>();
        double[] zf = new double[3];
        double[] tv = new double[3];
        for (int o = 0; o < 2; o++) {
            for (int t = 0; t < testImages; t++) {
                double[] y = testK.get(o).get(t);
                zf[o] += ImageMath.psnr(ops[o].adjoint(y), testImgs.get(t)) / testImages;
                tv[o] += ImageMath.psnr(CompressedSensingMri.reconstruct(CompressedSensingMri.Prior.TV, ops[o], y, 0.02,
                    300), testImgs.get(t)) / testImages;
            }
        }
        out.put("zero filled", zf);
        out.put("CS-TV", tv);
        // U-Net on the zero-filled image (residual)
        UNet2d unet = new UNet2d(1, 1, 6, 2, true, 0, new SplittableRandom(seed + 3));
        List<double[]> zfTrain = new ArrayList<>();
        for (double[] y : trainK) {
            zfTrain.add(train.adjoint(y));
        }
        Tensor zx = Tomography.stack(zfTrain, side, side);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom br = new SplittableRandom(seed + 4);
        for (int s = 0; s < steps; s++) {
            int start = br.nextInt(Math.max(1, trainImages - 3));
            Var in = Var.of(zx.slice(0, start, start + 4));
            Var loss = in.add(unet.forward(in)).sub(Var.of(x.slice(0, start, start + 4))).square().mean();
            loss.backward();
            opt.step(unet.parameters());
        }
        double[] un = new double[3];
        long unetParams = 0;
        for (Var p : unet.parameters()) {
            unetParams += p.value().size();
        }
        un[2] = unetParams;
        for (int o = 0; o < 2; o++) {
            List<double[]> z = new ArrayList<>();
            for (double[] y : testK.get(o)) {
                z.add(ops[o].adjoint(y));
            }
            Tensor in = Tomography.stack(z, side, side);
            float[] rec = Var.noGrad(() -> Var.of(in).add(unet.forward(Var.of(in)))).value().toArray();
            un[o] = meanPsnr(rec, testImgs, side);
        }
        out.put("U-Net on zero filled", un);
        UnrolledMri[] nets = {new UnrolledMri(UnrolledMri.Variant.MODL, 4, 4, 16, seed + 5),
            new UnrolledMri(UnrolledMri.Variant.VARNET, 4, 0, 8, seed + 6)};
        String[] names = {"MoDL", "VarNet"};
        for (int i = 0; i < 2; i++) {
            nets[i].fit(train, x, k, steps, 4, 3e-3f, seed + 7);
            double[] row = new double[3];
            row[2] = nets[i].parameterCount();
            for (int o = 0; o < 2; o++) {
                float[] rec = nets[i].reconstruct(ops[o], UnrolledMri.stack(ops[o], testK.get(o))).toArray();
                row[o] = meanPsnr(rec, testImgs, side);
            }
            out.put(names[i], row);
        }
        return out;
    }

    private static double meanPsnr(float[] rec, List<double[]> refs, int side) {
        double s = 0;
        for (int t = 0; t < refs.size(); t++) {
            double[] e = new double[side * side];
            for (int i = 0; i < e.length; i++) {
                e[i] = rec[t * e.length + i];
            }
            s += ImageMath.psnr(e, refs.get(t)) / refs.size();
        }
        return s;
    }
}
