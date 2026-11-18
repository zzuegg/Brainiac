package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.kernels.WinogradConv;
import io.github.zzuegg.brainiac.experimental.vision.Conv2dOps;
import io.github.zzuegg.brainiac.experimental.vision.SpectralConv;
import io.github.zzuegg.brainiac.experimental.vision.VisionOps;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 11 item 11.5.17: FFT convolution and spectral pooling.
 *
 * <ul>
 *   <li>{@link #timing}: milliseconds per same-padded convolution against kernel size and
 *       image size for the direct loops, im2col, {@link SpectralConv#fftConv}, the core
 *       {@code Functional.conv} kernel and {@code WinogradConv} (3x3 only), with the largest
 *       absolute error of the FFT result against the direct one.
 *   <li>{@link #pooling}: a small CNN on 16x16 {@code ShapeImages} with max, average or
 *       spectral pooling, and spectral pooling with spectrally parametrised filters.
 * </ul>
 */
public final class SpectralConvolution {

    private SpectralConvolution() {
    }

    /**
     * One timing row.
     *
     * @param size the image side
     * @param kernel the kernel side
     * @param directMs direct loops
     * @param im2colMs im2col plus a product
     * @param fftMs FFT convolution
     * @param coreMs the core convolution kernel
     * @param winogradMs Winograd F(2x2, 3x3), {@code NaN} unless the kernel is 3
     * @param fftError the largest absolute FFT error against the direct result
     */
    public record TimingRow(int size, int kernel, double directMs, double im2colMs, double fftMs, double coreMs,
            double winogradMs, double fftError) {
    }

    private static double time(Runnable r, int reps) {
        double best = Double.MAX_VALUE;
        for (int i = 0; i < reps; i++) {
            long t0 = System.nanoTime();
            r.run();
            best = Math.min(best, (System.nanoTime() - t0) / 1e6);
        }
        return best;
    }

    /**
     * Times the convolution implementations.
     *
     * @param sizes image sides
     * @param kernels odd kernel sides
     * @param channels input and output channels
     * @param reps repetitions (the minimum is kept)
     * @param seed the seed
     * @return one row per size and kernel
     */
    public static List<TimingRow> timing(int[] sizes, int[] kernels, int channels, int reps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<TimingRow> rows = new ArrayList<>();
        for (int s : sizes) {
            float[] x = new float[channels * s * s];
            for (int i = 0; i < x.length; i++) {
                x[i] = (float) r.nextGaussian();
            }
            for (int k : kernels) {
                float[] w = new float[channels * channels * k * k];
                for (int i = 0; i < w.length; i++) {
                    w[i] = (float) r.nextGaussian() / k;
                }
                int pad = k / 2;
                float[][] holder = new float[2][];
                double direct = time(() -> holder[0] = Conv2dOps.direct(x, channels, s, s, w, channels, k, 1, pad),
                        reps);
                double im2col = time(() -> Conv2dOps.im2colConv(x, channels, s, s, w, channels, k, 1, pad), reps);
                double fft = time(() -> holder[1] = SpectralConv.fftConv(x, channels, s, s, w, channels, k, pad),
                        reps);
                Tensor xt = Tensor.of(Shape.of(1, channels, s, s), x);
                Tensor wt = Tensor.of(Shape.of(channels, channels, k, k), w);
                double core = time(() -> Functional.conv(Var.of(xt), Var.of(wt), null, new int[] {1, 1},
                        new int[] {pad, pad}, new int[] {pad, pad}, new int[] {1, 1}, 1).value().toArray(), reps);
                double wino = Double.NaN;
                if (k == 3) {
                    WinogradConv wc = new WinogradConv(2, 0);
                    wino = time(() -> wc.forward(xt, wt).toArray(), reps);
                }
                double err = 0;
                for (int i = 0; i < holder[0].length; i++) {
                    err = Math.max(err, Math.abs(holder[0][i] - holder[1][i]));
                }
                rows.add(new TimingRow(s, k, direct, im2col, fft, core, wino, err));
            }
        }
        return rows;
    }

    /** Pooling choices. */
    public enum Pooling {
        /** 2x2 max pooling. */
        MAX,
        /** 2x2 average pooling. */
        AVERAGE,
        /** Spectral pooling to half the side. */
        SPECTRAL,
        /** Spectral pooling with spectrally parametrised filters. */
        SPECTRAL_PARAMETRISED
    }

    /**
     * One pooling row.
     *
     * @param pooling the pooling
     * @param accuracy test accuracy
     * @param trainMs training time
     */
    public record PoolingRow(Pooling pooling, double accuracy, double trainMs) {
    }

    private static Var pool(Var x, Pooling p) {
        int h = x.shape().dim(2);
        return switch (p) {
            case MAX -> VisionOps.maxPool2(x);
            case AVERAGE -> Functional.pool(x, Functional.PoolKind.AVG, new int[] {2, 2}, new int[] {2, 2},
                    new int[] {0, 0}, new int[] {0, 0}, new int[] {1, 1}, false, false, 1f);
            case SPECTRAL, SPECTRAL_PARAMETRISED -> SpectralConv.spectralPool(x, h / 2, h / 2);
        };
    }

    /**
     * Trains the pooling variants of one CNN.
     *
     * @param steps training steps
     * @param seed the seed
     * @return one row per pooling
     */
    public static List<PoolingRow> pooling(int steps, long seed) {
        ShapeImages gen = new ShapeImages(16, 0.25f);
        Tensor[] train = gen.generate(600, seed);
        Tensor[] test = gen.generate(300, seed + 1);
        List<PoolingRow> rows = new ArrayList<>();
        for (Pooling p : Pooling.values()) {
            SplittableRandom r = new SplittableRandom(seed + 7);
            boolean spectral = p == Pooling.SPECTRAL_PARAMETRISED;
            Var w1 = spectral ? Var.param(Tensor.normal(Shape.of(8, 9), 0, 0.47f, r)) : VisionOps.convWeight(r, 8, 1, 3);
            Var w2 = spectral ? Var.param(Tensor.normal(Shape.of(16, 72), 0, 0.17f, r))
                    : VisionOps.convWeight(r, 16, 8, 3);
            Var b1 = VisionOps.zeros(8);
            Var b2 = VisionOps.zeros(16);
            Var head = Var.param(Tensor.normal(Shape.of(16, ShapeImages.CLASSES), 0, 0.3f, r));
            List<Var> params = List.of(w1, w2, b1, b2, head);
            java.util.function.Function<Tensor, Var> net = x -> {
                Var f1 = spectral ? SpectralConv.spectralFilters(w1, 8, 1, 3) : w1;
                Var f2 = spectral ? SpectralConv.spectralFilters(w2, 16, 8, 3) : w2;
                Var h = pool(VisionOps.conv(Var.of(x), f1, b1).relu(), p);
                h = pool(VisionOps.conv(h, f2, b2).relu(), p);
                return Functional.globalPool(h, Functional.PoolKind.AVG, false).matmul(head);
            };
            Optimizer opt = Optimizer.adam(5e-3f);
            long t0 = System.nanoTime();
            for (int s = 0; s < steps; s++) {
                int from = r.nextInt(600 - 32);
                params.forEach(Var::zeroGrad);
                Var loss = Functional.crossEntropy(net.apply(train[0].slice(0, from, from + 32)),
                        train[1].slice(0, from, from + 32));
                loss.backward();
                opt.step(params);
            }
            double ms = (System.nanoTime() - t0) / 1e6;
            float[] logits = Var.noGrad(() -> net.apply(test[0])).value().toArray();
            float[] y = test[1].toArray();
            int hit = 0;
            int c = ShapeImages.CLASSES;
            for (int i = 0; i < y.length; i++) {
                int best = 0;
                for (int j = 1; j < c; j++) {
                    if (logits[i * c + j] > logits[i * c + best]) {
                        best = j;
                    }
                }
                hit += best == (int) y[i] ? 1 : 0;
            }
            rows.add(new PoolingRow(p, hit / (double) y.length, ms));
        }
        return rows;
    }
}
