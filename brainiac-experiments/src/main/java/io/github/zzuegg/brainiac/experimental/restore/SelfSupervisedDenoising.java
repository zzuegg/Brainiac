package io.github.zzuegg.brainiac.experimental.restore;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.random.RandomGenerator;

/**
 * Denoising without clean targets. If noise is zero-mean and independent, a network trained to
 * map a noisy image to an independent noisy copy of the same scene has the clean-target
 * minimiser; the methods differ in where that second copy comes from:
 *
 * <ul>
 *   <li><b>Noise2Noise</b> (Lehtinen et al., ICML 2018): a second noisy capture,
 *       {@link #noise2NoiseLoss};</li>
 *   <li><b>Noise2Void</b> (Krull et al., CVPR 2019): a blind spot. Random pixels are replaced
 *       by a random neighbour ({@link #blindSpot}) and the network predicts their original
 *       noisy values from the context only ({@link #blindSpotLoss}), which cannot predict
 *       independent noise;</li>
 *   <li><b>Neighbor2Neighbor</b> (Huang et al., CVPR 2021): two half-resolution sub-images
 *       built by picking two different adjacent pixels of every 2x2 cell
 *       ({@link #neighborMasks}), and a regulariser for their one-pixel offset
 *       ({@link #neighbor2NeighborLoss});</li>
 *   <li><b>Zero-shot Noise2Noise</b> (Mansour and Heckel, CVPR 2023): a tiny network trained on
 *       the single test image's two 2x2 diagonal-mean sub-images, with a residual and a
 *       consistency loss ({@link #zeroShotN2N});</li>
 *   <li><b>Median2Median</b> (2025): for spatially correlated noise. The image is cut into 3x3
 *       blocks; for each of the nine positions every block offers five first-order candidates
 *       (the pixel and the means of its 0, 45, 90 and 135 degree neighbour pairs); the three
 *       candidates nearest the median are kept, since structured-noise outliers sit far from
 *       it, and two of them are drawn at random in random order to form a sub-image pair for
 *       Noise2Noise training ({@link #median2Median}). Simplifications here: one network is
 *       shared by the nine positions (the paper trains nine) and the consistency term is
 *       left out.</li>
 * </ul>
 * Papers: <a href="https://arxiv.org/abs/1803.04189">arXiv:1803.04189</a>,
 * <a href="https://arxiv.org/abs/1811.10980">arXiv:1811.10980</a>,
 * <a href="https://arxiv.org/abs/2101.02824">arXiv:2101.02824</a>,
 * <a href="https://arxiv.org/abs/2303.11253">arXiv:2303.11253</a>,
 * <a href="https://arxiv.org/abs/2510.01666">arXiv:2510.01666</a>.
 */
public final class SelfSupervisedDenoising {

    private SelfSupervisedDenoising() {
    }

    /**
     * Noise2Noise: mean squared error between the prediction from one noisy copy and another.
     *
     * @param net the denoiser
     * @param noisyA the inputs {@code [n, 1, h, w]}
     * @param noisyB independent noisy copies of the same scenes
     * @return the loss
     */
    public static Var noise2NoiseLoss(Function<Var, Var> net, Tensor noisyA, Tensor noisyB) {
        return net.apply(Var.of(noisyA)).sub(noisyB).square().mean();
    }

    /**
     * A blind-spot batch.
     *
     * @param input the network input, masked pixels replaced by a neighbour
     * @param mask 1 at the masked pixels
     */
    public record BlindSpot(Tensor input, Tensor mask) {
    }

    /**
     * Noise2Void masking: each pixel is masked with probability {@code fraction} and replaced
     * by a uniformly drawn other pixel of its 5x5 neighbourhood.
     *
     * @param noisy {@code [n, 1, h, w]}
     * @param fraction the masked fraction (the paper uses about 0.2 to 2 percent)
     * @param random the source
     * @return the masked batch
     */
    public static BlindSpot blindSpot(Tensor noisy, double fraction, RandomGenerator random) {
        int n = noisy.shape().dim(0);
        int h = noisy.shape().dim(2);
        int w = noisy.shape().dim(3);
        float[] x = noisy.toArray();
        float[] in = x.clone();
        float[] mask = new float[x.length];
        for (int i = 0; i < n; i++) {
            for (int y = 0; y < h; y++) {
                for (int xx = 0; xx < w; xx++) {
                    if (random.nextDouble() >= fraction) {
                        continue;
                    }
                    int dy;
                    int dx;
                    do {
                        dy = random.nextInt(5) - 2;
                        dx = random.nextInt(5) - 2;
                    } while (dy == 0 && dx == 0);
                    int p = i * h * w + y * w + xx;
                    in[p] = x[i * h * w + Denoisers.reflect(y + dy, h) * w + Denoisers.reflect(xx + dx, w)];
                    mask[p] = 1;
                }
            }
        }
        return new BlindSpot(Tensor.of(noisy.shape(), in), Tensor.of(noisy.shape(), mask));
    }

    /**
     * The Noise2Void loss: squared error at the masked pixels only.
     *
     * @param net the denoiser
     * @param noisy the noisy batch (also the target)
     * @param fraction the masked fraction
     * @param random the source
     * @return the loss
     */
    public static Var blindSpotLoss(Function<Var, Var> net, Tensor noisy, double fraction, RandomGenerator random) {
        BlindSpot b = blindSpot(noisy, fraction, random);
        float masked = Math.max(1f, b.mask().sum().item());
        return net.apply(Var.of(b.input())).sub(noisy).square().mul(b.mask()).sum().div(masked);
    }

    /**
     * Neighbor2Neighbor's random neighbour sub-sampler: two masks {@code [n, 1, h, w]} that pick,
     * in every 2x2 cell, two different adjacent (horizontally or vertically neighbouring)
     * pixels.
     *
     * @param shape the batch shape {@code [n, 1, h, w]} (even sides)
     * @param random the source
     * @return the two masks
     */
    public static Tensor[] neighborMasks(Shape shape, RandomGenerator random) {
        int n = shape.dim(0);
        int h = shape.dim(2);
        int w = shape.dim(3);
        float[] m1 = new float[n * h * w];
        float[] m2 = new float[n * h * w];
        // cell positions 0 1 / 2 3; adjacent pairs
        int[][] pairs = {{0, 1}, {1, 0}, {0, 2}, {2, 0}, {1, 3}, {3, 1}, {2, 3}, {3, 2}};
        for (int i = 0; i < n; i++) {
            for (int cy = 0; cy < h / 2; cy++) {
                for (int cx = 0; cx < w / 2; cx++) {
                    int[] p = pairs[random.nextInt(pairs.length)];
                    m1[i * h * w + (2 * cy + p[0] / 2) * w + 2 * cx + p[0] % 2] = 1;
                    m2[i * h * w + (2 * cy + p[1] / 2) * w + 2 * cx + p[1] % 2] = 1;
                }
            }
        }
        return new Tensor[] {Tensor.of(shape, m1), Tensor.of(shape, m2)};
    }

    /**
     * Applies a sub-sampling mask: the masked pixel of every 2x2 cell.
     *
     * @param x {@code [n, 1, h, w]}
     * @param mask a mask of {@link #neighborMasks}
     * @return {@code [n, 1, h / 2, w / 2]}
     */
    public static Var subsample(Var x, Tensor mask) {
        return Functional.pixelUnshuffle(x.mul(mask), 2).sum(1, true);
    }

    /**
     * The Neighbor2Neighbor loss {@code |f(g1(y)) - g2(y)|^2 + gamma |f(g1(y)) - g2(y) -
     * (g1(f(y)) - g2(f(y)))|^2}, with {@code f(y)} computed without gradient.
     *
     * @param net the denoiser
     * @param noisy {@code [n, 1, h, w]}
     * @param gamma the regulariser weight (the paper uses 1 to 2)
     * @param random the source
     * @return the loss
     */
    public static Var neighbor2NeighborLoss(Function<Var, Var> net, Tensor noisy, float gamma,
            RandomGenerator random) {
        Tensor[] m = neighborMasks(noisy.shape(), random);
        Var y = Var.of(noisy);
        Var g1 = subsample(y, m[0]);
        Var g2 = subsample(y, m[1]);
        Var full = Var.noGrad(() -> net.apply(y)).detach();
        Var diff = net.apply(g1).sub(g2);
        Var offset = subsample(full, m[0]).sub(subsample(full, m[1]));
        return diff.square().mean().add(diff.sub(offset).square().mean().mul(gamma));
    }

    // ---------------------------------------------------------------- per-image (zero-shot)

    /** The tiny residual network of the zero-shot methods: 3x3, 3x3, 1x1 convolutions. */
    static final class TinyNet {
        final RestoreOps.Conv c1;
        final RestoreOps.Conv c2;
        final RestoreOps.Conv c3;

        TinyNet(int channels, RandomGenerator random) {
            c1 = new RestoreOps.Conv(1, channels, 3, random);
            c2 = new RestoreOps.Conv(channels, channels, 3, random);
            c3 = new RestoreOps.Conv(channels, 1, 1, 1e-2f, random);
        }

        /** The denoised image: the input minus the predicted noise. */
        Var apply(Var x) {
            Var h = RestoreOps.leaky(c1.forward(x), 0.2f);
            h = RestoreOps.leaky(c2.forward(h), 0.2f);
            return x.sub(c3.forward(h));
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(c1.parameters());
            p.addAll(c2.parameters());
            p.addAll(c3.parameters());
            return p;
        }
    }

    private static Var diagonal(Var x, boolean main) {
        float[] k = main ? new float[] {0.5f, 0, 0, 0.5f} : new float[] {0, 0.5f, 0.5f, 0};
        Var weight = Var.of(Tensor.of(Shape.of(1, 1, 2, 2), k));
        return Functional.conv(x, weight, null, new int[] {2, 2}, new int[] {0, 0}, new int[] {0, 0},
                new int[] {1, 1}, 1);
    }

    /**
     * Zero-shot Noise2Noise on one image: the downsamplers {@code D1}, {@code D2} average the
     * two diagonals of every 2x2 cell; the loss is
     * {@code 1/2 (|D1(y) - f(D2(y))|^2 + |D2(y) - f(D1(y))|^2) + 1/2 (|f(D1(y)) - D1(f(y))|^2 +
     * |f(D2(y)) - D2(f(y))|^2)} with {@code f} a tiny residual network.
     *
     * @param noisy the image (even side)
     * @param side its side
     * @param channels the network width (the paper uses 48)
     * @param steps Adam steps (the paper uses 2,000)
     * @param seed the seed
     * @return the denoised image
     */
    public static float[] zeroShotN2N(float[] noisy, int side, int channels, int steps, long seed) {
        TinyNet net = new TinyNet(channels, new SplittableRandom(seed));
        Var y = Var.of(Tensor.of(Shape.of(1, 1, side, side), noisy));
        Var d1 = diagonal(y, true);
        Var d2 = diagonal(y, false);
        Optimizer opt = Optimizer.adam(1e-3f);
        for (int s = 0; s < steps; s++) {
            Var f1 = net.apply(d1);
            Var f2 = net.apply(d2);
            Var res = d1.sub(f2).square().mean().add(d2.sub(f1).square().mean()).mul(0.5f);
            Var fy = net.apply(y);
            Var cons = f1.sub(diagonal(fy, true)).square().mean().add(f2.sub(diagonal(fy, false)).square().mean())
                    .mul(0.5f);
            res.add(cons).backward();
            opt.step(net.parameters());
        }
        return Var.noGrad(() -> net.apply(y)).value().toArray();
    }

    /**
     * Median2Median's sub-image pair for one of the nine positions of every 3x3 block.
     *
     * @param padded the reflect-padded image, side a multiple of 3
     * @param side the padded side
     * @param position the position in the block, 0..8 (row-major)
     * @param random the source of the random choice and assignment
     * @return {@code {x1, x2}}, each {@code side / 3} square
     */
    public static float[][] medianPair(float[] padded, int side, int position, RandomGenerator random) {
        int s = side / 3;
        int py = position / 3;
        int px = position % 3;
        float[][] out = new float[2][s * s];
        double[] cand = new double[5];
        int[][] dirs = {{0, 1}, {1, 1}, {1, 0}, {1, -1}};
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                int y = 3 * i + py;
                int x = 3 * j + px;
                cand[0] = padded[y * side + x];
                for (int d = 0; d < 4; d++) {
                    int dy = dirs[d][0];
                    int dx = dirs[d][1];
                    cand[d + 1] = 0.5 * (padded[Denoisers.reflect(y + dy, side) * side + Denoisers.reflect(x + dx, side)]
                            + padded[Denoisers.reflect(y - dy, side) * side + Denoisers.reflect(x - dx, side)]);
                }
                Arrays.sort(cand);
                // the three values nearest the median: ranks 1, 2, 3 of 0..4
                int a = 1 + random.nextInt(3);
                int b;
                do {
                    b = 1 + random.nextInt(3);
                } while (b == a);
                out[0][i * s + j] = (float) cand[a];
                out[1][i * s + j] = (float) cand[b];
            }
        }
        return out;
    }

    /**
     * Median2Median on one image: reflect-pads to a multiple of 3, trains a shared tiny residual
     * network with the symmetric Noise2Noise loss on the nine positions' pairs (a fresh random
     * draw every step), then denoises each position's pair {@code k = 4} times, averages
     * {@code (f(x1) + f(x2)) / 2} and reassembles the image.
     *
     * @param noisy the image
     * @param side its side
     * @param channels the network width
     * @param steps Adam steps
     * @param seed the seed
     * @return the denoised image
     */
    public static float[] median2Median(float[] noisy, int side, int channels, int steps, long seed) {
        int padded = ((side + 2) / 3) * 3;
        float[] p = new float[padded * padded];
        for (int y = 0; y < padded; y++) {
            for (int x = 0; x < padded; x++) {
                p[y * padded + x] = noisy[Denoisers.reflect(y, side) * side + Denoisers.reflect(x, side)];
            }
        }
        int s = padded / 3;
        SplittableRandom r = new SplittableRandom(seed);
        TinyNet net = new TinyNet(channels, new SplittableRandom(seed + 1));
        Optimizer opt = Optimizer.adam(1e-3f);
        for (int step = 0; step < steps; step++) {
            Tensor[] pair = pairBatch(p, padded, r);
            Var x1 = Var.of(pair[0]);
            Var x2 = Var.of(pair[1]);
            Var loss = net.apply(x1).sub(x2).square().mean().add(net.apply(x2).sub(x1).square().mean()).mul(0.5f);
            loss.backward();
            opt.step(net.parameters());
        }
        float[] sum = new float[padded * padded];
        int k = 4;
        for (int rep = 0; rep < k; rep++) {
            Tensor[] pair = pairBatch(p, padded, r);
            float[] a = Var.noGrad(() -> net.apply(Var.of(pair[0]))).value().toArray();
            float[] b = Var.noGrad(() -> net.apply(Var.of(pair[1]))).value().toArray();
            for (int pos = 0; pos < 9; pos++) {
                for (int i = 0; i < s; i++) {
                    for (int j = 0; j < s; j++) {
                        int o = pos * s * s + i * s + j;
                        sum[(3 * i + pos / 3) * padded + 3 * j + pos % 3] += 0.5f * (a[o] + b[o]) / k;
                    }
                }
            }
        }
        float[] out = new float[side * side];
        for (int y = 0; y < side; y++) {
            System.arraycopy(sum, y * padded, out, y * side, side);
        }
        return out;
    }

    private static Tensor[] pairBatch(float[] padded, int side, RandomGenerator random) {
        int s = side / 3;
        float[] a = new float[9 * s * s];
        float[] b = new float[9 * s * s];
        for (int pos = 0; pos < 9; pos++) {
            float[][] pr = medianPair(padded, side, pos, random);
            boolean swap = random.nextBoolean();
            System.arraycopy(swap ? pr[1] : pr[0], 0, a, pos * s * s, s * s);
            System.arraycopy(swap ? pr[0] : pr[1], 0, b, pos * s * s, s * s);
        }
        return new Tensor[] {Tensor.of(Shape.of(9, 1, s, s), a), Tensor.of(Shape.of(9, 1, s, s), b)};
    }
}
