package io.github.zzuegg.brainiac.experimental.restore;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.DegradedImages;
import io.github.zzuegg.brainiac.nn.BatchNorm;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * Supervised image denoisers and the classical reference filters.
 *
 * <ul>
 *   <li>{@link DnCnn} (Zhang et al., TIP 2017): a stack of 3x3 convolution, batch-norm and ReLU
 *       layers that predicts the <em>noise</em> (residual learning: the output is the input
 *       minus the prediction), so one blind network covers a range of noise levels; with
 *       {@code residual = false} it predicts the clean image directly (the ablation).</li>
 *   <li>{@link FfdNet} (Zhang et al., TIP 2018): works on the pixel-unshuffled image (four
 *       half-resolution sub-images, so four times fewer positions) together with a
 *       noise-level map as an extra input channel, then pixel-shuffles back; the map steers one
 *       model to any noise level, including spatially varying noise.</li>
 *   <li>Classical filters: {@link #gaussianFilter}, {@link #medianFilter} and
 *       {@link #nonLocalMeans} (Buades et al., 2005).</li>
 * </ul>
 * Papers: <a href="https://arxiv.org/abs/1608.03981">arXiv:1608.03981</a>,
 * <a href="https://arxiv.org/abs/1710.04026">arXiv:1710.04026</a>.
 */
public final class Denoisers {

    private Denoisers() {
    }

    /** DnCNN: conv-ReLU, then conv-BN-ReLU layers, then a conv to one channel. */
    public static final class DnCnn implements Layer, ImageDenoiser {
        private final List<RestoreOps.Conv> convs = new ArrayList<>();
        private final List<BatchNorm> norms = new ArrayList<>();
        private final boolean residual;
        private boolean training;

        /**
         * Creates the network.
         *
         * @param depth the number of convolutions, at least 2
         * @param features the hidden channels
         * @param residual whether it predicts the noise (DnCNN) or the clean image
         * @param random the source of initial weights
         */
        public DnCnn(int depth, int features, boolean residual, RandomGenerator random) {
            if (depth < 2) {
                throw new IllegalArgumentException("depth must be at least 2, got " + depth);
            }
            this.residual = residual;
            convs.add(new RestoreOps.Conv(1, features, 3, random));
            for (int i = 1; i < depth - 1; i++) {
                convs.add(new RestoreOps.Conv(features, features, 3, random));
                norms.add(new BatchNorm(features));
            }
            convs.add(new RestoreOps.Conv(features, 1, 3, 1e-2f, random));
        }

        /**
         * Describes a residual DnCNN for model builders on {@code [1, h, w]}.
         *
         * @param depth the convolutions
         * @param features the hidden channels
         * @return the spec
         */
        public static LayerSpec spec(int depth, int features) {
            return RestoreOps.spec(s -> s, (s, r) -> new DnCnn(depth, features, true, r));
        }

        @Override
        public Var forward(Var input) {
            return forward(input, false);
        }

        @Override
        public Var forward(Var input, boolean train) {
            Var x = convs.get(0).forward(input).relu();
            for (int i = 1; i < convs.size() - 1; i++) {
                x = norms.get(i - 1).forward(convs.get(i).forward(x), train).relu();
            }
            Var out = convs.get(convs.size() - 1).forward(x);
            return residual ? input.sub(out) : out;
        }

        /**
         * Sets the mode {@link #denoise} runs in (inference by default).
         *
         * @param train whether to use batch statistics
         */
        public void training(boolean train) {
            this.training = train;
        }

        @Override
        public Var denoise(Var noisy, float sigma) {
            return forward(noisy, training);
        }

        @Override
        public Shape outputShape(Shape input) {
            return input;
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>();
            for (int i = 0; i < convs.size(); i++) {
                p.addAll(convs.get(i).parameters());
                if (i > 0 && i - 1 < norms.size()) {
                    p.addAll(norms.get(i - 1).parameters());
                }
            }
            return p;
        }

        @Override
        public List<Var> buffers() {
            List<Var> b = new ArrayList<>();
            for (BatchNorm n : norms) {
                b.addAll(n.buffers());
            }
            return b;
        }

        @Override
        public String toString() {
            return "DnCNN(" + convs.size() + " convs" + (residual ? ", residual" : ", direct") + ")";
        }
    }

    /**
     * FFDNet: input {@code [batch, 2, h, w]} (the noisy image and its noise-level map), output
     * {@code [batch, 1, h, w]}; height and width must be even.
     */
    public static final class FfdNet implements Layer, ImageDenoiser {
        private final List<RestoreOps.Conv> convs = new ArrayList<>();
        private final List<BatchNorm> norms = new ArrayList<>();
        private final boolean residual;

        /**
         * Creates the network.
         *
         * @param depth the number of convolutions, at least 2
         * @param features the hidden channels
         * @param residual whether the shuffled output is subtracted from the input (as DnCNN)
         *     rather than taken as the clean image (as in the paper)
         * @param random the source of initial weights
         */
        public FfdNet(int depth, int features, boolean residual, RandomGenerator random) {
            if (depth < 2) {
                throw new IllegalArgumentException("depth must be at least 2, got " + depth);
            }
            this.residual = residual;
            convs.add(new RestoreOps.Conv(5, features, 3, random));
            for (int i = 1; i < depth - 1; i++) {
                convs.add(new RestoreOps.Conv(features, features, 3, random));
                norms.add(new BatchNorm(features));
            }
            convs.add(new RestoreOps.Conv(features, 4, 3, 1e-2f, random));
        }

        /**
         * Describes a residual FFDNet for model builders on {@code [2, h, w]}.
         *
         * @param depth the convolutions
         * @param features the hidden channels
         * @return the spec
         */
        public static LayerSpec spec(int depth, int features) {
            return RestoreOps.spec(s -> Shape.of(1, s.dim(1), s.dim(2)), (s, r) -> new FfdNet(depth, features, true, r));
        }

        @Override
        public Var forward(Var input) {
            return forward(input, false);
        }

        @Override
        public Var forward(Var input, boolean train) {
            Var noisy = input.slice(1, 0, 1);
            Var map = input.slice(1, 1, 2);
            Var low = Functional.pixelUnshuffle(noisy, 2);
            Var lowMap = Functional.pixelUnshuffle(map, 2).mean(1, true);
            Var x = convs.get(0).forward(Var.concat(1, low, lowMap)).relu();
            for (int i = 1; i < convs.size() - 1; i++) {
                x = norms.get(i - 1).forward(convs.get(i).forward(x), train).relu();
            }
            Var out = Functional.pixelShuffle(convs.get(convs.size() - 1).forward(x), 2);
            return residual ? noisy.sub(out) : out;
        }

        /**
         * Denoises with a per-pixel noise map.
         *
         * @param noisy {@code [batch, 1, h, w]}
         * @param sigmaMap {@code [batch, 1, h, w]}
         * @param train whether to use batch statistics
         * @return the estimates
         */
        public Var forward(Var noisy, Var sigmaMap, boolean train) {
            return forward(Var.concat(1, noisy, sigmaMap), train);
        }

        @Override
        public Var denoise(Var noisy, float sigma) {
            Tensor map = Tensor.full(noisy.shape(), sigma);
            return forward(noisy, Var.of(map), false);
        }

        @Override
        public Shape outputShape(Shape input) {
            return Shape.of(1, input.dim(1), input.dim(2));
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>();
            for (int i = 0; i < convs.size(); i++) {
                p.addAll(convs.get(i).parameters());
                if (i > 0 && i - 1 < norms.size()) {
                    p.addAll(norms.get(i - 1).parameters());
                }
            }
            return p;
        }

        @Override
        public List<Var> buffers() {
            List<Var> b = new ArrayList<>();
            for (BatchNorm n : norms) {
                b.addAll(n.buffers());
            }
            return b;
        }

        @Override
        public String toString() {
            return "FFDNet(" + convs.size() + " convs)";
        }
    }

    // ---------------------------------------------------------------- classical filters

    /**
     * Gaussian smoothing (reflective borders).
     *
     * @param x the image
     * @param side its side
     * @param sigma the kernel standard deviation in pixels
     * @return the filtered copy
     */
    public static float[] gaussianFilter(float[] x, int side, double sigma) {
        int k = 2 * (int) Math.ceil(2 * sigma) + 1;
        float[] kernel = DegradedImages.gaussianKernel(k, sigma);
        float[] out = new float[x.length];
        int c = k / 2;
        for (int y = 0; y < side; y++) {
            for (int xx = 0; xx < side; xx++) {
                double s = 0;
                for (int ky = 0; ky < k; ky++) {
                    int yy = reflect(y + ky - c, side);
                    for (int kx = 0; kx < k; kx++) {
                        s += kernel[ky * k + kx] * x[yy * side + reflect(xx + kx - c, side)];
                    }
                }
                out[y * side + xx] = (float) s;
            }
        }
        return out;
    }

    /**
     * Median filtering in a square window (reflective borders).
     *
     * @param x the image
     * @param side its side
     * @param radius the window radius (1 gives 3x3)
     * @return the filtered copy
     */
    public static float[] medianFilter(float[] x, int side, int radius) {
        int w = 2 * radius + 1;
        float[] win = new float[w * w];
        float[] out = new float[x.length];
        for (int y = 0; y < side; y++) {
            for (int xx = 0; xx < side; xx++) {
                int n = 0;
                for (int dy = -radius; dy <= radius; dy++) {
                    for (int dx = -radius; dx <= radius; dx++) {
                        win[n++] = x[reflect(y + dy, side) * side + reflect(xx + dx, side)];
                    }
                }
                Arrays.sort(win);
                out[y * side + xx] = win[win.length / 2];
            }
        }
        return out;
    }

    /**
     * Non-local means: each pixel becomes a weighted mean of the pixels in a search window,
     * weighted by {@code exp(-max(d^2 - 2 sigma^2, 0) / h^2)} where {@code d^2} is the mean
     * squared difference of the 3x3 patches around them.
     *
     * @param x the image
     * @param side its side
     * @param sigma the noise standard deviation
     * @param searchRadius the search window radius
     * @param h the filtering strength (about {@code 0.4 * sigma} to {@code sigma})
     * @return the filtered copy
     */
    public static float[] nonLocalMeans(float[] x, int side, double sigma, int searchRadius, double h) {
        float[] out = new float[x.length];
        double h2 = Math.max(1e-12, h * h);
        for (int y = 0; y < side; y++) {
            for (int xx = 0; xx < side; xx++) {
                double wsum = 0;
                double vsum = 0;
                for (int dy = -searchRadius; dy <= searchRadius; dy++) {
                    for (int dx = -searchRadius; dx <= searchRadius; dx++) {
                        int qy = reflect(y + dy, side);
                        int qx = reflect(xx + dx, side);
                        double d2 = 0;
                        for (int py = -1; py <= 1; py++) {
                            for (int px = -1; px <= 1; px++) {
                                double diff = x[reflect(y + py, side) * side + reflect(xx + px, side)]
                                        - x[reflect(qy + py, side) * side + reflect(qx + px, side)];
                                d2 += diff * diff;
                            }
                        }
                        d2 /= 9;
                        double wgt = Math.exp(-Math.max(d2 - 2 * sigma * sigma, 0) / h2);
                        wsum += wgt;
                        vsum += wgt * x[qy * side + qx];
                    }
                }
                out[y * side + xx] = (float) (vsum / wsum);
            }
        }
        return out;
    }

    static int reflect(int i, int n) {
        if (n == 1) {
            return 0;
        }
        int period = 2 * (n - 1);
        int j = Math.floorMod(i, period);
        return j >= n ? period - j : j;
    }

    /**
     * Wraps a per-image filter as an {@link ImageDenoiser}.
     *
     * @param filter the filter: image, side and noise level to the filtered image
     * @return the denoiser
     */
    public static ImageDenoiser of(PerImageFilter filter) {
        return (noisy, sigma) -> {
            int n = noisy.shape().dim(0);
            int side = noisy.shape().dim(2);
            List<float[]> imgs = DegradedImages.unstack(noisy.value());
            List<float[]> out = new ArrayList<>(n);
            for (float[] im : imgs) {
                out.add(filter.apply(im, side, sigma));
            }
            return Var.of(DegradedImages.stack(out, side));
        };
    }

    /** A filter on one square image. */
    @FunctionalInterface
    public interface PerImageFilter {
        /**
         * Filters an image.
         *
         * @param image the image
         * @param side its side
         * @param sigma the noise level
         * @return the filtered image
         */
        float[] apply(float[] image, int side, float sigma);
    }
}
