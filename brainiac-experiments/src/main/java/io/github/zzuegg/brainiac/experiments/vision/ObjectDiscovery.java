package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SlotAttention;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Conv;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.Padding;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 6 item 2.13: unsupervised object discovery on the two-shape scenes of
 * {@link ShapeSegmentation}. A slot-attention autoencoder (a SAME-padded CNN encoder with
 * position features, {@link SlotAttention} with 3 slots, and a spatial-broadcast decoder that
 * renders each slot as a value and an alpha logit per pixel, the alphas softmax-normalised over
 * the slots) is trained only to reconstruct the image. Its alpha masks are then a segmentation,
 * scored by the foreground adjusted Rand index (FG-ARI) against the true shape labels.
 *
 * <p>Baselines: a plain autoencoder of the same encoder with a dense bottleneck (reconstruction
 * only, no segmentation) and spatial k-means ({@code k = 3} on intensity and coordinates per
 * image) for FG-ARI.
 */
public final class ObjectDiscovery {

    private static final int SIDE = ShapeSegmentation.SIDE;
    private static final int PIXELS = SIDE * SIDE;
    private static final int SLOTS = 3;
    private static final int WIDTH = 32;
    private static final int CHANNELS = 16;

    private ObjectDiscovery() {
    }

    private static Model encoder(long seed) {
        return Model.sequential().input(1, SIDE, SIDE).seed(seed)
                .add(Conv.of(CHANNELS, 3, 3).padding(Padding.SAME).activation(Activation.RELU))
                .add(Conv.of(CHANNELS, 3, 3).padding(Padding.SAME).activation(Activation.RELU)).build();
    }

    /** Per-pixel coordinate features {@code [x, y, 1 - x, 1 - y]}, {@code [1, 144, 4]}. */
    private static Tensor grid() {
        float[] g = new float[PIXELS * 4];
        for (int i = 0; i < PIXELS; i++) {
            float x = (i / SIDE) / (SIDE - 1f);
            float y = (i % SIDE) / (SIDE - 1f);
            g[4 * i] = x;
            g[4 * i + 1] = y;
            g[4 * i + 2] = 1 - x;
            g[4 * i + 3] = 1 - y;
        }
        return Tensor.of(Shape.of(1, PIXELS, 4), g);
    }

    private static Var matrix(SplittableRandom r, int in, int out) {
        return Var.param(Tensor.normal(Shape.of(in, out), 0, (float) Math.sqrt(2.0 / (in + out)), r));
    }

    private static Var zeros(int n) {
        return Var.param(Tensor.zeros(Shape.of(n)));
    }

    /** The slot-attention autoencoder. */
    private static final class SlotAutoencoder {
        private final Model cnn;
        private final Var encPos;
        private final Var encPosBias;
        private final Var mlp;
        private final Var mlpBias;
        private final SlotAttention slots;
        private final Var decPos;
        private final Var dec1;
        private final Var dec1Bias;
        private final Var dec2;
        private final Var dec2Bias;
        private final Var dec3;
        private final Var dec3Bias;
        private final Tensor grid = grid();
        private Tensor masks;

        SlotAutoencoder(long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            cnn = encoder(seed);
            encPos = matrix(r, 4, CHANNELS);
            encPosBias = zeros(CHANNELS);
            mlp = matrix(r, CHANNELS, WIDTH);
            mlpBias = zeros(WIDTH);
            slots = new SlotAttention(WIDTH, SLOTS, WIDTH, 3, r);
            decPos = matrix(r, 4, WIDTH);
            dec1 = matrix(r, WIDTH, WIDTH);
            dec1Bias = zeros(WIDTH);
            dec2 = matrix(r, WIDTH, WIDTH);
            dec2Bias = zeros(WIDTH);
            dec3 = matrix(r, WIDTH, 2);
            dec3Bias = zeros(2);
        }

        Var reconstruct(Tensor images, boolean training) {
            int b = images.shape().dim(0);
            Var f = cnn.forward(Var.of(images), training).reshape(b, CHANNELS, PIXELS).permute(0, 2, 1);
            Var pos = Var.of(grid).matmul(encPos).add(encPosBias).expand(Shape.of(b, PIXELS, CHANNELS));
            Var features = f.add(pos).matmul(mlp).add(mlpBias).relu();
            Var s = slots.forward(features, training);                                  // [b, K, W]
            Var broadcast = s.reshape(b * SLOTS, 1, WIDTH).expand(Shape.of(b * SLOTS, PIXELS, WIDTH));
            Var p = Var.of(grid).matmul(decPos).expand(Shape.of(b * SLOTS, PIXELS, WIDTH));
            Var h = broadcast.add(p).reshape(b * SLOTS * PIXELS, WIDTH);
            h = h.matmul(dec1).add(dec1Bias).relu().matmul(dec2).add(dec2Bias).relu().matmul(dec3).add(dec3Bias);
            Var out = h.reshape(b, SLOTS, PIXELS, 2).permute(0, 2, 1, 3);               // [b, P, K, 2]
            Var values = out.slice(3, 0, 1).reshape(b, PIXELS, SLOTS);
            Var alpha = Activation.SOFTMAX.forward(out.slice(3, 1, 2).reshape(b, PIXELS, SLOTS));
            masks = alpha.value();
            return values.mul(alpha).sum(2, false).reshape(b, 1, SIDE, SIDE);
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(cnn.parameters());
            p.addAll(List.of(encPos, encPosBias, mlp, mlpBias, decPos, dec1, dec1Bias, dec2, dec2Bias, dec3, dec3Bias));
            p.addAll(slots.parameters());
            return p;
        }
    }

    /** The plain autoencoder baseline with a 32-wide dense bottleneck. */
    private static final class PlainAutoencoder {
        private final Model net;

        PlainAutoencoder(long seed) {
            net = Model.sequential().input(1, SIDE, SIDE).seed(seed)
                    .add(Conv.of(CHANNELS, 3, 3).padding(Padding.SAME).activation(Activation.RELU))
                    .add(Conv.of(CHANNELS, 3, 3).padding(Padding.SAME).activation(Activation.RELU)).flatten()
                    .dense(SLOTS * WIDTH, Activation.RELU).dense(128, Activation.RELU)
                    .dense(PIXELS, Activation.IDENTITY).build();
        }

        Var reconstruct(Tensor images, boolean training) {
            return net.forward(Var.of(images), training).reshape(images.shape().dim(0), 1, SIDE, SIDE);
        }
    }

    /**
     * Trains an autoencoder on scenes.
     *
     * @param slotted the slot-attention autoencoder if true, the plain one otherwise
     * @param steps Adam steps of 16 scenes
     * @param seed the seed
     * @return {@code {reconstruction MSE, FG-ARI}} on 128 fresh scenes (FG-ARI NaN for the plain one)
     */
    public static double[] run(boolean slotted, int steps, long seed) {
        SlotAutoencoder slot = slotted ? new SlotAutoencoder(seed) : null;
        PlainAutoencoder plain = slotted ? null : new PlainAutoencoder(seed);
        List<Var> params = slotted ? slot.parameters() : plain.net.parameters();
        Optimizer opt = Optimizer.adam(slotted ? 2e-3f : 1e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            Tensor x = ShapeSegmentation.scenes(16, r.nextLong())[0];
            Var y = slotted ? slot.reconstruct(x, true) : plain.reconstruct(x, true);
            y.sub(Var.of(x)).square().mean().backward();
            opt.step(params);
        }
        Tensor[] test = ShapeSegmentation.scenes(128, seed + 99);
        Var y = slotted ? slot.reconstruct(test[0], false) : plain.reconstruct(test[0], false);
        double mse = y.sub(Var.of(test[0])).square().mean().value().item();
        if (!slotted) {
            return new double[] {mse, Double.NaN};
        }
        float[] m = slot.masks.toArray();
        int[] assign = new int[128 * PIXELS];
        for (int i = 0; i < assign.length; i++) {
            int best = 0;
            for (int k = 1; k < SLOTS; k++) {
                if (m[i * SLOTS + k] > m[i * SLOTS + best]) {
                    best = k;
                }
            }
            assign[i] = best;
        }
        return new double[] {mse, foregroundAri(assign, test[1].toArray())};
    }

    /**
     * The spatial k-means baseline: per image, k-means with {@code k = 3} on (intensity,
     * x, y) of every pixel.
     *
     * @param seed the seed of the test scenes
     * @return FG-ARI on 128 scenes
     */
    public static double kMeans(long seed) {
        Tensor[] test = ShapeSegmentation.scenes(128, seed + 99);
        float[] img = test[0].toArray();
        int[] assign = new int[img.length];
        SplittableRandom r = new SplittableRandom(seed);
        for (int i = 0; i < 128; i++) {
            double[][] pts = new double[PIXELS][];
            for (int j = 0; j < PIXELS; j++) {
                pts[j] = new double[] {2 * img[i * PIXELS + j], (j / SIDE) / (SIDE - 1.0), (j % SIDE) / (SIDE - 1.0)};
            }
            double[][] c = new double[SLOTS][];
            for (int k = 0; k < SLOTS; k++) {
                c[k] = pts[r.nextInt(PIXELS)].clone();
            }
            int[] a = new int[PIXELS];
            for (int it = 0; it < 20; it++) {
                double[][] sum = new double[SLOTS][3];
                int[] count = new int[SLOTS];
                for (int j = 0; j < PIXELS; j++) {
                    int best = 0;
                    double bd = Double.MAX_VALUE;
                    for (int k = 0; k < SLOTS; k++) {
                        double d = 0;
                        for (int q = 0; q < 3; q++) {
                            d += (pts[j][q] - c[k][q]) * (pts[j][q] - c[k][q]);
                        }
                        if (d < bd) {
                            bd = d;
                            best = k;
                        }
                    }
                    a[j] = best;
                    count[best]++;
                    for (int q = 0; q < 3; q++) {
                        sum[best][q] += pts[j][q];
                    }
                }
                for (int k = 0; k < SLOTS; k++) {
                    if (count[k] > 0) {
                        for (int q = 0; q < 3; q++) {
                            c[k][q] = sum[k][q] / count[k];
                        }
                    }
                }
            }
            System.arraycopy(a, 0, assign, i * PIXELS, PIXELS);
        }
        return foregroundAri(assign, test[1].toArray());
    }

    /** Mean per-image adjusted Rand index over the foreground pixels (true label > 0). */
    static double foregroundAri(int[] assign, float[] labels) {
        int images = labels.length / PIXELS;
        double total = 0;
        int counted = 0;
        for (int i = 0; i < images; i++) {
            long[][] table = new long[ShapeSegmentation.CLASSES][SLOTS];
            int n = 0;
            for (int j = 0; j < PIXELS; j++) {
                int t = (int) labels[i * PIXELS + j];
                if (t > 0) {
                    table[t][assign[i * PIXELS + j]]++;
                    n++;
                }
            }
            if (n < 2) {
                continue;
            }
            double index = 0;
            long[] rows = new long[table.length];
            long[] cols = new long[SLOTS];
            for (int t = 0; t < table.length; t++) {
                for (int k = 0; k < SLOTS; k++) {
                    index += pairs(table[t][k]);
                    rows[t] += table[t][k];
                    cols[k] += table[t][k];
                }
            }
            double a = 0;
            double b = 0;
            for (long v : rows) {
                a += pairs(v);
            }
            for (long v : cols) {
                b += pairs(v);
            }
            double expected = a * b / pairs(n);
            double max = (a + b) / 2;
            total += max == expected ? 1 : (index - expected) / (max - expected);
            counted++;
        }
        return total / counted;
    }

    private static double pairs(long v) {
        return v * (v - 1) / 2.0;
    }
}
