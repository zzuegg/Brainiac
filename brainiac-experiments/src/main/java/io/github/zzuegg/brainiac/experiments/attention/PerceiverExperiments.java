package io.github.zzuegg.brainiac.experiments.attention;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.PerceiverEncoder;
import io.github.zzuegg.brainiac.experiments.vision.ClassicCnnShootout;
import io.github.zzuegg.brainiac.experiments.vision.VisionShootout;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 6 item 2.12: the Perceiver's latent bottleneck.
 *
 * <ul>
 *   <li>Image classification from raw pixels: each of the 64 pixels of an 8x8
 *       {@link ShapeImages} image is one input element (its value plus Fourier features of its
 *       coordinates), a {@link PerceiverEncoder} with 4 or 16 latents reads them, and the mean
 *       latent is classified. Baseline: the patch ViT of {@link VisionShootout}. Reported:
 *       accuracy and multiply-adds per image.</li>
 *   <li>Perceiver IO on a long sequence: reverse 64 tokens from a vocabulary of 8. The encoder
 *       compresses the sequence into {@code n} latents, and 64 output queries (sinusoidal
 *       position features) decode it. Baseline: a full transformer encoder over the 64 tokens.
 *       Reported: token accuracy against the latent count, and multiply-adds.</li>
 * </ul>
 */
public final class PerceiverExperiments {

    private static final ShapeImages IMAGES = new ShapeImages(8, 0.1f);
    private static final int SIDE = 8;
    private static final int BANDS = 4;
    private static final float PIXEL_SCALE = 4f;
    private static final int PIXEL_FEATURES = 1 + 2 + 4 * BANDS;
    /** Sequence length of the reversal task. */
    public static final int LENGTH = 64;
    /** Vocabulary of the reversal task. */
    public static final int VOCAB = 8;
    private static final int WIDTH = 32;

    private PerceiverExperiments() {
    }

    /** Pixel value, coordinates and their Fourier features per pixel: {@code [n, 64, 19]}. */
    private static Tensor pixelArray(Tensor images) {
        int n = images.shape().dim(0);
        float[] img = images.toArray();
        int p = SIDE * SIDE;
        float[] out = new float[n * p * PIXEL_FEATURES];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                int o = (i * p + j) * PIXEL_FEATURES;
                double x = (j / SIDE) / (SIDE - 1.0) * 2 - 1;
                double y = (j % SIDE) / (SIDE - 1.0) * 2 - 1;
                out[o] = PIXEL_SCALE * img[i * p + j];
                out[o + 1] = (float) x;
                out[o + 2] = (float) y;
                for (int k = 0; k < BANDS; k++) {
                    double f = Math.PI * (k + 1);
                    out[o + 3 + 4 * k] = (float) Math.sin(f * x);
                    out[o + 4 + 4 * k] = (float) Math.cos(f * x);
                    out[o + 5 + 4 * k] = (float) Math.sin(f * y);
                    out[o + 6 + 4 * k] = (float) Math.cos(f * y);
                }
            }
        }
        return Tensor.of(Shape.of(n, p, PIXEL_FEATURES), out);
    }

    private static Var matrix(SplittableRandom r, int in, int out) {
        return Var.param(Tensor.normal(Shape.of(in, out), 0, (float) Math.sqrt(2.0 / (in + out)), r));
    }

    /**
     * Classifies 8x8 shape images from raw pixels with a Perceiver.
     *
     * @param latents the latent array length
     * @param steps AdamW steps of 32 images
     * @param seed the seed
     * @return {@code {test accuracy, multiply-adds per image}}
     */
    public static double[] perceiverImages(int latents, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        PerceiverEncoder enc = new PerceiverEncoder(PIXEL_FEATURES, latents, WIDTH, 4, 1, 0, r);
        Var head = matrix(r, WIDTH, ShapeImages.CLASSES);
        Var bias = Var.param(Tensor.zeros(Shape.of(ShapeImages.CLASSES)));
        List<Var> params = new ArrayList<>(enc.parameters());
        params.add(head);
        params.add(bias);
        Optimizer opt = Optimizer.adamW(3e-3f, 1e-4f);
        SplittableRandom data = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            Tensor[] d = IMAGES.generate(32, data.nextLong());
            Var logits = enc.forward(Var.of(pixelArray(d[0])), true).mean(1, false).matmul(head).add(bias);
            Functional.crossEntropy(logits, d[1]).backward();
            opt.step(params);
        }
        Tensor[] test = IMAGES.generate(512, seed + 99);
        Var logits = enc.forward(Var.of(pixelArray(test[0])), false).mean(1, false).matmul(head).add(bias);
        double acc = logits.value().argmax(1).equalTo(test[1]).mean().item();
        return new double[] {acc, enc.multiplyAdds(SIDE * SIDE) + (long) WIDTH * ShapeImages.CLASSES};
    }

    /**
     * The ViT baseline of {@link VisionShootout} with its multiply-adds: the projections as
     * estimated by {@link ClassicCnnShootout#multiplyAdds} plus the attention scores and mixing
     * of its two blocks over 17 tokens.
     *
     * @param steps AdamW steps of 32 images
     * @param seed the seed
     * @return {@code {test accuracy, multiply-adds per image}}
     */
    public static double[] vitImages(int steps, long seed) {
        double acc = VisionShootout.run(VisionShootout.Architecture.VIT, steps, 32, seed);
        Model m = VisionShootout.build(VisionShootout.Architecture.VIT, seed);
        long tokens = (SIDE / 2) * (SIDE / 2) + 1;
        long macs = ClassicCnnShootout.multiplyAdds(m, Shape.of(1, SIDE, SIDE)) + 2 * 2 * tokens * tokens * WIDTH;
        return new double[] {acc, macs};
    }

    /** {@code {tokens [b, L], reversed targets [b * L]}}. */
    private static int[][] reversal(int b, SplittableRandom r) {
        int[][] x = new int[2][b * LENGTH];
        for (int i = 0; i < b; i++) {
            for (int t = 0; t < LENGTH; t++) {
                int v = r.nextInt(VOCAB);
                x[0][i * LENGTH + t] = v;
                x[1][i * LENGTH + LENGTH - 1 - t] = v;
            }
        }
        return x;
    }

    private static Tensor floats(int[] v, int... dims) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = v[i];
        }
        return Tensor.of(Shape.of(dims), f);
    }

    /** A Perceiver IO or a full transformer encoder for the reversal task. */
    private static final class Reverser {
        private final Var table;
        private final Tensor positions;
        private final PerceiverEncoder perceiver;
        private final List<Layer> blocks = new ArrayList<>();
        private final Var out;
        private final Var outBias;

        Reverser(int latents, SplittableRandom r) {
            table = Var.param(Tensor.normal(Shape.of(VOCAB, WIDTH / 2), 0, 1, r));
            positions = Functional.sinusoidal(LENGTH, WIDTH / 2, 100f, table.value().device());
            if (latents > 0) {
                perceiver = new PerceiverEncoder(WIDTH, latents, WIDTH, 4, 1, WIDTH / 2, r);
            } else {
                perceiver = null;
                for (int i = 0; i < 2; i++) {
                    blocks.add(TransformerBlock.of(4).ffnSize(2 * WIDTH).create(Shape.of(LENGTH, WIDTH), r));
                }
            }
            int head = latents > 0 ? WIDTH / 2 : WIDTH;
            out = matrix(r, head, VOCAB);
            outBias = Var.param(Tensor.zeros(Shape.of(VOCAB)));
        }

        Var logits(int[] tokens, int b, boolean training) {
            Var emb = table.gather(0, floats(tokens, b, LENGTH));
            Var pos = Var.of(positions.reshape(1, LENGTH, WIDTH / 2)).expand(Shape.of(b, LENGTH, WIDTH / 2));
            Var x = Var.concat(2, emb, pos);
            Var h;
            if (perceiver != null) {
                Var z = perceiver.forward(x, training);
                h = perceiver.decode(z, pos, training);
            } else {
                h = x;
                for (Layer l : blocks) {
                    h = l.forward(h, training);
                }
            }
            return h.reshape(b * LENGTH, h.shape().dim(2)).matmul(out).add(outBias);
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(table, out, outBias));
            if (perceiver != null) {
                p.addAll(perceiver.parameters());
            }
            for (Layer l : blocks) {
                p.addAll(l.parameters());
            }
            return p;
        }

        long multiplyAdds(int latents) {
            long l = LENGTH;
            long d = WIDTH;
            if (perceiver == null) {
                return 2 * (8 * l * d * d + 2 * l * l * d) + l * d * VOCAB;
            }
            long q = WIDTH / 2;
            long decoder = 2 * l * q * q + 2L * latents * d * q + 2 * l * latents * q + 4 * l * q * q;
            return perceiver.multiplyAdds(LENGTH) + decoder + l * q * VOCAB;
        }
    }

    /**
     * Reverses 64-token sequences.
     *
     * @param latents the Perceiver IO latent count, or 0 for the full transformer encoder
     * @param steps Adam steps of 16 sequences
     * @param seed the seed
     * @return {@code {token accuracy on 256 fresh sequences, multiply-adds per sequence}}
     */
    public static double[] reversal(int latents, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Reverser m = new Reverser(latents, r);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < steps; s++) {
            int[][] d = reversal(16, r);
            Functional.crossEntropy(m.logits(d[0], 16, true), floats(d[1], 16 * LENGTH)).backward();
            opt.step(m.parameters());
        }
        int[][] d = reversal(256, new SplittableRandom(seed + 5));
        Tensor pred = m.logits(d[0], 256, false).value().argmax(1);
        double acc = pred.equalTo(floats(d[1], 256 * LENGTH)).mean().item();
        return new double[] {acc, m.multiplyAdds(latents)};
    }
}
