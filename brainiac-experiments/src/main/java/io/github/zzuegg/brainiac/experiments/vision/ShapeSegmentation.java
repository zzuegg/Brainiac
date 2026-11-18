package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.UNet2d;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Wave 6 item 2.6: semantic segmentation of {@link ShapeImages} scenes with a {@link UNet2d}
 * against the same encoder-decoder without skip connections. Each 12x12 scene holds two
 * noisy shapes; every pixel must be labelled background or the class of the shape it belongs
 * to (7 classes), which needs both the shape's identity (context) and its exact outline
 * (position). Reported: the mean intersection-over-union of the six shape classes and the
 * pixel accuracy on fresh scenes.
 */
public final class ShapeSegmentation {
    /** The scene side. */
    public static final int SIDE = 12;
    /** Background plus the shape classes. */
    public static final int CLASSES = ShapeImages.CLASSES + 1;

    private static final ShapeImages CLEAN = new ShapeImages(SIDE, 0f);

    private ShapeSegmentation() {
    }

    /**
     * Generates scenes.
     *
     * @param n the number of scenes
     * @param seed the seed
     * @return {@code {images [n, 1, 12, 12], pixel labels [n * 144]}}
     */
    public static Tensor[] scenes(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Tensor[] a = CLEAN.generate(n, r.nextLong());
        Tensor[] b = CLEAN.generate(n, r.nextLong());
        float[] av = a[0].toArray();
        float[] bv = b[0].toArray();
        float[] ac = a[1].toArray();
        float[] bc = b[1].toArray();
        int p = SIDE * SIDE;
        float[] img = new float[n * p];
        float[] lab = new float[n * p];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                int k = i * p + j;
                if (bv[k] > 0) {
                    img[k] = bv[k];
                    lab[k] = bc[i] + 1;
                } else if (av[k] > 0) {
                    img[k] = av[k];
                    lab[k] = ac[i] + 1;
                }
                img[k] += (float) (0.15 * r.nextGaussian());
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(n, 1, SIDE, SIDE), img), Tensor.of(Shape.of(n * p), lab)};
    }

    private static Var pixelLogits(UNet2d net, Tensor images) {
        int n = images.shape().dim(0);
        return net.forward(Var.of(images)).permute(0, 2, 3, 1).reshape(n * SIDE * SIDE, CLASSES);
    }

    /**
     * Trains a segmenter.
     *
     * @param skips a U-Net if true, the plain encoder-decoder otherwise
     * @param steps Adam steps of 16 scenes
     * @param seed the seed
     * @return {@code {mean IoU over the shape classes, pixel accuracy}} on 128 fresh scenes
     */
    public static double[] run(boolean skips, int steps, long seed) {
        UNet2d net = new UNet2d(1, CLASSES, 8, 2, skips, 0, new SplittableRandom(seed));
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            Tensor[] d = scenes(16, r.nextLong());
            Functional.crossEntropy(pixelLogits(net, d[0]), d[1]).backward();
            opt.step(net.parameters());
        }
        Tensor[] test = scenes(128, seed + 99);
        float[] pred = pixelLogits(net, test[0]).value().argmax(1).toArray();
        float[] truth = test[1].toArray();
        long[] inter = new long[CLASSES];
        long[] union = new long[CLASSES];
        int correct = 0;
        for (int i = 0; i < truth.length; i++) {
            int p = (int) pred[i];
            int t = (int) truth[i];
            correct += p == t ? 1 : 0;
            if (p == t) {
                inter[t]++;
                union[t]++;
            } else {
                union[t]++;
                union[p]++;
            }
        }
        double miou = 0;
        int classes = 0;
        for (int c = 1; c < CLASSES; c++) {
            if (union[c] > 0) {
                miou += (double) inter[c] / union[c];
                classes++;
            }
        }
        return new double[] {miou / classes, (double) correct / truth.length};
    }
}
