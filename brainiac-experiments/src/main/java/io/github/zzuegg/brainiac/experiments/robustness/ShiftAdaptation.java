package io.github.zzuegg.brainiac.experiments.robustness;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.robust.TestTimeAdaptation;
import io.github.zzuegg.brainiac.experiments.techniques.CnnNormalizationShootout;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Wave 8 item 8.6.12: test-time adaptation ({@link TestTimeAdaptation}) of the
 * {@link CnnNormalizationShootout} CNNs (batch norm and group norm) on {@link ShapeImages}
 * under test-time corruptions (Gaussian noise, blur, low contrast, inversion) at severities 1
 * to 5. Compared: no adaptation, test-batch BN statistics, TENT and MEMO (batch size 1), the
 * group-norm model (which has no batch statistics to adapt) and a model trained with the
 * corruptions as augmentation. A label-shift stream (every test batch holds one class) checks
 * the known failure of batch-statistics methods, and a long TENT stream checks for collapse.
 */
public final class ShiftAdaptation {

    /** A test-time corruption. */
    public enum Corruption {
        /** additive Gaussian noise */
        NOISE,
        /** 3x3 box blur, repeated with severity */
        BLUR,
        /** contrast reduced towards the image mean */
        CONTRAST,
        /** {@code x -> 1 - x} with severity-dependent strength */
        INVERT
    }

    /**
     * One evaluation.
     *
     * @param accuracy test accuracy
     * @param clean the same model's clean accuracy without adaptation
     */
    public record Outcome(double accuracy, double clean) {
    }

    private static final ShapeImages IMAGES = new ShapeImages(8, 0.1f);

    private ShiftAdaptation() {
    }

    /**
     * Corrupts a batch.
     *
     * @param x images {@code [n, 1, 8, 8]}
     * @param c the corruption
     * @param severity 1 to 5, 0 for none
     * @param r the noise source
     * @return the corrupted images
     */
    public static Tensor corrupt(Tensor x, Corruption c, int severity, SplittableRandom r) {
        if (severity == 0) {
            return x;
        }
        float[] v = x.toArray();
        int n = x.shape().dim(0);
        switch (c) {
            case NOISE -> {
                float s = 0.15f * severity;
                for (int i = 0; i < v.length; i++) {
                    v[i] += (float) r.nextGaussian() * s;
                }
            }
            case BLUR -> {
                for (int rep = 0; rep < (severity + 1) / 2; rep++) {
                    float[] out = new float[v.length];
                    for (int i = 0; i < n; i++) {
                        for (int y = 0; y < 8; y++) {
                            for (int xx = 0; xx < 8; xx++) {
                                float sum = 0;
                                int cnt = 0;
                                for (int dy = -1; dy <= 1; dy++) {
                                    for (int dx = -1; dx <= 1; dx++) {
                                        int yy = y + dy;
                                        int xc = xx + dx;
                                        if (yy >= 0 && yy < 8 && xc >= 0 && xc < 8) {
                                            sum += v[i * 64 + yy * 8 + xc];
                                            cnt++;
                                        }
                                    }
                                }
                                float self = v[i * 64 + y * 8 + xx];
                                float w = severity % 2 == 1 ? 0.5f : 1f;
                                out[i * 64 + y * 8 + xx] = (1 - w) * self + w * sum / cnt;
                            }
                        }
                    }
                    v = out;
                }
            }
            case CONTRAST -> {
                float k = 1 - 0.17f * severity;
                for (int i = 0; i < n; i++) {
                    float mean = 0;
                    for (int j = 0; j < 64; j++) {
                        mean += v[i * 64 + j] / 64;
                    }
                    for (int j = 0; j < 64; j++) {
                        v[i * 64 + j] = mean + k * (v[i * 64 + j] - mean) + 0.1f * severity;
                    }
                }
            }
            case INVERT -> {
                float a = 0.2f * severity;
                for (int i = 0; i < v.length; i++) {
                    v[i] = (1 - a) * v[i] + a * (1 - v[i]);
                }
            }
        }
        return Tensor.of(x.shape(), v);
    }

    /**
     * Trains a CNN on clean shapes, or with random corruptions as augmentation.
     *
     * @param norm batch norm or group norm
     * @param augmented train on randomly corrupted images (half of each batch)
     * @param steps Adam steps
     * @param seed the seed
     * @return the model
     */
    public static Model train(CnnNormalizationShootout.Norm norm, boolean augmented, int steps, long seed) {
        Model m = CnnNormalizationShootout.build(norm, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            Tensor[] d = IMAGES.generate(32, r.nextLong());
            Tensor x = d[0];
            if (augmented && s % 2 == 1) {
                x = corrupt(x, Corruption.values()[r.nextInt(4)], 1 + r.nextInt(5), r);
            }
            Functional.crossEntropy(m.forward(Var.of(x), true), d[1]).backward();
            opt.step(m.parameters());
        }
        return m;
    }

    /**
     * Accuracy of an adapted model on a corrupted test stream.
     *
     * @param model the trained model (reset afterwards)
     * @param method the adaptation
     * @param c the corruption
     * @param severity the severity
     * @param batches test batches
     * @param batchSize images per batch (MEMO handles them one by one)
     * @param labelShift every batch holds a single class
     * @param seed the seed
     * @return the accuracy
     */
    public static double evaluate(Model model, TestTimeAdaptation.Method method, Corruption c, int severity,
            int batches, int batchSize, boolean labelShift, long seed) {
        TestTimeAdaptation tta = TestTimeAdaptation.of(model, method)
                .learningRate(method == TestTimeAdaptation.Method.MEMO ? 0.005f : 1e-3f)
                .augmentations(ShiftAdaptation::augment);
        SplittableRandom r = new SplittableRandom(seed);
        double right = 0;
        int total = 0;
        for (int b = 0; b < batches; b++) {
            Tensor[] d = labelShift ? oneClass(batchSize, b % ShapeImages.CLASSES, r) : IMAGES.generate(batchSize,
                    r.nextLong());
            Tensor x = corrupt(d[0], c, severity, r);
            Tensor logits = tta.predict(x);
            right += logits.argmax(1).equalTo(d[1]).sum().item();
            total += batchSize;
        }
        tta.reset();
        return right / total;
    }

    private static Tensor[] oneClass(int size, int cls, SplittableRandom r) {
        float[] img = new float[size * 64];
        float[] lab = new float[size];
        int got = 0;
        while (got < size) {
            Tensor[] g = IMAGES.generate(32, r.nextLong());
            float[] pv = g[0].toArray();
            float[] iv = g[1].toArray();
            for (int i = 0; i < 32 && got < size; i++) {
                if ((int) iv[i] == cls) {
                    System.arraycopy(pv, i * 64, img, got * 64, 64);
                    lab[got++] = cls;
                }
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(size, 1, 8, 8), img), Tensor.of(Shape.of(size), lab)};
    }

    /** MEMO's augmentations: 16 random wrap-around shifts with small noise. */
    static Tensor augment(Tensor one) {
        float[] v = one.toArray();
        int copies = 16;
        float[] out = new float[copies * 64];
        SplittableRandom r = new SplittableRandom(java.util.Arrays.hashCode(v));
        for (int k = 0; k < copies; k++) {
            int dy = r.nextInt(3) - 1;
            int dx = r.nextInt(3) - 1;
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    out[k * 64 + y * 8 + x] = v[Math.floorMod(y - dy, 8) * 8 + Math.floorMod(x - dx, 8)]
                            + (float) r.nextGaussian() * 0.05f;
                }
            }
        }
        return Tensor.of(Shape.of(copies, 1, 8, 8), out);
    }
}
