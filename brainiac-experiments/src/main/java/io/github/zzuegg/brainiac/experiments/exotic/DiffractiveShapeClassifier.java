package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.DiffractiveLayer;
import io.github.zzuegg.brainiac.experiments.generative.ToyImages;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * An all-optical classifier in the style of the diffractive deep neural network (Lin et al.,
 * <a href="https://arxiv.org/abs/1804.08711">1804.08711</a>): an 8x8 shape image lights a
 * coherent field (amplitude encoding, upsampled and padded), which passes through a few
 * trained phase masks ({@link DiffractiveLayer}) and lands on four detector squares, one
 * per class. The class is the brightest detector; the only trained quantities are the
 * phases of the masks, which in hardware would be the thickness of 3-D printed plates.
 */
@SuppressWarnings("exports")   // experimental.* packages are exported by the lead at integration
public final class DiffractiveShapeClassifier {
    private static final int UPSAMPLE = 2;
    private final int size;
    private final List<DiffractiveLayer> layers = new ArrayList<>();
    private final Tensor detectors;
    private final Optimizer optimizer;
    private final SplittableRandom random;
    private final float temperature;

    private DiffractiveShapeClassifier(int size, int depth, double wavelength, double distance, float learningRate,
            long seed) {
        this.size = size;
        for (int i = 0; i < depth; i++) {
            layers.add(DiffractiveLayer.of(size, wavelength, distance, seed + i));
        }
        this.detectors = detectors(size);
        this.optimizer = Optimizer.adam(learningRate);
        this.random = new SplittableRandom(seed + depth);
        this.temperature = 10;
    }

    /**
     * Creates the classifier.
     *
     * @param size the field side length in pixels, at least 24
     * @param depth the number of diffractive layers
     * @param wavelength the wavelength in pixel pitches
     * @param distance the spacing between planes in pixel pitches
     * @param learningRate the Adam learning rate for the phases
     * @param seed the seed
     * @return the classifier
     */
    public static DiffractiveShapeClassifier of(int size, int depth, double wavelength, double distance,
            float learningRate, long seed) {
        if (size < ToyImages.SIDE * UPSAMPLE + 8) {
            throw new IllegalArgumentException("size too small for the padded input");
        }
        return new DiffractiveShapeClassifier(size, depth, wavelength, distance, learningRate, seed);
    }

    /** Four 4x4 detector squares around the centre, as a {@code [n*n, 4]} selection matrix. */
    private static Tensor detectors(int n) {
        float[] m = new float[n * n * 4];
        int c = n / 2;
        int[][] corners = {{c - 7, c - 7}, {c - 7, c + 3}, {c + 3, c - 7}, {c + 3, c + 3}};
        for (int k = 0; k < 4; k++) {
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    m[((corners[k][0] + y) * n + corners[k][1] + x) * 4 + k] = 1;
                }
            }
        }
        return Tensor.of(Shape.of(n * n, 4), m);
    }

    /**
     * Encodes images as input fields: lit pixels have amplitude 1 and phase 0.
     *
     * @param images 8x8 images in {-1, +1}
     * @return {@code [batch, n, n, 2]}
     */
    public Tensor encode(List<float[]> images) {
        int n = size;
        float[] f = new float[images.size() * n * n * 2];
        int offset = (n - ToyImages.SIDE * UPSAMPLE) / 2;
        for (int b = 0; b < images.size(); b++) {
            float[] img = images.get(b);
            for (int y = 0; y < ToyImages.SIDE * UPSAMPLE; y++) {
                for (int x = 0; x < ToyImages.SIDE * UPSAMPLE; x++) {
                    if (img[(y / UPSAMPLE) * ToyImages.SIDE + x / UPSAMPLE] > 0) {
                        f[((b * n + offset + y) * n + offset + x) * 2] = 1;
                    }
                }
            }
        }
        return Tensor.of(Shape.of(images.size(), n, n, 2), f);
    }

    /**
     * Light reaching each detector, as shares of the total detected light.
     *
     * @param field input fields
     * @return {@code [batch, 4]}
     */
    public Var detect(Tensor field) {
        Var e = Var.of(field);
        for (DiffractiveLayer l : layers) {
            e = l.forward(e);
        }
        int b = field.shape().dim(0);
        Var signal = DiffractiveLayer.intensity(e).reshape(b, size * size).matmul(Var.of(detectors));
        return signal.div(signal.sum(1, true).add(1e-6f));
    }

    /**
     * One training step on random shapes.
     *
     * @param batch the batch size
     * @return the cross-entropy loss
     */
    public float trainStep(int batch) {
        List<float[]> images = new ArrayList<>();
        float[] labels = new float[batch];
        for (int i = 0; i < batch; i++) {
            labels[i] = i % ToyImages.CLASSES;
            images.add(ToyImages.sample(i % ToyImages.CLASSES, random));
        }
        Var loss = Functional.crossEntropy(detect(encode(images)).mul(temperature), Tensor.of(Shape.of(batch), labels));
        float v = loss.value().item();
        loss.backward();
        List<Var> params = new ArrayList<>();
        for (DiffractiveLayer l : layers) {
            params.addAll(l.parameters());
        }
        optimizer.step(params);
        return v;
    }

    /**
     * Accuracy over every image of the data set.
     *
     * @return the share whose brightest detector is their class
     */
    public double accuracy() {
        List<float[]> images = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        for (int c = 0; c < ToyImages.CLASSES; c++) {
            for (float[] img : ToyImages.all(c)) {
                images.add(img);
                labels.add(c);
            }
        }
        float[] pred = detect(encode(images)).value().argmax(1).toArray();
        int right = 0;
        for (int i = 0; i < pred.length; i++) {
            right += Math.round(pred[i]) == labels.get(i) ? 1 : 0;
        }
        return (double) right / pred.length;
    }
}
