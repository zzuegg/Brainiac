package io.github.zzuegg.brainiac.experiments.techniques;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.ConcreteDropout;
import io.github.zzuegg.brainiac.experimental.nn.DropConnectDense;
import io.github.zzuegg.brainiac.experimental.nn.VariationalGru;
import io.github.zzuegg.brainiac.experimental.train.Calibration;
import io.github.zzuegg.brainiac.experiments.language.CountingTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Wave 6 item 2.1, classic dropout variants against plain dropout:
 *
 * <ul>
 *   <li><b>Few-shot noisy shapes.</b> A small CNN (a 3x3 convolution, pooling, then a 64-unit hidden layer and the
 *       classifier, the two regularised layers) learns {@link ShapeImages} from 200
 *       noisy images with no regulariser, core {@code Dropout}, {@link DropConnectDense} (rate 0.5 on the hidden layer) or
 *       {@link ConcreteDropout} (learned rate). Reported: test accuracy and expected calibration
 *       error ({@link Calibration}).</li>
 *   <li><b>GRU language model.</b> A GRU next-token model on {@link CountingTasks} trained on a
 *       small fixed set of sequences, with {@link VariationalGru} masks tied over time
 *       (variational dropout), fresh per-step masks (the naive baseline) or none. Reported: test
 *       next-token accuracy on fresh sequences.</li>
 * </ul>
 */
public final class ClassicRegularizers {

    /** The regularisers of the image arm. Dropout and concrete dropout act on both dense layers. */
    public enum Regularizer {
        /** no regulariser */
        NONE,
        /** core unit dropout, rate 0.3 */
        DROPOUT,
        /** DropConnect on the hidden weights, rate 0.5 */
        DROPCONNECT,
        /** concrete dropout with learned rates */
        CONCRETE
    }

    /**
     * An image-arm result.
     *
     * @param accuracy test accuracy
     * @param ece expected calibration error (15 bins)
     * @param learnedRate the mean learned drop rate for concrete dropout, else {@code NaN}
     */
    public record ImageResult(double accuracy, double ece, double learnedRate) {
    }

    private static final int TRAIN = 200;

    private ClassicRegularizers() {
    }

    private static Model imageModel(Regularizer r, long seed) {
        Model.Builder b = Model.sequential().input(1, 8, 8).seed(seed).conv2d(8, 3, Activation.RELU).maxPool(2, 2)
                .flatten();
        switch (r) {
            case NONE -> b.dense(64, Activation.RELU).dense(ShapeImages.CLASSES, Activation.IDENTITY);
            case DROPOUT -> b.dropout(0.3f).dense(64, Activation.RELU).dropout(0.3f)
                    .dense(ShapeImages.CLASSES, Activation.IDENTITY);
            case DROPCONNECT -> b.add(DropConnectDense.spec(64, Activation.RELU, 0.5f))
                    .dense(ShapeImages.CLASSES, Activation.IDENTITY);
            case CONCRETE -> b.add(ConcreteDropout.spec(Dense.spec(64, Activation.RELU, Init.HE), 0.1f, 1e-5f,
                    1f / TRAIN)).add(ConcreteDropout.spec(Dense.spec(ShapeImages.CLASSES, Activation.IDENTITY,
                    Init.XAVIER), 0.1f, 1e-5f, 1f / TRAIN));
        }
        return b.build();
    }

    /**
     * Trains the MLP on few noisy images with one regulariser.
     *
     * @param r the regulariser
     * @param steps full-batch Adam steps
     * @param seed the seed
     * @return the result
     */
    public static ImageResult images(Regularizer r, int steps, long seed) {
        ShapeImages data = new ShapeImages(8, 0.25f);
        Tensor[] train = data.generate(TRAIN, seed);
        Tensor x = train[0];
        Model model = imageModel(r, seed);
        Optimizer opt = Optimizer.adam(2e-3f);
        for (int s = 0; s < steps; s++) {
            Var loss = Functional.crossEntropy(model.forward(Var.of(x), true), train[1]);
            Var aux = model.auxiliaryLoss();
            if (aux != null) {
                loss = loss.add(aux);
            }
            loss.backward();
            opt.step(model.parameters());
        }
        int n = 1000;
        Tensor[] test = data.generate(n, seed + 7);
        float[] logits = model.forward(Var.of(test[0])).value().toArray();
        float[] ids = test[1].toArray();
        double[][] probs = new double[n][];
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) {
            double[] l = new double[ShapeImages.CLASSES];
            for (int c = 0; c < l.length; c++) {
                l[c] = logits[i * l.length + c];
            }
            probs[i] = Calibration.softmax(l, 1);
            labels[i] = (int) ids[i];
        }
        double rate = Double.NaN;
        if (r == Regularizer.CONCRETE) {
            double sum = 0;
            int count = 0;
            for (Layer l : model.layers()) {
                if (l instanceof ConcreteDropout c) {
                    sum += c.rate();
                    count++;
                }
            }
            rate = sum / count;
        }
        return new ImageResult(Calibration.accuracy(probs, labels), Calibration.expectedCalibrationError(probs, labels,
                15), rate);
    }

    private static final int LENGTH = 12;
    private static final int VOCAB = 16;
    private static final int[] STRIDES = {1, 2, 3, 4, 5, 6, 7};

    private static Tensor tokens(int[][] seqs) {
        float[] f = new float[seqs.length * LENGTH];
        for (int i = 0; i < seqs.length; i++) {
            for (int t = 0; t < LENGTH; t++) {
                f[i * LENGTH + t] = seqs[i][t];
            }
        }
        return Tensor.of(Shape.of(seqs.length, LENGTH), f);
    }

    /** Next-token targets; the first position is skipped (the stride is unknown there). */
    private static Tensor targets(int[][] seqs) {
        int n = seqs.length;
        float[] f = new float[n * LENGTH];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < LENGTH; t++) {
                f[i * LENGTH + t] = t == 0 || t == LENGTH - 1 ? -1 : seqs[i][t + 1];
            }
        }
        return Tensor.of(Shape.of(n * LENGTH), f);
    }

    /**
     * Trains a GRU next-token model on a small fixed set of counting sequences.
     *
     * @param masks how the GRU's dropout masks are drawn
     * @param rate the dropout rate
     * @param trainSequences the size of the fixed training set
     * @param steps full-batch Adam steps
     * @param seed the seed
     * @return next-token accuracy on 256 fresh sequences (positions 1 to 10)
     */
    public static double gruLanguageModel(VariationalGru.Masks masks, float rate, int trainSequences, int steps,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int[][] train = CountingTasks.sequences(r, trainSequences, LENGTH, VOCAB, STRIDES);
        Model model = Model.sequential().input(LENGTH).embedding(VOCAB, 16).add(VariationalGru.spec(64, masks, rate))
                .dense(VOCAB, Activation.IDENTITY).seed(seed).build();
        Optimizer opt = Optimizer.adam(5e-3f);
        Tensor x = tokens(train);
        Tensor y = targets(train);
        for (int s = 0; s < steps; s++) {
            Var logits = model.forward(Var.of(x), true).reshape(trainSequences * LENGTH, VOCAB);
            Functional.crossEntropy(logits, y).backward();
            opt.step(model.parameters());
        }
        int[][] test = CountingTasks.sequences(new SplittableRandom(seed + 99), 256, LENGTH, VOCAB, STRIDES);
        int[] pred = argmax(model.forward(Var.of(tokens(test))).value().reshape(256 * LENGTH, VOCAB));
        float[] truth = targets(test).toArray();
        int hit = 0;
        int total = 0;
        for (int i = 0; i < truth.length; i++) {
            if (truth[i] >= 0) {
                total++;
                hit += pred[i] == (int) truth[i] ? 1 : 0;
            }
        }
        return (double) hit / total;
    }

    private static int[] argmax(Tensor t) {
        int rows = t.shape().dim(0);
        int cols = t.shape().dim(1);
        float[] v = t.toArray();
        int[] out = new int[rows];
        for (int i = 0; i < rows; i++) {
            int best = 0;
            for (int c = 1; c < cols; c++) {
                if (v[i * cols + c] > v[i * cols + best]) {
                    best = c;
                }
            }
            out[i] = best;
        }
        return out;
    }
}
