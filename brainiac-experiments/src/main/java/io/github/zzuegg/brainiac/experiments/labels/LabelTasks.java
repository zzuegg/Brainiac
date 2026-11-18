package io.github.zzuegg.brainiac.experiments.labels;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Augmentation;
import io.github.zzuegg.brainiac.experimental.semisup.SoftTargets;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * The shared harness of the imperfect-supervision experiments: the digits task (the
 * {@code ToyTasks} MNIST-like prototypes, held-out samples drawn from the same prototypes), a
 * small two-part MLP (a trunk whose features some methods regularise, and a linear head),
 * minibatch sampling and scoring.
 */
public final class LabelTasks {

    private LabelTasks() {
    }

    /** Side of the digit images. */
    public static final int SIDE = 8;

    /**
     * A train/test pair of the digits task.
     *
     * @param train the training rows with clean labels
     * @param test the balanced test rows
     */
    public record Task(ToyTasks.Data train, ToyTasks.Data test) {
    }

    /**
     * The digits task: ten random prototypes, pixel flips with probability {@code noise}, shifts
     * of up to one pixel.
     *
     * @param classes the number of classes
     * @param trainPerClass training rows per class
     * @param testPerClass test rows per class
     * @param noise pixel flip probability
     * @param seed the prototype seed
     * @return the task
     */
    public static Task digits(int classes, int trainPerClass, int testPerClass, double noise, long seed) {
        return new Task(ToyTasks.digits(classes, trainPerClass, SIDE, noise, seed, seed * 31 + 7),
                ToyTasks.digits(classes, testPerClass, SIDE, noise, seed, seed * 31 + 1009));
    }

    /**
     * A trunk-and-head MLP.
     *
     * @param trunk inputs to features
     * @param head features to logits
     */
    public record Net(Model trunk, Model head) {

        /**
         * @param x inputs
         * @param training whether in training mode
         * @return the logits
         */
        public Var logits(Var x, boolean training) {
            return head.forward(trunk.forward(x, training), training);
        }

        /**
         * @param x inputs
         * @return the logits as a tensor, without gradient
         */
        public Tensor predict(Tensor x) {
            return Var.noGrad(() -> logits(Var.of(x), false).value());
        }

        /** @return all parameters */
        public List<Var> parameters() {
            List<Var> out = new java.util.ArrayList<>(trunk.parameters());
            out.addAll(head.parameters());
            return out;
        }
    }

    /**
     * Builds the MLP {@code inputs -> hidden (ReLU) -> features (ReLU) -> classes}.
     *
     * @param inputs input width
     * @param hidden hidden width
     * @param features feature width
     * @param classes classes
     * @param seed the initialisation seed
     * @return the network
     */
    public static Net mlp(int inputs, int hidden, int features, int classes, long seed) {
        Model trunk = Model.sequential().input(inputs).dense(hidden, Activation.RELU).dense(features, Activation.RELU)
                .seed(seed).build();
        Model head = Model.sequential().input(features).dense(classes, Activation.IDENTITY).seed(seed + 1).build();
        return new Net(trunk, head);
    }

    /**
     * @param n how many
     * @param from the pool of row indices
     * @param r the random source
     * @return {@code n} indices drawn with replacement from {@code from}
     */
    public static int[] draw(int n, int[] from, SplittableRandom r) {
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            out[i] = from[r.nextInt(from.length)];
        }
        return out;
    }

    /**
     * @param x rows
     * @param idx indices
     * @return those rows as a tensor
     */
    public static Tensor rows(float[][] x, int[] idx) {
        float[][] out = new float[idx.length][];
        for (int i = 0; i < idx.length; i++) {
            out[i] = x[idx[i]];
        }
        return Tensor.of(out);
    }

    /**
     * @param y labels
     * @param idx indices
     * @return the labels at those indices
     */
    public static int[] labels(int[] y, int[] idx) {
        int[] out = new int[idx.length];
        for (int i = 0; i < idx.length; i++) {
            out[i] = y[idx[i]];
        }
        return out;
    }

    /**
     * @param n the size
     * @return {@code 0..n-1}
     */
    public static int[] range(int n) {
        int[] out = new int[n];
        for (int i = 0; i < n; i++) {
            out[i] = i;
        }
        return out;
    }

    /**
     * @param net the network
     * @param data the test data
     * @return accuracy
     */
    public static double accuracy(Net net, ToyTasks.Data data) {
        return ToyTasks.accuracy(net.predict(data.inputs()), data.y());
    }

    /**
     * Per-class accuracy.
     *
     * @param logits {@code [n, classes]} scores
     * @param y labels
     * @param classes classes
     * @return each class's recall
     */
    public static double[] classAccuracy(Tensor logits, int[] y, int classes) {
        int[] pred = SoftTargets.argmax(SoftTargets.probabilities(logits, 1f));
        double[] hit = new double[classes];
        double[] cnt = new double[classes];
        for (int i = 0; i < y.length; i++) {
            cnt[y[i]]++;
            hit[y[i]] += pred[i] == y[i] ? 1 : 0;
        }
        for (int c = 0; c < classes; c++) {
            hit[c] /= Math.max(1, cnt[c]);
        }
        return hit;
    }

    /**
     * @param values numbers
     * @return their mean
     */
    public static double mean(double[] values) {
        double s = 0;
        for (double v : values) {
            s += v;
        }
        return s / values.length;
    }

    /**
     * @param values numbers
     * @return their minimum
     */
    public static double min(double[] values) {
        double m = Double.POSITIVE_INFINITY;
        for (double v : values) {
            m = Math.min(m, v);
        }
        return m;
    }

    /**
     * The weak augmentation of the digit images: a wrap-around shift of up to one pixel and a
     * little Gaussian pixel noise (no flips: a mirrored prototype is a different pattern).
     *
     * @return the augmentation
     */
    public static Augmentation weak() {
        return (f, offset, r) -> {
            int dx = r.nextInt(3) - 1;
            int dy = r.nextInt(3) - 1;
            float[] img = new float[SIDE * SIDE];
            for (int y = 0; y < SIDE; y++) {
                for (int x = 0; x < SIDE; x++) {
                    img[y * SIDE + x] = f[offset + Math.floorMod(y - dy, SIDE) * SIDE + Math.floorMod(x - dx, SIDE)];
                }
            }
            for (int i = 0; i < img.length; i++) {
                f[offset + i] = img[i] + (float) (0.05 * r.nextGaussian());
            }
        };
    }

    /**
     * The strong augmentation: the weak one, then 15% random pixel flips, then Cutout (core
     * {@code Augmentation.randomErasing} of 5-20% of the image) and a random contrast change.
     *
     * @return the augmentation
     */
    public static Augmentation strong() {
        Augmentation cut = Augmentation.randomErasing(1, SIDE, SIDE, 1.0, 0.05, 0.2);
        Augmentation flips = (f, offset, r) -> {
            float scale = (float) (0.6 + 0.8 * r.nextDouble());
            for (int i = 0; i < SIDE * SIDE; i++) {
                if (r.nextDouble() < 0.15) {
                    f[offset + i] = 1 - f[offset + i];
                }
                f[offset + i] *= scale;
            }
        };
        return weak().then(flips).then(cut);
    }

    /**
     * Adds Gaussian noise to a batch (the Π-model's augmentation for tabular rows).
     *
     * @param x the batch
     * @param sigma the standard deviation
     * @param r the random source
     * @return {@code x + sigma * N(0, 1)}
     */
    public static Tensor gaussian(Tensor x, double sigma, SplittableRandom r) {
        float[] v = x.toArray();
        for (int i = 0; i < v.length; i++) {
            v[i] += (float) (sigma * r.nextGaussian());
        }
        return Tensor.of(x.shape(), v);
    }
}
