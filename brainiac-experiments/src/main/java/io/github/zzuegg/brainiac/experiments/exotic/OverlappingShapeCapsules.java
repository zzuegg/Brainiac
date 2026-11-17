package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.CapsuleLayer;
import io.github.zzuegg.brainiac.experiments.generative.ToyImages;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The MultiMNIST idea of the capsule paper (Sabour et al., <a href="https://arxiv.org/abs/1710.09829">1710.09829</a>)
 * on 8x8 toy shapes: primary capsules from a dense layer, a {@link CapsuleLayer} with dynamic
 * routing to one capsule per shape class, and the margin loss on capsule lengths. Because
 * each class has its own length rather than a share of a softmax, the network trained on
 * single shapes and pairs reports both shapes of an overlay at once.
 */
@SuppressWarnings("exports")   // experimental.* packages are exported by the lead at integration
public final class OverlappingShapeCapsules {
    private static final int PRIMARY = 16;
    private static final int PRIMARY_DIM = 6;
    private final Dense primary;
    private final CapsuleLayer digits;
    private final Optimizer optimizer;
    private final SplittableRandom random;

    private OverlappingShapeCapsules(int routing, float learningRate, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.primary = new Dense(ToyImages.PIXELS, PRIMARY * PRIMARY_DIM, Activation.IDENTITY, Init.XAVIER, r);
        this.digits = CapsuleLayer.of(PRIMARY, PRIMARY_DIM, ToyImages.CLASSES, 8, routing, seed + 1);
        this.optimizer = Optimizer.adam(learningRate);
        this.random = r;
    }

    /**
     * Creates the experiment.
     *
     * @param routing the routing iterations
     * @param learningRate the Adam learning rate
     * @param seed the seed
     * @return the experiment
     */
    public static OverlappingShapeCapsules of(int routing, float learningRate, long seed) {
        return new OverlappingShapeCapsules(routing, learningRate, seed);
    }

    /**
     * Draws an image of one shape, or the pixel-wise maximum of two shapes of different classes.
     *
     * @param pair whether to overlay two shapes
     * @param random the random source
     * @param classes receives the class ids
     * @return the image in {-1, +1}
     */
    public static float[] image(boolean pair, SplittableRandom random, List<Integer> classes) {
        int a = random.nextInt(ToyImages.CLASSES);
        float[] x = ToyImages.sample(a, random);
        classes.add(a);
        if (pair) {
            int b = (a + 1 + random.nextInt(ToyImages.CLASSES - 1)) % ToyImages.CLASSES;
            float[] y = ToyImages.sample(b, random);
            for (int i = 0; i < x.length; i++) {
                x[i] = Math.max(x[i], y[i]);
            }
            classes.add(b);
        }
        return x;
    }

    /**
     * Capsule lengths for images.
     *
     * @param images {@code [batch, 64]}
     * @return {@code [batch, classes]}
     */
    public Var lengths(Tensor images) {
        int b = images.shape().dim(0);
        Var u = CapsuleLayer.squash(primary.forward(Var.of(images)).reshape(b, PRIMARY, PRIMARY_DIM));
        return CapsuleLayer.lengths(digits.forward(u));
    }

    /** @return the routing layer, to inspect its couplings */
    public CapsuleLayer capsules() {
        return digits;
    }

    /**
     * One training step on a batch with half single shapes and half pairs.
     *
     * @param batch the batch size
     * @return the margin loss
     */
    public float trainStep(int batch) {
        float[] x = new float[batch * ToyImages.PIXELS];
        float[] t = new float[batch * ToyImages.CLASSES];
        for (int n = 0; n < batch; n++) {
            List<Integer> c = new ArrayList<>();
            System.arraycopy(image(n % 2 == 1, random, c), 0, x, n * ToyImages.PIXELS, ToyImages.PIXELS);
            for (int k : c) {
                t[n * ToyImages.CLASSES + k] = 1;
            }
        }
        Var loss = CapsuleLayer.MARGIN_LOSS.compute(lengths(Tensor.of(Shape.of(batch, ToyImages.PIXELS), x)),
                Tensor.of(Shape.of(batch, ToyImages.CLASSES), t));
        float value = loss.value().item();
        loss.backward();
        List<Var> params = new ArrayList<>(primary.parameters());
        params.addAll(digits.parameters());
        optimizer.step(params);
        return value;
    }

    /**
     * Accuracy: for single shapes the longest capsule must be the class; for pairs the two
     * longest must be the two classes.
     *
     * @param pairs whether to test on overlays
     * @param n the number of test images
     * @param seed the test seed
     * @return the share correct
     */
    public double accuracy(boolean pairs, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] x = new float[n * ToyImages.PIXELS];
        List<List<Integer>> truth = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            List<Integer> c = new ArrayList<>();
            System.arraycopy(image(pairs, r, c), 0, x, i * ToyImages.PIXELS, ToyImages.PIXELS);
            truth.add(c);
        }
        float[] len = lengths(Tensor.of(Shape.of(n, ToyImages.PIXELS), x)).value().toArray();
        int correct = 0;
        for (int i = 0; i < n; i++) {
            Integer[] order = {0, 1, 2, 3};
            int row = i;
            java.util.Arrays.sort(order, (a, b) -> Float.compare(len[row * 4 + b], len[row * 4 + a]));
            List<Integer> top = List.of(order).subList(0, truth.get(i).size());
            if (top.containsAll(truth.get(i))) {
                correct++;
            }
        }
        return (double) correct / n;
    }
}
