package io.github.zzuegg.brainiac.experiments.compression;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.compress.NetworkSlimming;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The small shapes CNN the compression and kernel experiments share: on {@link ShapeImages}
 * ({@code 8x8}, six classes), {@code conv3x3(1 -> c1) - BN - ReLU - conv3x3/2(c1 -> c2) - BN - ReLU -
 * global mean - linear(c2 -> 6)}, with every weight a plain {@link Var} so channels can be removed
 * (network slimming), layers decomposed (Tucker) and the convolution swapped for another
 * algorithm (Winograd). Batch norm always uses the statistics of the batch it sees, which is the
 * whole training or test set here.
 */
public final class ShapesCnn {

    /** Image side. */
    public static final int SIDE = 8;

    /** How a layer convolves: input {@code [n, c, h, w]}, filters {@code [f, c, 3, 3]}, padding 1. */
    @FunctionalInterface
    public interface Convolution {
        /**
         * @param x the input
         * @param weight the filters
         * @param stride the stride
         * @return the output {@code [n, f, h', w']}
         */
        Var apply(Var x, Var weight, int stride);
    }

    /** The core im2col convolution. */
    public static final Convolution CORE = (x, w, stride) -> Functional.conv(x, w, null, new int[] {stride, stride},
            new int[] {1, 1}, new int[] {1, 1}, new int[] {1, 1}, 1);

    private final Var w1;
    private final Var g1;
    private final Var b1;
    private final Var w2;
    private final Var g2;
    private final Var b2;
    private final Var head;
    private final Var headBias;

    /**
     * Creates the network from weights.
     *
     * @param w1 {@code [c1, 1, 3, 3]}
     * @param g1 batch-norm scales {@code [c1]}
     * @param b1 batch-norm shifts {@code [c1]}
     * @param w2 {@code [c2, c1, 3, 3]}
     * @param g2 batch-norm scales {@code [c2]}
     * @param b2 batch-norm shifts {@code [c2]}
     * @param head {@code [c2, 6]}
     * @param headBias {@code [1, 6]}
     */
    public ShapesCnn(Tensor w1, Tensor g1, Tensor b1, Tensor w2, Tensor g2, Tensor b2, Tensor head, Tensor headBias) {
        this.w1 = Var.param(w1);
        this.g1 = Var.param(g1);
        this.b1 = Var.param(b1);
        this.w2 = Var.param(w2);
        this.g2 = Var.param(g2);
        this.b2 = Var.param(b2);
        this.head = Var.param(head);
        this.headBias = Var.param(headBias);
    }

    /**
     * @param c1 first-layer channels
     * @param c2 second-layer channels
     * @param seed the seed
     * @return a He-initialised network
     */
    public static ShapesCnn create(int c1, int c2, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        return new ShapesCnn(Tensor.normal(Shape.of(c1, 1, 3, 3), 0, (float) Math.sqrt(2.0 / 9), r),
                Tensor.ones(Shape.of(c1)), Tensor.zeros(Shape.of(c1)),
                Tensor.normal(Shape.of(c2, c1, 3, 3), 0, (float) Math.sqrt(2.0 / (9 * c1)), r),
                Tensor.ones(Shape.of(c2)), Tensor.zeros(Shape.of(c2)),
                Tensor.normal(Shape.of(c2, ShapeImages.CLASSES), 0, (float) Math.sqrt(1.0 / c2), r),
                Tensor.zeros(Shape.of(1, ShapeImages.CLASSES)));
    }

    /**
     * @param n images
     * @param seed the seed
     * @return {@code {images [n, 1, 8, 8], ids [n]}}
     */
    public static Tensor[] data(int n, long seed) {
        return new ShapeImages(SIDE, 0.15f).generate(n, seed);
    }

    /** @return the first convolution's filters */
    public Var w1() {
        return w1;
    }

    /** @return the second convolution's filters */
    public Var w2() {
        return w2;
    }

    /** @return the two batch-norm scale vectors */
    public List<Var> gammas() {
        return List.of(g1, g2);
    }

    /** @return every parameter */
    public List<Var> parameters() {
        return List.of(w1, g1, b1, w2, g2, b2, head, headBias);
    }

    /** @return {@code {c1, c2}} */
    public int[] channels() {
        return new int[] {w1.shape().dim(0), w2.shape().dim(0)};
    }

    /**
     * @param x images
     * @return logits with the core convolution
     */
    public Var forward(Var x) {
        return forward(x, CORE, CORE);
    }

    /**
     * The forward pass with any convolution algorithm per layer.
     *
     * @param x images {@code [n, 1, 8, 8]}
     * @param conv1 the first layer's convolution
     * @param conv2 the second layer's convolution
     * @return logits {@code [n, 6]}
     */
    public Var forward(Var x, Convolution conv1, Convolution conv2) {
        Var h = batchNorm(conv1.apply(x, w1, 1), g1, b1).relu();
        h = batchNorm(conv2.apply(h, w2, 2), g2, b2).relu();
        int n = h.shape().dim(0);
        int c = h.shape().dim(1);
        Var pooled = h.reshape(n, c, h.shape().dim(2) * h.shape().dim(3)).mean(2, false);
        return pooled.matmul(head).add(headBias);
    }

    /**
     * Batch normalisation with batch statistics.
     *
     * @param x {@code [n, c, h, w]}
     * @param gamma scales {@code [c]}
     * @param beta shifts {@code [c]}
     * @return the normalised activations
     */
    public static Var batchNorm(Var x, Var gamma, Var beta) {
        int c = x.shape().dim(1);
        Var mean = x.mean(0, true).mean(2, true).mean(3, true);
        Var centred = x.sub(mean);
        Var var = centred.square().mean(0, true).mean(2, true).mean(3, true);
        return centred.div(var.add(1e-5f).sqrt()).mul(gamma.reshape(1, c, 1, 1)).add(beta.reshape(1, c, 1, 1));
    }

    /**
     * Full-batch Adam training, optionally with an L1 penalty on the batch-norm scales.
     *
     * @param data {@code {images, ids}}
     * @param optimizer the optimizer
     * @param steps the steps
     * @param l1 the L1 weight on every {@code |gamma|}; 0 for none
     */
    public void train(Tensor[] data, Optimizer optimizer, int steps, float l1) {
        List<Var> params = parameters();
        for (int s = 0; s < steps; s++) {
            Var loss = Functional.crossEntropy(forward(Var.of(data[0])), data[1]);
            if (l1 > 0) {
                loss = loss.add(NetworkSlimming.l1Penalty(gammas(), l1));
            }
            loss.backward();
            optimizer.step(params);
        }
    }

    /**
     * Full-batch training with a replacement for the second convolution (a decomposed layer),
     * whose own parameters are trained instead of {@link #w2()}.
     *
     * @param data {@code {images, ids}}
     * @param optimizer the optimizer
     * @param steps the steps
     * @param conv2 the replacement convolution
     * @param conv2Parameters its parameters
     */
    public void train(Tensor[] data, Optimizer optimizer, int steps, Convolution conv2, List<Var> conv2Parameters) {
        List<Var> params = new ArrayList<>(List.of(w1, g1, b1, g2, b2, head, headBias));
        params.addAll(conv2Parameters);
        for (int s = 0; s < steps; s++) {
            Functional.crossEntropy(forward(Var.of(data[0]), CORE, conv2), data[1]).backward();
            optimizer.step(params);
        }
    }

    /**
     * @param data {@code {images, ids}}
     * @param conv1 the first convolution
     * @param conv2 the second convolution
     * @return the accuracy
     */
    public double accuracy(Tensor[] data, Convolution conv1, Convolution conv2) {
        Tensor logits = Var.noGrad(() -> forward(Var.of(data[0]), conv1, conv2).value());
        float[] l = logits.toArray();
        float[] ids = data[1].toArray();
        int k = ShapeImages.CLASSES;
        int correct = 0;
        for (int i = 0; i < ids.length; i++) {
            int best = 0;
            for (int c = 1; c < k; c++) {
                if (l[i * k + c] > l[i * k + best]) {
                    best = c;
                }
            }
            correct += best == (int) ids[i] ? 1 : 0;
        }
        return correct / (double) ids.length;
    }

    /**
     * @param data {@code {images, ids}}
     * @return the accuracy with the core convolution
     */
    public double accuracy(Tensor[] data) {
        return accuracy(data, CORE, CORE);
    }

    /** @return multiply-adds per image */
    public long macs() {
        int c1 = channels()[0];
        int c2 = channels()[1];
        return (long) SIDE * SIDE * c1 * 9 + (long) (SIDE / 2) * (SIDE / 2) * c2 * c1 * 9 + (long) c2
                * ShapeImages.CLASSES;
    }

    /** @return the parameter count */
    public long parameterCount() {
        long n = 0;
        for (Var p : parameters()) {
            n += p.shape().size();
        }
        return n;
    }

    /**
     * @param images a batch
     * @return the median forward time in milliseconds
     */
    public double latencyMillis(Tensor images) {
        return io.github.zzuegg.brainiac.experimental.kernels.KernelTimer.millis(
                () -> Var.noGrad(() -> forward(Var.of(images)).value().toArray()), 3, 7);
    }

    /**
     * @return a copy with the same weights
     */
    public ShapesCnn copy() {
        List<Tensor> v = new ArrayList<>();
        for (Var p : parameters()) {
            v.add(p.value());
        }
        return new ShapesCnn(v.get(0), v.get(1), v.get(2), v.get(3), v.get(4), v.get(5), v.get(6), v.get(7));
    }
}
