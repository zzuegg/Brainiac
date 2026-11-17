package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A capsule layer with dynamic routing by agreement (Sabour, Frosst and Hinton,
 * <a href="https://arxiv.org/abs/1710.09829">1710.09829</a>). Each input capsule {@code i}
 * (a vector) predicts every output capsule through its own matrix,
 * {@code u_hat_(j|i) = W_ij u_i}; routing then repeatedly couples each input to the outputs
 * its prediction agrees with:
 *
 * <pre>
 * c_ij = softmax_j(b_ij);  s_j = sum_i c_ij u_hat_(j|i);  v_j = squash(s_j);  b_ij += u_hat_(j|i) . v_j
 * </pre>
 *
 * <p>{@code squash(s) = |s|^2 / (1 + |s|^2) s / |s|} keeps the direction and maps the length
 * into [0, 1), so an output's length reads as the probability that its entity is present.
 * Input {@code [batch, inputCapsules, inputDim]}, output {@code [batch, outputCapsules, outputDim]}.
 * {@link #MARGIN_LOSS} is the paper's loss on the output lengths.
 */
public final class CapsuleLayer implements Layer {
    /**
     * The margin loss on capsule lengths {@code [batch, classes]} against class ids
     * {@code [batch]} or one-hot targets: {@code T max(0, 0.9 - |v|)^2 + 0.5 (1 - T) max(0, |v| - 0.1)^2},
     * summed over classes and averaged over the batch.
     */
    public static final Loss MARGIN_LOSS = (lengths, target) -> {
        Tensor t = Loss.classIds(lengths.shape(), target.shape()) ? Loss.oneHot(target, lengths.shape()) : target;
        Var present = lengths.neg().add(0.9f).relu().square().mul(t);
        Var absent = lengths.sub(0.1f).relu().square().mul(t.neg().add(1)).mul(0.5f);
        return present.add(absent).sum().div(lengths.shape().dim(0));
    };

    private final int inputCapsules;
    private final int inputDim;
    private final int outputCapsules;
    private final int outputDim;
    private final int iterations;
    private final Var weight;
    private Tensor lastCoupling;

    private CapsuleLayer(int inputCapsules, int inputDim, int outputCapsules, int outputDim, int iterations,
            SplittableRandom random) {
        this.inputCapsules = inputCapsules;
        this.inputDim = inputDim;
        this.outputCapsules = outputCapsules;
        this.outputDim = outputDim;
        this.iterations = iterations;
        this.weight = Var.param(Tensor.normal(Shape.of(inputCapsules, inputDim, outputCapsules * outputDim), 0,
                (float) (1 / Math.sqrt(inputDim)), random));
    }

    /**
     * Creates a layer.
     *
     * @param inputCapsules the number of input capsules
     * @param inputDim their length
     * @param outputCapsules the number of output capsules
     * @param outputDim their length
     * @param iterations routing iterations, 3 in the paper
     * @param seed the seed
     * @return the layer
     */
    public static CapsuleLayer of(int inputCapsules, int inputDim, int outputCapsules, int outputDim, int iterations,
            long seed) {
        if (iterations < 1) {
            throw new IllegalArgumentException("need at least one routing iteration");
        }
        return new CapsuleLayer(inputCapsules, inputDim, outputCapsules, outputDim, iterations,
                new SplittableRandom(seed));
    }

    /**
     * Describes the layer for model builders; input {@code [inputCapsules, inputDim]}.
     *
     * @param outputCapsules the number of output capsules
     * @param outputDim their length
     * @param iterations routing iterations
     * @return the spec
     */
    public static LayerSpec spec(int outputCapsules, int outputDim, int iterations) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(outputCapsules, outputDim);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new CapsuleLayer(input.dim(0), input.dim(1), outputCapsules, outputDim, iterations, random);
            }
        };
    }

    /**
     * The squash nonlinearity along the last axis.
     *
     * @param s vectors
     * @return squashed vectors, length in [0, 1)
     */
    public static Var squash(Var s) {
        int last = s.shape().rank() - 1;
        Var n2 = s.square().sum(last, true);
        return s.mul(n2.div(n2.add(1)).div(n2.add(1e-9f).sqrt()));
    }

    /**
     * The length of each capsule.
     *
     * @param capsules {@code [batch, capsules, dim]}
     * @return {@code [batch, capsules]}
     */
    public static Var lengths(Var capsules) {
        Shape s = capsules.shape();
        return capsules.square().sum(2, false).add(1e-9f).sqrt().reshape(s.dim(0), s.dim(1));
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.rank() != 3 || s.dim(1) != inputCapsules || s.dim(2) != inputDim) {
            throw new IllegalArgumentException("CapsuleLayer expects [batch, " + inputCapsules + ", " + inputDim
                    + "], got " + s);
        }
        int b = s.dim(0);
        // [b, in, 1, inDim] x [in, inDim, out*outDim] -> [b, in, 1, out*outDim]
        Var predictions = input.reshape(b, inputCapsules, 1, inputDim).matmul(weight)
                .reshape(b, inputCapsules, outputCapsules, outputDim);
        Var logits = Var.of(Tensor.zeros(Shape.of(b, inputCapsules, outputCapsules, 1), input.value().device()));
        Var v = null;
        for (int r = 0; r < iterations; r++) {
            Var e = logits.sub(Var.of(logits.value().max(2, true))).exp();
            Var c = e.div(e.sum(2, true));
            if (r == iterations - 1) {
                lastCoupling = c.value();
            }
            v = squash(c.mul(predictions).sum(1, true));                 // [b, 1, out, outDim]
            if (r < iterations - 1) {
                logits = logits.add(predictions.mul(v).sum(3, true));      // agreement
            }
        }
        return v.reshape(b, outputCapsules, outputDim);
    }

    /** @return the coupling coefficients of the last forward pass, {@code [batch, in, out, 1]} */
    public Tensor lastCoupling() {
        return lastCoupling;
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight);
    }

    @Override
    public int outputSize(int inputSize) {
        return outputCapsules * outputDim;
    }
}
