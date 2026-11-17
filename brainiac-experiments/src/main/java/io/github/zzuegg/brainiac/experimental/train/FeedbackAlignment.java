package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * Trains a {@link SimpleMlp} without weight transport: the error reaches the hidden layers
 * through <em>fixed random</em> matrices instead of the transposed forward weights.
 *
 * <ul>
 *   <li>{@link Mode#FEEDBACK_ALIGNMENT} (Lillicrap et al. 2016,
 *       <a href="https://arxiv.org/abs/1411.0247">arXiv:1411.0247</a>): the backward pass
 *       is layer by layer as in backprop, but uses a random {@code B_l} in place of
 *       {@code W_l^T}. The forward weights learn to align with {@code B}.</li>
 *   <li>{@link Mode#DIRECT_FEEDBACK_ALIGNMENT} (Nøkland 2016,
 *       <a href="https://arxiv.org/abs/1609.01596">arXiv:1609.01596</a>): the output error
 *       goes straight to every hidden layer through its own random matrix, so all layers
 *       can update in parallel as soon as the output error is known.</li>
 *   <li>{@link Mode#SIGN_SYMMETRY} (Liao et al. 2016,
 *       <a href="https://arxiv.org/abs/1510.05067">arXiv:1510.05067</a>): the backward
 *       weights are {@code sign(W^T)}: only the signs are transported.</li>
 * </ul>
 *
 * <p>The local weight gradients are written into the parameters' {@code grad}, so any core
 * {@link Optimizer} applies them.
 */
public final class FeedbackAlignment implements LocalLearner {

    /** How the error reaches hidden layers. */
    public enum Mode {
        /** layer by layer through fixed random matrices */
        FEEDBACK_ALIGNMENT,
        /** from the output straight to every layer through fixed random matrices */
        DIRECT_FEEDBACK_ALIGNMENT,
        /** layer by layer through the signs of the forward weights */
        SIGN_SYMMETRY
    }

    private final SimpleMlp net;
    private final Optimizer optimizer;
    private final Mode mode;
    private final Tensor[] feedback;

    /**
     * @param net the network to train
     * @param optimizer the optimizer that applies the local gradients
     * @param mode how errors are sent back
     * @param seed the seed of the random feedback matrices
     */
    public FeedbackAlignment(SimpleMlp net, Optimizer optimizer, Mode mode, long seed) {
        this.net = net;
        this.optimizer = optimizer;
        this.mode = mode;
        int[] sizes = net.sizes();
        int classes = sizes[sizes.length - 1];
        SplittableRandom r = new SplittableRandom(seed);
        feedback = new Tensor[net.depth()];
        for (int l = 1; l < net.depth(); l++) {
            // feedback[l] carries the error of layer l's output back to its input (size sizes[l])
            int from = mode == Mode.DIRECT_FEEDBACK_ALIGNMENT ? classes : sizes[l + 1];
            feedback[l] = Tensor.normal(Shape.of(from, sizes[l]), 0, (float) (1 / Math.sqrt(from)), r);
        }
    }

    /** @return the network */
    public SimpleMlp network() {
        return net;
    }

    /**
     * How well the forward weights have aligned with the feedback: the cosine between
     * {@code W_l^T} and {@code B_l} for layer {@code l >= 1} (feedback alignment only).
     *
     * @param l the layer, from 1
     * @return the cosine similarity
     */
    public double alignment(int l) {
        return LearningMath.cosine(net.weight(l).value().transpose().toArray(), feedback[l].toArray());
    }

    @Override
    public float trainBatch(Tensor inputs, Tensor labels) {
        int depth = net.depth();
        int batch = inputs.shape().dim(0);
        Tensor[] h = net.activations(inputs);
        Tensor logits = h[depth];
        Tensor p = LearningMath.softmax(logits);
        Tensor y = LearningMath.oneHot(labels, logits.shape().dim(1));
        float loss = -p.add(1e-7f).log().mul(y).sum().item() / batch;
        Tensor error = p.sub(y).div(batch);   // d loss / d logits
        Tensor delta = error;                 // gradient at the current layer's output
        for (int l = depth - 1; l >= 0; l--) {
            Tensor deltaPre = l == depth - 1 ? delta : activationGrad(h[l], l, delta);
            Var in = Var.of(h[l]);
            net.linear(l, in).backward(deltaPre);   // fills W_l.grad and b_l.grad locally
            if (l > 0) {
                delta = switch (mode) {
                    case FEEDBACK_ALIGNMENT -> deltaPre.matmul(feedback[l]);
                    case DIRECT_FEEDBACK_ALIGNMENT -> error.matmul(feedback[l]);
                    case SIGN_SYMMETRY -> deltaPre.matmul(net.weight(l).value().transpose().sign());
                };
            }
        }
        optimizer.step(net.parameters());
        return loss;
    }

    /** The gradient at layer l's pre-activation from the gradient at its output. */
    private Tensor activationGrad(Tensor input, int l, Tensor gradOut) {
        Var pre = Var.param(net.linear(l, Var.of(input)).value());
        net.activation().apply(pre).backward(gradOut);
        return pre.grad();
    }

    @Override
    public Tensor predict(Tensor inputs) {
        return net.predict(inputs);
    }
}
