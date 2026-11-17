package io.github.zzuegg.brainiac.experimental.nn.diffusion;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.diffusion.Denoiser;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A core {@link Denoiser.Conditional} whose condition is a real vector per example: an
 * observation for a robot policy (diffusion policy, a pi0-style flow action head), an
 * observation and action for a diffusion world model, or nothing at all. It is an MLP over
 * the noisy sample, sinusoidal features of the time and the condition, with a residual
 * connection around each hidden layer, the same body as {@link MlpDenoiser} but for vector
 * conditions instead of class labels.
 *
 * <p>{@link #applyWithHidden} also returns a hidden state, for representation alignment
 * (REPA, {@link RepresentationAlignment}).
 *
 * <pre>{@code
 * VectorConditionedDenoiser net = VectorConditionedDenoiser.of(16, 2, 128, 2, 1);   // 8x2 action chunk, 2-D obs
 * Var loss = FlowMatching.loss(net.given(observations), chunks, random);
 * }</pre>
 */
public final class VectorConditionedDenoiser implements Denoiser.Conditional {
    private static final int TIME_FEATURES = 16;
    private final int dimension;
    private final int conditionSize;
    private final int width;
    private final Dense in;
    private final List<Dense> hidden = new ArrayList<>();
    private final Dense out;
    private float timeScale = 1;

    private VectorConditionedDenoiser(int dimension, int conditionSize, int width, int depth, long seed) {
        if (dimension < 1 || conditionSize < 0 || width < 1 || depth < 0) {
            throw new IllegalArgumentException("bad VectorConditionedDenoiser size");
        }
        SplittableRandom random = new SplittableRandom(seed);
        this.dimension = dimension;
        this.conditionSize = conditionSize;
        this.width = width;
        in = GenBlocks.linear(dimension + TIME_FEATURES + conditionSize, width, random);
        for (int i = 0; i < depth; i++) {
            hidden.add(GenBlocks.linear(width, width, random));
        }
        out = GenBlocks.linear(width, dimension, random);
    }

    /**
     * Creates the network.
     *
     * @param dimension the sample size
     * @param conditionSize the condition size, 0 for an unconditional network
     * @param width the hidden width
     * @param depth the number of residual hidden layers
     * @param seed the initialization seed
     * @return the network
     */
    public static VectorConditionedDenoiser of(int dimension, int conditionSize, int width, int depth, long seed) {
        return new VectorConditionedDenoiser(dimension, conditionSize, width, depth, seed);
    }

    /**
     * Sets the factor applied to incoming times so they land in about [0, 1]: {@code 1 / steps}
     * for core {@code Diffusion}, 1 (the default) for flow matching.
     *
     * @param scale the factor
     * @return this network
     */
    public VectorConditionedDenoiser timeScale(float scale) {
        this.timeScale = scale;
        return this;
    }

    @Override
    public Var apply(Var noisy, Tensor time, Tensor condition) {
        return applyWithHidden(noisy, time, condition == null ? null : Var.of(condition), -1)[0];
    }

    /**
     * Runs the network with a condition that may carry gradients (for example an action
     * proposed by an actor), and hands out one hidden state.
     *
     * @param noisy the noisy data {@code [batch, dimension]}
     * @param time per example {@code [batch]}
     * @param condition {@code [batch, conditionSize]}, or {@code null} when the size is 0
     * @param layer which hidden state to return: 0 after the input layer, {@code k} after the
     *     k-th residual layer, -1 for none
     * @return {@code {prediction, hidden}} (hidden {@code [batch, width]} or {@code null})
     */
    public Var[] applyWithHidden(Var noisy, Tensor time, Var condition, int layer) {
        float[] t = time.toArray();
        for (int i = 0; i < t.length; i++) {
            t[i] *= timeScale;
        }
        Var features = Var.concat(1, noisy, Var.of(GenBlocks.timeEmbedding(t, TIME_FEATURES)
                .to(noisy.value().device())));
        if (conditionSize > 0) {
            if (condition == null) {
                throw new IllegalArgumentException("this denoiser needs a condition of size " + conditionSize);
            }
            features = Var.concat(1, features, condition);
        }
        Var h = GenBlocks.silu(in.forward(features));
        Var picked = layer == 0 ? h : null;
        for (int i = 0; i < hidden.size(); i++) {
            h = h.add(GenBlocks.silu(hidden.get(i).forward(h)));
            if (layer == i + 1) {
                picked = h;
            }
        }
        return new Var[] {out.forward(h), picked};
    }

    /** @return this network as a plain core denoiser, for a condition size of 0 */
    public Denoiser unconditional() {
        return (noisy, time) -> apply(noisy, time, null);
    }

    /**
     * Repeats one condition row for a whole batch.
     *
     * @param condition one condition
     * @param batch the batch size
     * @return {@code [batch, condition.length]}
     */
    public static Tensor repeat(float[] condition, int batch) {
        float[] flat = new float[batch * condition.length];
        for (int i = 0; i < batch; i++) {
            System.arraycopy(condition, 0, flat, i * condition.length, condition.length);
        }
        return Tensor.of(Shape.of(batch, condition.length), flat);
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(in.parameters());
        for (Dense l : hidden) {
            p.addAll(l.parameters());
        }
        p.addAll(out.parameters());
        return p;
    }

    /** @return the sample size */
    public int dimension() {
        return dimension;
    }

    /** @return the condition size */
    public int conditionSize() {
        return conditionSize;
    }

    /** @return the hidden width */
    public int width() {
        return width;
    }

    /** @return the number of residual hidden layers */
    public int depth() {
        return hidden.size();
    }
}
