package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Dynamic loss scaling for hand-written float16 training loops ({@code Trainer} has it
 * built in through {@code mixedPrecision}, {@code TrainingLoop} does not). Small float16
 * gradients underflow to zero; multiplying the loss by a large scale before
 * {@code backward()} keeps them representable. After the backward pass,
 * {@link #unscale(List)} divides the gradients by the scale and checks them: on overflow
 * (an infinite or NaN gradient) it clears them, halves the scale and reports that the step
 * must be skipped; after {@code growthInterval} good steps in a row it doubles the scale.
 *
 * <pre>{@code
 * Var loss = model.forward(x.to(DType.FLOAT16)).to(DType.FLOAT32) ...;
 * scaler.scale(loss).backward();
 * if (scaler.unscale(params)) optimizer.step(params);
 * }</pre>
 *
 * <p>Paper: Micikevicius et al., "Mixed Precision Training" (2017),
 * <a href="https://arxiv.org/abs/1710.03740">arXiv:1710.03740</a>.
 */
public final class DynamicLossScaler {

    private float scale;
    private final int growthInterval;
    private int goodSteps;
    private int skipped;

    /**
     * Creates the scaler.
     *
     * @param initialScale the first scale, such as 65536
     * @param growthInterval good steps before the scale doubles, such as 2000
     */
    public DynamicLossScaler(float initialScale, int growthInterval) {
        if (!(initialScale > 0) || growthInterval < 1) {
            throw new IllegalArgumentException("need a positive scale and growth interval");
        }
        this.scale = initialScale;
        this.growthInterval = growthInterval;
    }

    /**
     * Multiplies the loss by the current scale.
     *
     * @param loss the loss
     * @return the scaled loss, to call {@code backward()} on
     */
    public Var scale(Var loss) {
        return loss.mul(scale);
    }

    /**
     * Divides the gradients by the scale and updates the scale.
     *
     * @param parameters the parameters whose gradients to unscale
     * @return {@code true} if the gradients are finite and the optimizer may step;
     *     {@code false} if they overflowed, in which case they were cleared
     */
    public boolean unscale(List<Var> parameters) {
        boolean finite = true;
        for (Var p : parameters) {
            Tensor g = p.grad();
            if (g == null) {
                continue;
            }
            float s = g.abs().sum().item();
            if (!Float.isFinite(s)) {
                finite = false;
                break;
            }
        }
        if (!finite) {
            for (Var p : parameters) {
                p.zeroGrad();
            }
            scale = Math.max(1f, scale / 2);
            goodSteps = 0;
            skipped++;
            return false;
        }
        float inverse = 1f / scale;
        for (Var p : parameters) {
            Tensor g = p.grad();
            if (g != null) {
                p.setGrad(g.mul(inverse));
            }
        }
        if (++goodSteps >= growthInterval) {
            scale *= 2;
            goodSteps = 0;
        }
        return true;
    }

    /** @return the current scale */
    public float currentScale() {
        return scale;
    }

    /** @return the steps skipped because of overflow */
    public int skippedSteps() {
        return skipped;
    }
}
