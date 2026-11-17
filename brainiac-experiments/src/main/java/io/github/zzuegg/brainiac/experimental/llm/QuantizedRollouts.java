package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.List;
import java.util.function.UnaryOperator;

/**
 * A low-precision rollout engine for RL on language models: maps the current policy to a copy
 * whose weight matrices are round-to-nearest quantized with the core
 * {@link Tensor#quantized} (one scale per column), the way fast inference engines serve
 * rollouts in int8/int4/FP8 while the learner trains in full precision. The gap between the
 * two distributions makes RL silently off-policy; pass it to
 * {@link GrpoTrainer#behaviour} and measure or correct the mismatch with
 * {@link GrpoTrainer#truncatedImportance}.
 */
public final class QuantizedRollouts implements UnaryOperator<WindowLm> {

    private final int bits;

    /**
     * @param bits bits per weight, 2 to 8
     */
    public QuantizedRollouts(int bits) {
        if (bits < 2 || bits > 8) {
            throw new IllegalArgumentException("bits must be 2..8, got " + bits);
        }
        this.bits = bits;
    }

    /**
     * @param policy the full-precision policy (not changed)
     * @return a copy with quantized weight matrices (biases stay in full precision)
     */
    @Override
    public WindowLm apply(WindowLm policy) {
        WindowLm copy = policy.copy();
        List<Var> params = copy.parameters();
        for (Var p : params) {
            Tensor w = p.value();
            if (w.shape().rank() == 2) {
                float[] q = w.quantized(bits, w.shape().dim(0)).toArray();
                p.assign(Tensor.of(w.shape(), q));
            }
        }
        return copy;
    }
}
