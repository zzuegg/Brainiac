package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Adam (AdamW with a weight decay) that keeps its two averages as 8-bit levels with a scale
 * per block (Dettmers et al. 2022): a quarter of float32 Adam's state memory. Each step
 * computes the averages in float32 from the stored levels, steps exactly as Adam would, and
 * quantizes them again on the device.
 *
 * <p>The second average is stored as its square root, which halves its range of exponents so
 * eight bits keep small entries. Where an entry still rounds to zero, the step divides by
 * {@code max(sqrt(v), |m|)} instead of {@code sqrt(v)}: the same in exact arithmetic, where
 * {@code |m| <= sqrt(v)}, and never larger than the learning rate. Parameters with fewer than
 * 4096 values, or whose size no block layout fits, keep float32 state as bitsandbytes does.
 */
final class Adam8bit extends StatefulOptimizer {
    private static final int MIN_SIZE = 4096;

    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float weightDecay;

    Adam8bit(float learningRate, float beta1, float beta2, float epsilon, float weightDecay) {
        super(learningRate, "m", "sqrtV");
        if (beta1 < 0 || beta1 >= 1 || beta2 < 0 || beta2 >= 1 || epsilon < 0 || weightDecay < 0) {
            throw new IllegalArgumentException("bad Adam settings");
        }
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        float lr = learningRate();
        Shape shape = p.value().shape();
        float correction1 = 1 - (float) Math.pow(beta1, t);
        float correction2 = 1 - (float) Math.pow(beta2, t);
        Tensor m = s[0] == null ? g.mul(1 - beta1) : values(s[0], shape).mul(beta1).add(g.mul(1 - beta1));
        Tensor v = s[1] == null ? g.square().mul(1 - beta2)
                : values(s[1], shape).square().mul(beta2).add(g.square().mul(1 - beta2));
        Tensor r = v.sqrt();
        Tensor mHat = m.div(correction1);
        Tensor denominator = r.div((float) Math.sqrt(correction2)).maximum(mHat.abs()).add(epsilon);
        Tensor w = weightDecay > 0 ? p.value().mul(1 - lr * weightDecay) : p.value();
        p.assign(w.sub(mHat.div(denominator).mul(lr)));
        int rows = blockRows(shape.size());
        if (rows == 0) {
            return new Tensor[] {m, r};
        }
        int cols = Math.toIntExact(shape.size() / rows);
        // [rows, cols] with one scale per column: block j holds every cols-th value from j
        return new Tensor[] {m.reshape(rows, cols).quantized(8, rows), r.reshape(rows, cols).quantized(8, rows)};
    }

    private static Tensor values(Tensor state, Shape shape) {
        return (state.isPacked() ? state.unpacked() : state).reshape(shape);   // saved state: [rows, cols]
    }

    /** The values per block for a parameter of {@code size} values; 0 for float32 state. */
    static int blockRows(long size) {
        if (size < MIN_SIZE) {
            return 0;
        }
        for (int rows = 256; rows >= 16; rows /= 2) {
            if (size % (rows * 4L) == 0) {   // columns a multiple of 4: whole words on a GPU
                return rows;
            }
        }
        return 0;
    }

    @Override
    public String toString() {
        return "Adam8bit(lr=" + learningRate() + (weightDecay > 0 ? ", weightDecay=" + weightDecay : "") + ")";
    }
}
