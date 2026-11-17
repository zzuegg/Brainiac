package io.github.zzuegg.brainiac.experimental.autograd;

import io.github.zzuegg.brainiac.autograd.Op;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Sine and cosine as differentiable operations, which {@link Var} lacks: needed by Fourier
 * features, phase masks of optical layers, periodic activations (SIREN) and rotations.
 * Computed element by element on the host, then moved back to the input's device.
 */
public final class Trig {
    private Trig() {
    }

    private static Tensor map(Tensor x, boolean sine) {
        float[] v = x.toArray();
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (sine ? Math.sin(v[i]) : Math.cos(v[i]));
        }
        return Tensor.of(x.shape(), v).to(x.device());
    }

    /** {@code sin(x)} with gradient {@code cos(x)}. */
    static final Op SIN = new Op() {
        @Override
        public Tensor forward(Tensor... in) {
            return map(in[0], true);
        }

        @Override
        public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
            return new Tensor[] {grad.mul(map(in[0], false))};
        }

        @Override
        public String name() {
            return "sin";
        }
    };

    /** {@code cos(x)} with gradient {@code -sin(x)}. */
    static final Op COS = new Op() {
        @Override
        public Tensor forward(Tensor... in) {
            return map(in[0], false);
        }

        @Override
        public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
            return new Tensor[] {grad.mul(map(in[0], true)).neg()};
        }

        @Override
        public String name() {
            return "cos";
        }
    };

    /**
     * Element-wise sine.
     *
     * @param x the input
     * @return {@code sin(x)}
     */
    public static Var sin(Var x) {
        return Var.apply(SIN, x);
    }

    /**
     * Element-wise cosine.
     *
     * @param x the input
     * @return {@code cos(x)}
     */
    public static Var cos(Var x) {
        return Var.apply(COS, x);
    }

    /** @return the sine operation, for {@code GradCheck} or {@code Var.apply} */
    public static Op sinOp() {
        return SIN;
    }

    /** @return the cosine operation */
    public static Op cosOp() {
        return COS;
    }
}
