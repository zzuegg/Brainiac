package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Op;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * The quantisation-clip-floor-shift (QCFS) activation of Bu et al. (ICLR 2022, "Optimal
 * ANN-SNN Conversion for High-accuracy and Ultra-low-latency Spiking Neural Networks"):
 *
 * <pre>
 * y = λ / L · clip(floor(x · L / λ + 1/2), 0, L)
 * </pre>
 *
 * a ReLU replacement whose output takes the {@code L + 1} values an integrate-and-fire neuron
 * with threshold {@code λ} and initial membrane {@code λ/2} can produce in {@code L} steps, so
 * an ANN trained with it converts to an SNN that is accurate at {@code T ≈ L}. The threshold
 * {@code λ} is trainable. Gradients are straight-through: {@code ∂y/∂x = 1} on
 * {@code (0, λ)} and 0 outside, {@code ∂y/∂λ = 1} where {@code x ≥ λ} (the clipped region).
 * {@link SnnConverter} uses {@link #threshold()} as the layer's firing threshold.
 */
public final class QuantClipFloorActivation implements Layer {
    private final int levels;
    private final Var lambda;

    private QuantClipFloorActivation(int levels, float initial) {
        if (levels < 1 || !(initial > 0)) {
            throw new IllegalArgumentException("need levels >= 1 and a positive threshold");
        }
        this.levels = levels;
        this.lambda = Var.param(Tensor.of(Shape.of(1), initial));
    }

    /**
     * Creates the activation.
     *
     * @param levels the quantisation steps {@code L}, matched to the SNN's time steps
     * @param initial the initial threshold {@code λ}
     * @return the layer
     */
    public static QuantClipFloorActivation of(int levels, float initial) {
        return new QuantClipFloorActivation(levels, initial);
    }

    /**
     * Describes the activation for model builders.
     *
     * @param levels the quantisation steps
     * @param initial the initial threshold
     * @return the spec
     */
    public static LayerSpec spec(int levels, float initial) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new QuantClipFloorActivation(levels, initial);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        return Var.apply(new Qcfs(levels), input, lambda);
    }

    @Override
    public List<Var> parameters() {
        return List.of(lambda);
    }

    /** @return the current threshold {@code λ} */
    public float threshold() {
        return lambda.value().item();
    }

    /** @return the quantisation steps {@code L} */
    public int levels() {
        return levels;
    }

    /** The QCFS operation on {@code (x, λ)}. */
    private record Qcfs(int levels) implements Op {

        @Override
        public Tensor forward(Tensor... in) {
            float l = Math.max(in[1].item(), 1e-6f);
            float[] x = in[0].toArray();
            float[] y = new float[x.length];
            for (int i = 0; i < x.length; i++) {
                float q = (float) Math.floor(x[i] * levels / l + 0.5f);
                y[i] = l / levels * Math.max(0, Math.min(levels, q));
            }
            return Tensor.of(in[0].shape(), y);
        }

        @Override
        public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
            float l = in[1].item();
            float[] x = in[0].toArray();
            float[] g = grad.toArray();
            float[] gx = new float[x.length];
            double gl = 0;
            for (int i = 0; i < x.length; i++) {
                if (x[i] >= l) {
                    gl += g[i];
                } else if (x[i] > 0) {
                    gx[i] = g[i];
                }
            }
            return new Tensor[] {Tensor.of(in[0].shape(), gx), Tensor.of(Shape.of(1), (float) gl)};
        }

        @Override
        public String name() {
            return "qcfs";
        }
    }
}
