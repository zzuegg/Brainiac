package io.github.zzuegg.brainiac.experimental.autograd;

import io.github.zzuegg.brainiac.autograd.Op;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * The spike nonlinearity of spiking neural networks with a surrogate gradient (Neftci,
 * Mostafa and Zenke, <a href="https://arxiv.org/abs/1901.09948">1901.09948</a>; SuperSpike):
 * the forward pass is the Heaviside step {@code s = [v > 0]} (a spike when the membrane
 * potential crosses the threshold), whose true derivative is zero almost everywhere. The
 * backward pass pretends it was smooth, using the fast-sigmoid derivative
 * {@code 1 / (1 + k |v|)^2}, so backpropagation through time can train the network.
 */
public final class SurrogateSpike implements Op {
    private final float slope;

    private SurrogateSpike(float slope) {
        this.slope = slope;
    }

    /**
     * Creates the operation.
     *
     * @param slope the sharpness {@code k} of the surrogate; larger is closer to the step
     * @return the operation
     */
    public static SurrogateSpike of(float slope) {
        return new SurrogateSpike(slope);
    }

    /**
     * Spikes where {@code v > 0}.
     *
     * @param v the membrane potential minus the threshold
     * @param slope the surrogate sharpness
     * @return 0/1 spikes with surrogate gradients
     */
    public static Var spike(Var v, float slope) {
        return Var.apply(of(slope), v);
    }

    @Override
    public Tensor forward(Tensor... in) {
        return in[0].step();
    }

    @Override
    public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
        Tensor d = in[0].abs().mul(slope).add(1f).square();
        return new Tensor[] {grad.div(d)};
    }

    @Override
    public String name() {
        return "surrogate spike";
    }
}
