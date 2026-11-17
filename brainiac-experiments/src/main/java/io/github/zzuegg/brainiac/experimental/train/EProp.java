package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * e-prop (Bellec et al. 2020, <a href="https://www.nature.com/articles/s41467-020-17236-y">Nat.
 * Commun. 11, 3625</a>): online learning for recurrent networks with per-synapse
 * <em>eligibility traces</em> instead of backpropagation through time. The network is a
 * leaky recurrent layer,
 *
 * <pre>
 *   c_t = alpha c_{t-1} + (1 - alpha)(x_t W_in + h_{t-1} W_rec + b),   h_t = tanh(c_t),
 *   y_t = h_t W_out + b_out
 * </pre>
 *
 * <p>Each synapse {@code i -> j} keeps the eligibility {@code e_ij = psi_j(t) * ebar_i(t)}:
 * the low-pass filtered presynaptic activity {@code ebar_i(t) = alpha ebar_i(t-1) +
 * (1 - alpha) pre_i(t)} times the postsynaptic derivative {@code psi_j = 1 - h_j^2}. A
 * learning signal {@code L_j(t)}, the output error sent back through {@code W_out^T}
 * (symmetric e-prop) or a fixed random matrix (random e-prop), gates it:
 * {@code dW_ij = sum_t L_j(t) e_ij(t)}. Memory does not grow with sequence length, and
 * updates are available at every step, which makes it suitable for agents that learn while
 * they act. {@link Mode#BPTT} trains the same network with exact backpropagation through
 * time for comparison.
 */
public final class EProp {

    /** How the recurrent weights learn. */
    public enum Mode {
        /** e-prop with the learning signal through {@code W_out^T} */
        SYMMETRIC,
        /** e-prop with a fixed random feedback matrix */
        RANDOM,
        /** exact backpropagation through time (baseline) */
        BPTT
    }

    private final int hidden;
    private final float alpha;
    private final Mode mode;
    private final Optimizer optimizer;
    private final Var wIn;
    private final Var wRec;
    private final Var b;
    private final Var wOut;
    private final Var bOut;
    private final Tensor feedback;

    /**
     * @param inputs input features per step
     * @param hidden recurrent units
     * @param outputs output features per step
     * @param alpha the leak, in {@code [0, 1)}: how much of the state is kept per step
     * @param mode the learning rule
     * @param optimizer the optimizer
     * @param seed the seed
     */
    public EProp(int inputs, int hidden, int outputs, float alpha, Mode mode, Optimizer optimizer, long seed) {
        this.hidden = hidden;
        this.alpha = alpha;
        this.mode = mode;
        this.optimizer = optimizer;
        SplittableRandom r = new SplittableRandom(seed);
        wIn = Var.param(Tensor.normal(Shape.of(inputs, hidden), 0, (float) Math.sqrt(1.0 / inputs), r));
        wRec = Var.param(Tensor.normal(Shape.of(hidden, hidden), 0, (float) Math.sqrt(1.0 / hidden), r));
        b = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        wOut = Var.param(Tensor.normal(Shape.of(hidden, outputs), 0, (float) Math.sqrt(1.0 / hidden), r));
        bOut = Var.param(Tensor.zeros(Shape.of(1, outputs)));
        feedback = Tensor.normal(Shape.of(outputs, hidden), 0, (float) Math.sqrt(1.0 / outputs), r);
    }

    /** @return all parameters */
    public List<Var> parameters() {
        return List.of(wIn, wRec, b, wOut, bOut);
    }

    /**
     * Runs the network.
     *
     * @param xs inputs, one {@code [batch, inputs]} tensor per step
     * @return outputs, one per step
     */
    public Tensor[] run(Tensor[] xs) {
        int batch = xs[0].shape().dim(0);
        Tensor c = Tensor.zeros(Shape.of(batch, hidden));
        Tensor h = c;
        Tensor[] ys = new Tensor[xs.length];
        for (int t = 0; t < xs.length; t++) {
            c = c.mul(alpha).add(xs[t].matmul(wIn.value()).add(h.matmul(wRec.value())).add(b.value()).mul(1 - alpha));
            h = c.tanh();
            ys[t] = h.matmul(wOut.value()).add(bOut.value());
        }
        return ys;
    }

    /**
     * Mean squared error over a sequence.
     *
     * @param xs the inputs per step
     * @param targets the targets per step
     * @return {@code mean_t 1/2 |y_t - target_t|^2} averaged over the batch
     */
    public float loss(Tensor[] xs, Tensor[] targets) {
        Tensor[] ys = run(xs);
        double sum = 0;
        for (int t = 0; t < ys.length; t++) {
            sum += 0.5 * ys[t].sub(targets[t]).square().sum().item() / ys[t].shape().dim(0);
        }
        return (float) (sum / ys.length);
    }

    /**
     * Learns from one batch of sequences and steps the optimizer.
     *
     * @param xs the inputs per step
     * @param targets the targets per step
     * @return the loss before the update
     */
    public float trainSequence(Tensor[] xs, Tensor[] targets) {
        return mode == Mode.BPTT ? bptt(xs, targets) : eprop(xs, targets);
    }

    private float eprop(Tensor[] xs, Tensor[] targets) {
        int batch = xs[0].shape().dim(0);
        int steps = xs.length;
        float norm = 1f / (batch * steps);
        Tensor c = Tensor.zeros(Shape.of(batch, hidden));
        Tensor h = c;
        Tensor traceIn = Tensor.zeros(xs[0].shape());
        Tensor traceRec = Tensor.zeros(Shape.of(batch, hidden));
        Tensor traceB = Tensor.zeros(Shape.of(batch, 1));
        Tensor gIn = null;
        Tensor gRec = null;
        Tensor gB = null;
        Tensor gOut = null;
        Tensor gBOut = null;
        double loss = 0;
        for (int t = 0; t < steps; t++) {
            // filtered presynaptic traces (the eligibility vectors of leaky units)
            traceIn = traceIn.mul(alpha).add(xs[t].mul(1 - alpha));
            traceRec = traceRec.mul(alpha).add(h.mul(1 - alpha));
            traceB = traceB.mul(alpha).add(1 - alpha);
            c = c.mul(alpha).add(xs[t].matmul(wIn.value()).add(h.matmul(wRec.value())).add(b.value()).mul(1 - alpha));
            h = c.tanh();
            Tensor y = h.matmul(wOut.value()).add(bOut.value());
            Tensor err = y.sub(targets[t]);
            loss += 0.5 * err.square().sum().item() / batch;
            Tensor signal = err.matmul(mode == Mode.SYMMETRIC ? wOut.value().transpose() : feedback);
            Tensor gated = signal.mul(LearningMath.tanhDerivative(h));   // L_j psi_j
            gIn = add(gIn, traceIn.transpose().matmul(gated));
            gRec = add(gRec, traceRec.transpose().matmul(gated));
            gB = add(gB, traceB.transpose().matmul(gated));
            gOut = add(gOut, h.transpose().matmul(err));
            gBOut = add(gBOut, err.sum(0, true));
            Tensor.evaluate(List.of(c, h, traceIn, traceRec, gIn, gRec, gOut));
        }
        wIn.setGrad(gIn.mul(norm));
        wRec.setGrad(gRec.mul(norm));
        b.setGrad(gB.mul(norm));
        wOut.setGrad(gOut.mul(norm));
        bOut.setGrad(gBOut.mul(norm));
        optimizer.step(parameters());
        return (float) (loss / steps);
    }

    private static Tensor add(Tensor acc, Tensor x) {
        return acc == null ? x : acc.add(x);
    }

    private float bptt(Tensor[] xs, Tensor[] targets) {
        int batch = xs[0].shape().dim(0);
        Var c = Var.of(Tensor.zeros(Shape.of(batch, hidden)));
        Var h = c;
        Var loss = null;
        for (int t = 0; t < xs.length; t++) {
            c = c.mul(alpha).add(Var.of(xs[t]).matmul(wIn).add(h.matmul(wRec)).add(b).mul(1 - alpha));
            h = c.tanh();
            Var term = h.matmul(wOut).add(bOut).sub(targets[t]).square().sum().mul(0.5f / batch);
            loss = loss == null ? term : loss.add(term);
        }
        loss = loss.div(xs.length);
        loss.backward();
        float value = loss.value().item();
        optimizer.step(parameters());
        return value;
    }
}
