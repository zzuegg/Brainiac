package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.BitLinear;
import io.github.zzuegg.brainiac.experimental.nn.StochasticBitstreamDense;
import io.github.zzuegg.brainiac.experimental.tensor.TernaryMatrix;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A trained digit classifier (a 64-32-10 tanh MLP on {@link ToyTasks#digits}, inputs mapped to
 * {@code [-1, 1]}) run on simulated stochastic-computing hardware with
 * {@link StochasticBitstreamDense}: accuracy against bitstream length for the multiplexer and
 * the parallel-counter adder, against the other cheap-arithmetic formats of the module at the
 * same trained weights (8-bit post-training quantization, ternary post-training quantization
 * through {@link TernaryMatrix}) and against a model trained from the start with
 * {@link BitLinear} (ternary weights, quantization-aware). Alaghi and Hayes 2013,
 * <a href="https://dl.acm.org/doi/10.1145/2465787.2465794">10.1145/2465787.2465794</a>.
 */
public final class StochasticComputing {

    /**
     * One inference format.
     *
     * @param format the format
     * @param accuracy held-out accuracy
     * @param bitOpsPerSample gate-level bit operations per sample (0 when not counted)
     */
    public record Result(String format, double accuracy, double bitOpsPerSample) {
    }

    private final Model mlp;
    private final ToyTasks.Data test;
    private final Tensor testInputs;

    private StochasticComputing(Model mlp, ToyTasks.Data test, Tensor testInputs) {
        this.mlp = mlp;
        this.test = test;
        this.testInputs = testInputs;
    }

    /**
     * Trains the float MLP.
     *
     * @param steps Adam steps on minibatches of 32
     * @param seed the seed
     * @return the experiment
     */
    public static StochasticComputing train(int steps, long seed) {
        ToyTasks.Data train = ToyTasks.digits(10, 40, 8, 0.05, seed);
        ToyTasks.Data test = ToyTasks.digits(10, 20, 8, 0.05, seed, seed + 999);
        Model m = Model.sequential().input(64).dense(32, Activation.TANH).dense(10).seed(seed).build();
        fit(m, train, steps, seed);
        return new StochasticComputing(m, test, bipolar(test.inputs()));
    }

    private static void fit(Model m, ToyTasks.Data train, int steps, long seed) {
        Tensor x = bipolar(train.inputs());
        Tensor y = train.ids();
        Optimizer opt = Optimizer.adam(0.01f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            int from = r.nextInt(train.size() - 32);
            Var loss = Functional.crossEntropy(m.forward(Var.of(x.slice(0, from, from + 32)), true),
                    y.slice(0, from, from + 32));
            loss.backward();
            opt.step(m.parameters());
        }
    }

    private static Tensor bipolar(Tensor x) {
        return x.mul(2f).sub(1f).maximum(Tensor.full(Shape.of(), -1f)).minimum(Tensor.full(Shape.of(), 1f));
    }

    private Tensor[] weights() {
        List<Var> p = mlp.parameters();
        return new Tensor[] {p.get(0).value(), p.get(1).value(), p.get(2).value(), p.get(3).value()};
    }

    /** @return the float model's accuracy */
    public Result floating() {
        return new Result("float32", ToyTasks.accuracy(mlp.forward(Var.of(testInputs), false).value(), test.y()), 0);
    }

    /**
     * Runs both layers as stochastic bitstreams; the hidden tanh is applied to the decoded counts.
     *
     * @param length the stream length
     * @param adder the adder circuit
     * @param seed the stream seed
     * @return the result
     */
    public Result stochastic(int length, StochasticBitstreamDense.Adder adder, long seed) {
        Tensor[] w = weights();
        StochasticBitstreamDense l1 = StochasticBitstreamDense.of(w[0], w[1], length, adder, seed);
        StochasticBitstreamDense l2 = StochasticBitstreamDense.of(w[2], w[3], length, adder, seed + 1);
        Var h = l1.forward(Var.of(testInputs)).tanh();
        Tensor scores = l2.forward(h).value();
        double ops = (double) (l1.bitOperations() + l2.bitOperations()) / test.size();
        return new Result("SC " + adder + " L=" + length, ToyTasks.accuracy(scores, test.y()), ops);
    }

    /** @return 8-bit symmetric post-training quantization of weights and activations */
    public Result int8() {
        Tensor[] w = weights();
        Tensor h = Var.of(quant8(testInputs)).matmul(Var.of(quant8(w[0]))).add(w[1]).tanh().value();
        Tensor scores = Var.of(quant8(h)).matmul(Var.of(quant8(w[2]))).add(w[3]).value();
        return new Result("int8 PTQ", ToyTasks.accuracy(scores, test.y()), 0);
    }

    private static Tensor quant8(Tensor t) {
        float max = Math.max(t.abs().max(-1, false).max(0, false).item(), 1e-8f);
        float s = 127f / max;
        return t.mul(s).round().div(s);
    }

    /** @return ternary post-training quantization of both layers (BitNet b1.58 inference) */
    public Result ternary() {
        Tensor[] w = weights();
        int n = test.size();
        TernaryMatrix t1 = TernaryMatrix.quantize(w[0].toArray(), 64, 32);
        TernaryMatrix t2 = TernaryMatrix.quantize(w[2].toArray(), 32, 10);
        Tensor h = Tensor.of(Shape.of(n, 32), t1.matmul(testInputs.toArray(), n));
        h = Var.of(h).add(w[1]).tanh().value();
        Tensor scores = Var.of(Tensor.of(Shape.of(n, 10), t2.matmul(h.toArray(), n))).add(w[3]).value();
        return new Result("ternary PTQ", ToyTasks.accuracy(scores, test.y()), 0);
    }

    /**
     * Trains a BitLinear (ternary-weight, 8-bit activation, quantization-aware) model of the same
     * shape.
     *
     * @param steps Adam steps
     * @param seed the seed
     * @return the result
     */
    public Result bitLinear(int steps, long seed) {
        ToyTasks.Data train = ToyTasks.digits(10, 40, 8, 0.05, seed);
        Model m = Model.sequential().input(64).add(BitLinear.spec(32)).activation(Activation.TANH)
                .add(BitLinear.spec(10)).seed(seed).build();
        fit(m, train, steps, seed);
        return new Result("BitLinear QAT", ToyTasks.accuracy(m.forward(Var.of(testInputs), false).value(), test.y()), 0);
    }
}
