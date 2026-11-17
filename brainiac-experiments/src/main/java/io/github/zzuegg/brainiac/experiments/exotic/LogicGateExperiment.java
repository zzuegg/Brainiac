package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.LogicCircuit;
import io.github.zzuegg.brainiac.experimental.nn.LogicGateLayer;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.function.IntPredicate;

/**
 * A differentiable logic gate network learning a boolean function of 8 input bits (by
 * default "is the high nibble greater than the low nibble?"), then compiled to a
 * {@code LogicCircuit} that classifies 64 inputs per pass with bit operations only. The
 * class scores are group sums of the last layer's outputs, divided by a temperature
 * while training.
 */
public final class LogicGateExperiment {
    private final List<LogicGateLayer> layers = new ArrayList<>();
    private final IntPredicate function;
    private final float tau;
    private final Optimizer optimizer = Optimizer.adam(0.05f);

    private LogicGateExperiment(int[] widths, IntPredicate function, float tau, long seed) {
        this.function = function;
        this.tau = tau;
        int in = 8;
        for (int i = 0; i < widths.length; i++) {
            layers.add(LogicGateLayer.of(in, widths[i], seed + i));
            in = widths[i];
        }
    }

    /**
     * Creates the network.
     *
     * @param widths the gates per layer; the last must be even (two class groups)
     * @param function the boolean function of the 8-bit input to learn
     * @param tau the group-sum temperature
     * @param seed the seed
     * @return the experiment
     */
    public static LogicGateExperiment of(int[] widths, IntPredicate function, float tau, long seed) {
        return new LogicGateExperiment(widths, function, tau, seed);
    }

    /**
     * The comparator task.
     *
     * @return true when the high nibble is greater than the low nibble
     */
    public static IntPredicate highNibbleGreater() {
        return v -> (v >> 4) > (v & 15);
    }

    /** @return the layers */
    public List<LogicGateLayer> layers() {
        return List.copyOf(layers);
    }

    private static Tensor allInputs() {
        float[] x = new float[256 * 8];
        for (int v = 0; v < 256; v++) {
            for (int b = 0; b < 8; b++) {
                x[v * 8 + b] = v >> b & 1;
            }
        }
        return Tensor.of(Shape.of(256, 8), x);
    }

    /**
     * Runs the (soft) network.
     *
     * @param x inputs {@code [batch, 8]} in [0, 1]
     * @return the last layer's outputs
     */
    public Var forward(Var x) {
        for (LogicGateLayer l : layers) {
            x = l.forward(x);
        }
        return x;
    }

    private Var scores(Var out) {
        int b = out.shape().dim(0);
        int width = out.shape().dim(1);
        return out.reshape(b, 2, width / 2).sum(2, false).div(tau);
    }

    /**
     * Full-batch training on all 256 inputs.
     *
     * @param steps Adam steps
     * @return the final loss
     */
    public float train(int steps) {
        Tensor x = allInputs();
        float[] y = new float[256];
        for (int v = 0; v < 256; v++) {
            y[v] = function.test(v) ? 1 : 0;
        }
        Tensor labels = Tensor.of(Shape.of(256), y);
        float value = 0;
        for (int s = 0; s < steps; s++) {
            Var loss = Functional.crossEntropy(scores(forward(Var.of(x))), labels);
            value = loss.value().item();
            loss.backward();
            List<Var> params = new ArrayList<>();
            for (LogicGateLayer l : layers) {
                params.addAll(l.parameters());
            }
            optimizer.step(params);
        }
        return value;
    }

    /**
     * Compiles and scores the discrete circuit on all 256 inputs.
     *
     * @return {@code {accuracy, predictions as 0/1 per input}}
     */
    public double[] circuitAccuracy() {
        LogicCircuit circuit = LogicCircuit.compile(layers);
        double[] pred = new double[257];
        int right = 0;
        for (int chunk = 0; chunk < 4; chunk++) {
            long[] in = new long[8];
            for (int lane = 0; lane < 64; lane++) {
                int v = chunk * 64 + lane;
                for (int b = 0; b < 8; b++) {
                    in[b] |= (long) (v >> b & 1) << lane;
                }
            }
            int[][] s = LogicCircuit.groupSum(circuit.evaluate(in), 2, 64);
            for (int lane = 0; lane < 64; lane++) {
                int v = chunk * 64 + lane;
                int p = s[lane][1] > s[lane][0] ? 1 : 0;
                pred[1 + v] = p;
                right += (p == 1) == function.test(v) ? 1 : 0;
            }
        }
        pred[0] = right / 256.0;
        return pred;
    }

    /**
     * Hardens every layer and runs the network through the float path, to check that the
     * compiled circuit computes exactly the same thing.
     *
     * @return the hardened network's 0/1 outputs on all 256 inputs, {@code [256][width]}
     */
    public float[] hardenedOutputs() {
        for (LogicGateLayer l : layers) {
            l.harden();
        }
        return forward(Var.of(allInputs())).value().toArray();
    }
}
