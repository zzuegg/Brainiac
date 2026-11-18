package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.random.RandomGenerator;

/**
 * u-muP (Blake et al. 2024, <a href="https://arxiv.org/abs/2407.17465">arXiv:2407.17465</a>):
 * the maximal-update parameterization ({@link MaximalUpdate}) expressed in unit-scaled
 * operations ({@code experimental.nn.UnitScaledOps}). Every weight is initialized from
 * {@code N(0, 1)}; the width dependence moves into the operations ({@code 1/sqrt(fan_in)} for
 * hidden matmuls, {@code 1/fan_in} for the readout) and into Adam's per-tensor learning rates:
 * input weights and biases take the base rate, hidden weights {@code base / sqrt(fan_in)}, the
 * readout the base rate. Optimal base learning rates then transfer across widths, and the
 * unit-scaled tensors suit low-precision formats.
 */
public final class UnitMup {

    /** The role of a parameter, which sets its learning rate. */
    public enum Role {
        /** The first layer's weight, or any bias or gain. */
        INPUT,
        /** A hidden {@code [width, width]} weight. */
        HIDDEN,
        /** The readout weight. */
        OUTPUT
    }

    /** Creates the rules; they hold no state. */
    public UnitMup() {
    }

    /**
     * A unit-scaled weight.
     *
     * @param in the fan-in
     * @param out the fan-out
     * @param random the random source
     * @return a {@code [in, out]} parameter drawn from {@code N(0, 1)}
     */
    public Var weight(int in, int out, RandomGenerator random) {
        return Var.param(Tensor.normal(Shape.of(in, out), 0, 1, random));
    }

    /**
     * The Adam learning rate for a parameter.
     *
     * @param role its role
     * @param fanIn its fan-in
     * @param base the base learning rate, the one that transfers across widths
     * @return the rate
     */
    public float learningRate(Role role, int fanIn, float base) {
        return role == Role.HIDDEN ? (float) (base / Math.sqrt(fanIn)) : base;
    }

    /**
     * Adam with the u-muP learning rate per group.
     *
     * @param base the base learning rate
     * @param input the input weights and all biases
     * @param hidden the hidden weights, all with the same fan-in
     * @param output the readout weights
     * @return the optimizer
     */
    public Optimizer adam(float base, List<Var> input, List<Var> hidden, List<Var> output) {
        int fanIn = hidden.isEmpty() ? 1 : hidden.get(0).shape().dim(0);
        return Optimizer.groups(Optimizer.adam(base),
                Optimizer.Group.of(hidden, Optimizer.adam(learningRate(Role.HIDDEN, fanIn, base))),
                Optimizer.Group.of(input, Optimizer.adam(learningRate(Role.INPUT, fanIn, base))),
                Optimizer.Group.of(output, Optimizer.adam(learningRate(Role.OUTPUT, fanIn, base))));
    }

    @Override
    public String toString() {
        return "UnitMup";
    }
}
