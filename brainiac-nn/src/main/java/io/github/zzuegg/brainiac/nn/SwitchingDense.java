package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A dense layer whose inputs also decide, per unit and per sample, which activation the
 * unit uses. Each unit computes, from the same inputs, its usual weighted sum {@code z}
 * and a second one, its control value {@code c}:
 *
 * <pre>
 * z = x W + b          c = x V + d
 * y = sum_k p_k f_k(z) with p = softmax_k(c u_k + e_k)
 * </pre>
 *
 * <p>So one input pattern can make a unit act like {@code tanh} and another like
 * {@code relu}. The choice is soft, so where to switch is learned by backpropagation. The
 * offsets {@code u_k} start spread from -1 (first candidate) to 1 (last candidate), so a
 * low control value leans to the first candidate and a high one to the last; {@code e}
 * starts at 0, an equal blend. In {@link #hard} mode each unit uses only its most likely
 * candidate (no gradient reaches the control weights then), for evolution or for a
 * cheaper model after training.
 *
 * <p>Everything after the two matrix products is element-wise, so a GPU computes the
 * blend of all candidates in one fused kernel.
 *
 * <pre>{@code
 * Model.sequential().input(2)
 *         .add(SwitchingDense.of(32, Activation.RELU, Activation.TANH, Activation.SILU))
 *         .dense(3, Activation.SOFTMAX)
 *         .build();
 * }</pre>
 */
public final class SwitchingDense implements Layer {

    private final int inputSize;
    private final int units;
    private final Activation[] candidates;
    private final boolean hard;
    private final Var weight;
    private final Var bias;
    private final Var control;
    private final Var controlBias;
    private final Var scale;
    private final Var offset;

    /**
     * Creates a layer with initialized weights.
     *
     * @param inputSize the number of input features
     * @param units the number of outputs
     * @param candidates the activations each unit chooses between, at least two
     * @param hard whether each unit uses only its most likely candidate
     * @param random the random source for the initializers
     */
    public SwitchingDense(int inputSize, int units, Activation[] candidates, boolean hard, RandomGenerator random) {
        this(inputSize, units, candidates, hard,
                init(candidates).create(Shape.of(inputSize, units), inputSize, units, random),
                Tensor.zeros(Shape.of(units)),
                Init.XAVIER.create(Shape.of(inputSize, units), inputSize, units, random),
                Tensor.zeros(Shape.of(units)),
                spread(candidates.length, units),
                Tensor.zeros(Shape.of(candidates.length, units)));
    }

    /**
     * Creates a layer with given parameters, for example loaded from a file.
     *
     * @param inputSize the number of input features
     * @param units the number of outputs
     * @param candidates the activations each unit chooses between, at least two
     * @param hard whether each unit uses only its most likely candidate
     * @param weight the weights of {@code z}, {@code [inputSize, units]}
     * @param bias the bias of {@code z}, {@code [units]}
     * @param control the weights of the control value, {@code [inputSize, units]}
     * @param controlBias the bias of the control value, {@code [units]}
     * @param scale how strongly the control value moves each candidate, {@code [candidates, units]}
     * @param offset each candidate's base preference, {@code [candidates, units]}
     */
    public SwitchingDense(int inputSize, int units, Activation[] candidates, boolean hard, Tensor weight,
                          Tensor bias, Tensor control, Tensor controlBias, Tensor scale, Tensor offset) {
        if (inputSize < 1 || units < 1) {
            throw new IllegalArgumentException("SwitchingDense needs at least 1 input and 1 unit, got " + inputSize
                    + " -> " + units);
        }
        if (candidates.length < 2) {
            throw new IllegalArgumentException("SwitchingDense needs at least two activations, got "
                    + candidates.length);
        }
        int k = candidates.length;
        check("weight", weight, Shape.of(inputSize, units));
        check("bias", bias, Shape.of(units));
        check("control", control, Shape.of(inputSize, units));
        check("controlBias", controlBias, Shape.of(units));
        check("scale", scale, Shape.of(k, units));
        check("offset", offset, Shape.of(k, units));
        this.inputSize = inputSize;
        this.units = units;
        this.candidates = new Activation[k];
        for (int i = 0; i < k; i++) {
            this.candidates[i] = Objects.requireNonNull(candidates[i], "activation").bind(units);
        }
        this.hard = hard;
        this.weight = Var.param(weight);
        this.bias = Var.param(bias);
        this.control = Var.param(control);
        this.controlBias = Var.param(controlBias);
        this.scale = Var.param(scale);
        this.offset = Var.param(offset);
    }

    private static void check(String name, Tensor t, Shape expected) {
        if (!t.shape().equals(expected)) {
            throw new IllegalArgumentException("SwitchingDense needs " + name + " " + expected + ", got " + t.shape());
        }
    }

    private static Init init(Activation[] candidates) {
        Init first = candidates[0].preferredInit();
        for (Activation a : candidates) {
            if (a.preferredInit() != first) {
                return Init.XAVIER;
            }
        }
        return first;
    }

    /** Offsets from -1 (first candidate) to 1 (last), the same for every unit. */
    private static Tensor spread(int k, int units) {
        float[] v = new float[k * units];
        for (int i = 0; i < k; i++) {
            java.util.Arrays.fill(v, i * units, (i + 1) * units, -1f + 2f * i / (k - 1));
        }
        return Tensor.of(Shape.of(k, units), v);
    }

    /**
     * Describes a layer whose input width is taken from the last dimension of its input.
     *
     * @param units the number of outputs
     * @param candidates the activations each unit chooses between, at least two
     * @return the spec
     */
    public static Spec of(int units, Activation... candidates) {
        return new Spec(units, candidates.clone(), false);
    }

    /** An immutable description; {@link #hard} returns a new spec. */
    public static final class Spec implements LayerSpec {

        private final int units;
        private final Activation[] candidates;
        private final boolean hard;

        private Spec(int units, Activation[] candidates, boolean hard) {
            if (units < 1) {
                throw new IllegalArgumentException("units must be at least 1, got " + units);
            }
            if (candidates.length < 2) {
                throw new IllegalArgumentException("SwitchingDense needs at least two activations, got "
                        + candidates.length);
            }
            for (Activation a : candidates) {
                Objects.requireNonNull(a, "activation");
            }
            this.units = units;
            this.candidates = candidates;
            this.hard = hard;
        }

        /**
         * @param hard whether each unit uses only its most likely candidate
         * @return the new spec
         */
        public Spec hard(boolean hard) {
            return new Spec(units, candidates, hard);
        }

        @Override
        public Shape outputShape(Shape input) {
            int[] dims = input.toArray();
            dims[dims.length - 1] = units;
            return Shape.of(dims);
        }

        @Override
        public Layer create(Shape input, SplittableRandom random) {
            return new SwitchingDense(input.dim(-1), units, candidates, hard, random);
        }
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Shape s = input.shape();
        if (s.rank() < 2 || s.dim(-1) != inputSize) {
            throw new IllegalArgumentException("SwitchingDense(" + units + ") expects input [*, " + inputSize
                    + "], got " + s);
        }
        Var z = input.matmul(weight).add(bias);
        Var c = input.matmul(control).add(controlBias);
        int k = candidates.length;
        Var[] logits = new Var[k];
        Var max = null;
        for (int i = 0; i < k; i++) {
            logits[i] = c.mul(scale.slice(0, i, i + 1)).add(offset.slice(0, i, i + 1));
            max = max == null ? logits[i] : max.maximum(logits[i]);
        }
        Var weighted = null;
        Var total = null;
        for (int i = 0; i < k; i++) {
            Var f = candidates[i].forward(z, training);
            if (hard) {
                // the first candidate at the maximum wins; earlier winners mask later ties
                Tensor pick = logits[i].value().greater(max.value().sub(Tensor.full(Shape.of(), 1e-6f,
                        max.value().device())));
                if (total != null) {
                    pick = pick.mul(total.value().neg().add(Tensor.full(Shape.of(), 1f, pick.device())));
                }
                Var term = f.mul(pick);
                weighted = weighted == null ? term : weighted.add(term);
                total = total == null ? Var.of(pick) : Var.of(total.value().add(pick));
            } else {
                Var e = logits[i].sub(max).exp();
                Var term = f.mul(e);
                weighted = weighted == null ? term : weighted.add(term);
                total = total == null ? e : total.add(e);
            }
        }
        return hard ? weighted : weighted.div(total);
    }

    /**
     * How much each candidate counts for a batch, per sample and unit: the softmax of
     * {@code c u_k + e_k}.
     *
     * @param input a batch, {@code [*, inputSize]}
     * @return the shares, {@code [candidates, *, units]}; they sum to 1 over the first axis
     */
    public Tensor shares(Tensor input) {
        Var x = Var.of(input);
        Var c = x.matmul(Var.of(control.value())).add(Var.of(controlBias.value()));
        int k = candidates.length;
        List<Tensor> parts = new ArrayList<>(k);
        for (int i = 0; i < k; i++) {
            Var l = c.mul(Var.of(scale.value().slice(0, i, i + 1))).add(Var.of(offset.value().slice(0, i, i + 1)));
            parts.add(l.value().reshape(prepend(l.shape())));
        }
        Tensor stacked = Tensor.concat(0, parts.toArray(new Tensor[0]));
        return Activation.softmax(0).forward(Var.of(stacked)).value();
    }

    private static int[] prepend(Shape s) {
        int[] d = new int[s.rank() + 1];
        d[0] = 1;
        for (int i = 0; i < s.rank(); i++) {
            d[i + 1] = s.dim(i);
        }
        return d;
    }

    @Override
    public List<Var> parameters() {
        List<Var> params = new ArrayList<>(List.of(weight, bias, control, controlBias, scale, offset));
        for (Activation a : candidates) {
            params.addAll(a.parameters());
        }
        return params;
    }

    @Override
    public List<Var> buffers() {
        List<Var> out = new ArrayList<>();
        for (Activation a : candidates) {
            out.addAll(a.buffers());
        }
        return out;
    }

    @Override
    public int outputSize(int inputSize) {
        return units;
    }

    /** @return the number of input features */
    public int inputSize() {
        return inputSize;
    }

    /** @return the number of outputs */
    public int units() {
        return units;
    }

    /** @return the activations each unit chooses between, bound to this layer's width */
    public List<Activation> candidates() {
        return List.of(candidates);
    }

    /** @return whether each unit uses only its most likely candidate */
    public boolean hard() {
        return hard;
    }

    /** @return the weights of {@code z}, {@code [inputSize, units]} */
    public Var weight() {
        return weight;
    }

    /** @return the weights of the control value, {@code [inputSize, units]} */
    public Var control() {
        return control;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("SwitchingDense(" + inputSize + " -> " + units + ", ");
        for (int i = 0; i < candidates.length; i++) {
            sb.append(i > 0 ? "/" : "").append(candidates[i].name());
        }
        return sb.append(hard ? ", hard)" : ")").toString();
    }
}
