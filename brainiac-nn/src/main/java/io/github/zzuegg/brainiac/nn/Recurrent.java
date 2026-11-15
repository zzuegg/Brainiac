package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A recurrent layer over sequences {@code [steps, features]}: a simple RNN, an LSTM or a
 * GRU, optionally bidirectional.
 *
 * <pre>{@code
 * Recurrent.lstm(64)                              // last hidden state: [steps, f] -> [64]
 *     .returnSequences(true)                      // every hidden state: [steps, 64]
 *     .bidirectional(Recurrent.Merge.CONCAT)      // forward and backward pass: [steps, 128]
 *     .reverse(true)                              // read the sequence backwards
 *     .activation(ActivationOp.TANH)              // candidate / output activation
 *     .recurrentActivation(ActivationOp.HARD_SIGMOID)   // gate activation
 *     .unitForgetBias(true)                       // LSTM forget gate starts open
 *     .learnInitialState(true)                    // learned h0 (and c0) instead of zeros
 *     .bias(false);
 * Recurrent.gru(32).resetAfter(false);            // original GRU instead of the cuDNN/PyTorch variant
 * Recurrent.rnn(16).activation(ActivationOp.RELU);
 * }</pre>
 *
 * <p>Gates are stored in the order {@code i, f, g, o} (LSTM) and {@code r, z, n} (GRU), as
 * in PyTorch. The backward pass through time is hand-written, so cost grows linearly
 * with the sequence length.
 */
public final class Recurrent implements Layer {

    /** The kind of recurrent cell. */
    public enum Cell {
        /** {@code h = act(W x + U h + b)} */
        RNN(1),
        /** long short-term memory: input, forget, candidate and output gates and a cell state */
        LSTM(4),
        /** gated recurrent unit: reset and update gates */
        GRU(3);

        private final int gates;

        Cell(int gates) {
            this.gates = gates;
        }

        /** @return the number of weight blocks */
        public int gates() {
            return gates;
        }
    }

    /** How a bidirectional layer joins its two directions. */
    public enum Merge {
        /** side by side: twice the units */
        CONCAT,
        /** added */
        SUM,
        /** averaged */
        AVERAGE,
        /** multiplied */
        MULTIPLY
    }

    /**
     * Every setting of a recurrent layer, as stored in model files.
     *
     * @param cell the cell type
     * @param inputs the features per step
     * @param units the hidden size
     * @param returnSequences whether to return every step or only the last
     * @param reverse whether to read the sequence backwards
     * @param merge how to join two directions, or {@code null} for one direction
     * @param activation the candidate / output activation
     * @param recurrentActivation the gate activation
     * @param resetAfter GRU: apply the reset gate after the recurrent matmul
     * @param bias whether to learn biases
     * @param learnInitialState whether the initial state is learned
     */
    public record Settings(Cell cell, int inputs, int units, boolean returnSequences, boolean reverse, Merge merge,
            ActivationOp activation, ActivationOp recurrentActivation, boolean resetAfter, boolean bias,
            boolean learnInitialState) {
    }

    private final Settings settings;
    private final Direction[] directions;

    /** The weights of one direction. */
    private static final class Direction {
        final Var wx;
        final Var wh;
        final Var b;
        final Var bhn;
        final Var h0;
        final Var c0;

        Direction(Var wx, Var wh, Var b, Var bhn, Var h0, Var c0) {
            this.wx = wx;
            this.wh = wh;
            this.b = b;
            this.bhn = bhn;
            this.h0 = h0;
            this.c0 = c0;
        }

        void collect(List<Var> out) {
            for (Var v : new Var[] {wx, wh, b, bhn, h0, c0}) {
                if (v != null) {
                    out.add(v);
                }
            }
        }
    }

    /**
     * Creates a recurrent layer with fresh weights.
     *
     * @param settings the settings
     * @param init the input weight initializer
     * @param recurrentInit the recurrent weight initializer
     * @param unitForgetBias LSTM: start the forget gate bias at 1
     * @param random the random source
     */
    public Recurrent(Settings settings, Init init, Init recurrentInit, boolean unitForgetBias, SplittableRandom random) {
        this.settings = Objects.requireNonNull(settings, "settings");
        if (settings.inputs() < 1 || settings.units() < 1) {
            throw new IllegalArgumentException("a recurrent layer needs at least one input and one unit");
        }
        int count = settings.merge() == null ? 1 : 2;
        directions = new Direction[count];
        int h = settings.units();
        int g = settings.cell().gates() * h;
        for (int d = 0; d < count; d++) {
            Var wx = Var.param(init.create(Shape.of(settings.inputs(), g), settings.inputs(), g, random));
            Var wh = Var.param(recurrentInit.create(Shape.of(h, g), h, g, random));
            Var b = null;
            if (settings.bias()) {
                float[] bias = new float[g];
                if (settings.cell() == Cell.LSTM && unitForgetBias) {
                    java.util.Arrays.fill(bias, h, 2 * h, 1f);
                }
                b = Var.param(Tensor.of(Shape.of(g), bias));
            }
            Var bhn = settings.cell() == Cell.GRU && settings.resetAfter()
                    ? (settings.bias() ? Var.param(Tensor.zeros(Shape.of(h))) : Var.of(Tensor.zeros(Shape.of(h))))
                    : null;
            Var h0 = settings.learnInitialState() ? Var.param(Tensor.zeros(Shape.of(h))) : null;
            Var c0 = settings.learnInitialState() && settings.cell() == Cell.LSTM
                    ? Var.param(Tensor.zeros(Shape.of(h))) : null;
            directions[d] = new Direction(wx, wh, b, bhn, h0, c0);
        }
    }

    /**
     * Describes a simple RNN layer.
     *
     * @param units the hidden size
     * @return the spec
     */
    public static Spec rnn(int units) {
        return new Spec(Cell.RNN, units);
    }

    /**
     * Describes an LSTM layer.
     *
     * @param units the hidden size
     * @return the spec
     */
    public static Spec lstm(int units) {
        return new Spec(Cell.LSTM, units);
    }

    /**
     * Describes a GRU layer.
     *
     * @param units the hidden size
     * @return the spec
     */
    public static Spec gru(int units) {
        return new Spec(Cell.GRU, units);
    }

    /** An immutable description of a recurrent layer; every setter returns a new spec. */
    public static final class Spec implements LayerSpec {

        private final Cell cell;
        private final int units;
        private boolean returnSequences;
        private boolean reverse;
        private Merge merge;
        private ActivationOp activation = ActivationOp.TANH;
        private ActivationOp recurrentActivation = ActivationOp.SIGMOID;
        private boolean resetAfter = true;
        private boolean bias = true;
        private boolean learnInitialState;
        private boolean unitForgetBias = true;
        private Init init = Init.XAVIER;
        private Init recurrentInit = Init.ORTHOGONAL;

        private Spec(Cell cell, int units) {
            if (units < 1) {
                throw new IllegalArgumentException("units must be at least 1, got " + units);
            }
            this.cell = cell;
            this.units = units;
        }

        private Spec copy() {
            Spec s = new Spec(cell, units);
            s.returnSequences = returnSequences;
            s.reverse = reverse;
            s.merge = merge;
            s.activation = activation;
            s.recurrentActivation = recurrentActivation;
            s.resetAfter = resetAfter;
            s.bias = bias;
            s.learnInitialState = learnInitialState;
            s.unitForgetBias = unitForgetBias;
            s.init = init;
            s.recurrentInit = recurrentInit;
            return s;
        }

        /**
         * Returns every hidden state instead of only the last; needed to stack layers.
         *
         * @param all whether to return all steps
         * @return the new spec
         */
        public Spec returnSequences(boolean all) {
            Spec s = copy();
            s.returnSequences = all;
            return s;
        }

        /**
         * Reads the sequence from the end; returned sequences are flipped back so step
         * {@code t} still lines up with input step {@code t}.
         *
         * @param reverse whether to read backwards
         * @return the new spec
         */
        public Spec reverse(boolean reverse) {
            Spec s = copy();
            s.reverse = reverse;
            return s;
        }

        /**
         * Runs a second layer over the reversed sequence and merges both.
         *
         * @param merge how to join the two directions
         * @return the new spec
         */
        public Spec bidirectional(Merge merge) {
            Spec s = copy();
            s.merge = Objects.requireNonNull(merge, "merge");
            return s;
        }

        /**
         * Sets the activation of the candidate (and of the LSTM cell output).
         *
         * @param op the activation
         * @return the new spec
         */
        public Spec activation(ActivationOp op) {
            Spec s = copy();
            s.activation = Objects.requireNonNull(op, "op");
            return s;
        }

        /**
         * Sets the activation of the gates.
         *
         * @param op the activation, usually {@code SIGMOID} or {@code HARD_SIGMOID}
         * @return the new spec
         */
        public Spec recurrentActivation(ActivationOp op) {
            Spec s = copy();
            s.recurrentActivation = Objects.requireNonNull(op, "op");
            return s;
        }

        /**
         * GRU only: {@code true} (default, as cuDNN and PyTorch) applies the reset gate
         * after multiplying the state by its weights; {@code false} is the original GRU.
         *
         * @param after where to apply the reset gate
         * @return the new spec
         */
        public Spec resetAfter(boolean after) {
            Spec s = copy();
            s.resetAfter = after;
            return s;
        }

        /**
         * Turns the biases on or off.
         *
         * @param bias whether to learn biases
         * @return the new spec
         */
        public Spec bias(boolean bias) {
            Spec s = copy();
            s.bias = bias;
            return s;
        }

        /**
         * Learns the initial hidden (and cell) state instead of starting at zero.
         *
         * @param learn whether to learn it
         * @return the new spec
         */
        public Spec learnInitialState(boolean learn) {
            Spec s = copy();
            s.learnInitialState = learn;
            return s;
        }

        /**
         * LSTM only: start the forget gate bias at 1 so memory is kept early in training.
         *
         * @param unit whether to do so (default)
         * @return the new spec
         */
        public Spec unitForgetBias(boolean unit) {
            Spec s = copy();
            s.unitForgetBias = unit;
            return s;
        }

        /**
         * Sets the initializers.
         *
         * @param input for the input weights (default Xavier)
         * @param recurrent for the recurrent weights (default orthogonal)
         * @return the new spec
         */
        public Spec init(Init input, Init recurrent) {
            Spec s = copy();
            s.init = Objects.requireNonNull(input, "input");
            s.recurrentInit = Objects.requireNonNull(recurrent, "recurrent");
            return s;
        }

        @Override
        public Shape outputShape(Shape input) {
            if (input.rank() != 2) {
                throw new IllegalArgumentException("a recurrent layer expects [steps, features], got " + input);
            }
            int width = merge == Merge.CONCAT ? 2 * units : units;
            return returnSequences ? Shape.of(input.dim(0), width) : Shape.of(width);
        }

        @Override
        public Layer create(Shape input, SplittableRandom random) {
            outputShape(input);
            return new Recurrent(new Settings(cell, input.dim(1), units, returnSequences, reverse, merge, activation,
                    recurrentActivation, resetAfter, bias, learnInitialState), init, recurrentInit, unitForgetBias,
                    random);
        }
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.rank() != 3 || s.dim(2) != settings.inputs()) {
            throw new IllegalArgumentException(this + " expects [batch, steps, " + settings.inputs() + "], got " + s);
        }
        Var forward = run(directions[0], input, settings.reverse());
        if (directions.length == 1) {
            return finish(forward, settings.reverse());
        }
        Var backward = run(directions[1], input, !settings.reverse());
        Var a = finish(forward, settings.reverse());
        Var b = finish(backward, !settings.reverse());
        return switch (settings.merge()) {
            case CONCAT -> Var.concat(-1, a, b);
            case SUM -> a.add(b);
            case AVERAGE -> a.add(b).mul(0.5f);
            case MULTIPLY -> a.mul(b);
        };
    }

    /** Runs one direction; the result is in processing order. */
    private Var run(Direction d, Var input, boolean reversed) {
        int n = input.shape().dim(0);
        int h = settings.units();
        Var x = reversed ? flip(input) : input;
        Var xp = x.matmul(d.wx);
        if (d.b != null) {
            xp = xp.add(d.b);
        }
        Var h0 = d.h0 != null ? d.h0.expand(Shape.of(n, h)) : Var.of(Tensor.zeros(Shape.of(n, h), input.value().device()));
        RecurrentOp op = new RecurrentOp(settings.cell(), settings.activation(), settings.recurrentActivation(),
                settings.resetAfter());
        return switch (settings.cell()) {
            case RNN -> Var.apply(op, xp, d.wh, h0);
            case LSTM -> Var.apply(op, xp, d.wh, h0, d.c0 != null ? d.c0.expand(Shape.of(n, h))
                    : Var.of(Tensor.zeros(Shape.of(n, h), input.value().device())));
            case GRU -> settings.resetAfter() ? Var.apply(op, xp, d.wh, h0, d.bhn) : Var.apply(op, xp, d.wh, h0);
        };
    }

    /** Picks the last processed state, or flips a returned sequence back to input order. */
    private Var finish(Var states, boolean reversed) {
        int steps = states.shape().dim(1);
        if (!settings.returnSequences()) {
            return states.slice(1, steps - 1, steps).reshape(states.shape().dim(0), settings.units());
        }
        return reversed ? flip(states) : states;
    }

    private static Var flip(Var x) {
        int steps = x.shape().dim(1);
        float[] idx = new float[steps];
        for (int i = 0; i < steps; i++) {
            idx[i] = steps - 1 - i;
        }
        return x.gather(1, Tensor.of(Shape.of(steps), idx, x.value().device()));
    }

    @Override
    public Shape outputShape(Shape input) {
        int width = settings.merge() == Merge.CONCAT ? 2 * settings.units() : settings.units();
        return settings.returnSequences() ? Shape.of(input.dim(0), width) : Shape.of(width);
    }

    @Override
    public List<Var> parameters() {
        List<Var> out = new ArrayList<>();
        for (Direction d : directions) {
            d.collect(out);
        }
        out.removeIf(v -> !v.requiresGrad());
        return out;
    }

    /** @return the settings */
    public Settings settings() {
        return settings;
    }

    @Override
    public String toString() {
        return (settings.merge() != null ? "Bidirectional" : "") + settings.cell() + "(" + settings.inputs() + " -> "
                + settings.units() + (settings.returnSequences() ? ", sequences" : "") + ")";
    }

    static Config write(Recurrent r) {
        Settings s = r.settings;
        return Config.builder()
                .put("cell", s.cell().name())
                .put("inputs", s.inputs())
                .put("units", s.units())
                .put("returnSequences", s.returnSequences())
                .put("reverse", s.reverse())
                .put("merge", s.merge() == null ? "" : s.merge().name())
                .put("activation", s.activation().name())
                .put("recurrentActivation", s.recurrentActivation().name())
                .put("resetAfter", s.resetAfter())
                .put("bias", s.bias())
                .put("learnInitialState", s.learnInitialState())
                .build();
    }

    static Recurrent read(Config c) {
        String merge = c.getString("merge");
        Settings s = new Settings(Cell.valueOf(c.getString("cell")), c.getInt("inputs"), c.getInt("units"),
                c.getBoolean("returnSequences"), c.getBoolean("reverse"), merge.isEmpty() ? null : Merge.valueOf(merge),
                ActivationOp.valueOf(c.getString("activation")), ActivationOp.valueOf(c.getString("recurrentActivation")),
                c.getBoolean("resetAfter"), c.getBoolean("bias"), c.getBoolean("learnInitialState"));
        return new Recurrent(s, Init.ZEROS, Init.ZEROS, false, new SplittableRandom(0));
    }
}
