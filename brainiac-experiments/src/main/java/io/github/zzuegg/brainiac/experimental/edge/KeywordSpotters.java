package io.github.zzuegg.brainiac.experimental.edge;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Keyword-spotting model families of "Hello Edge" sized to microcontroller budgets: a fully
 * connected DNN, a CNN, a depthwise-separable CNN (DS-CNN), and GRU, LSTM and convolutional-
 * recurrent (CRNN) models. Inputs are MFCC images {@code [1, frames, coefficients]}; outputs are
 * the keyword classes plus "unknown" and "silence".
 *
 * <p>The DNN, CNN and DS-CNN are {@link EdgeNet}s, so {@link IntegerInference} runs them in int8
 * and {@link MemoryPlanner} counts their peak memory; the recurrent families use core
 * {@link Recurrent} and run in float (their sizes count weights as int8 bytes). Each family is
 * scaled by a width multiplier, and {@link #sized} picks the largest width whose int8 weight
 * bytes and multiply-adds fit a budget.
 *
 * <p>Paper: Zhang et al., "Hello Edge: Keyword Spotting on Microcontrollers", 2017,
 * <a href="https://arxiv.org/abs/1711.07128">arXiv:1711.07128</a>.
 */
public final class KeywordSpotters {

    /** Model families. */
    public enum Family {
        /** three fully connected layers */
        DNN,
        /** two strided convolutions and two fully connected layers */
        CNN,
        /** a strided convolution, depthwise-separable blocks, average pooling and a classifier */
        DS_CNN,
        /** a GRU over the frames */
        GRU,
        /** an LSTM over the frames */
        LSTM,
        /** a convolution over each frame's coefficients, then a GRU */
        CRNN
    }

    /**
     * A microcontroller budget.
     *
     * @param name a label
     * @param weightBytes the most int8 weight bytes
     * @param macs the most multiply-adds per inference
     */
    public record Budget(String name, long weightBytes, long macs) {
    }

    /** Small, medium and large budgets, scaled to this task's 16 x 24 input. */
    public static final List<Budget> BUDGETS = List.of(new Budget("small", 12_000, 300_000),
            new Budget("medium", 40_000, 1_200_000), new Budget("large", 120_000, 4_000_000));

    /**
     * A sized model.
     *
     * @param family the family
     * @param model the model ({@link EdgeNet} for DNN, CNN and DS-CNN)
     * @param weightBytes weights at one byte each
     * @param macs multiply-adds per inference
     * @param width the width multiplier chosen
     */
    public record Model(Family family, Layer model, long weightBytes, long macs, int width) {

        /** @return the network when the family runs in int8, else {@code null} */
        public EdgeNet edgeNet() {
            return model instanceof EdgeNet e ? e : null;
        }
    }

    private KeywordSpotters() {
    }

    /**
     * Builds a family at a width.
     *
     * @param f the family
     * @param width the width multiplier (channels or units scale with it)
     * @param frames frames per utterance
     * @param coefficients coefficients per frame
     * @param classes classes
     * @param seed the seed
     * @return the model
     */
    public static Model build(Family f, int width, int frames, int coefficients, int classes, long seed) {
        return switch (f) {
            case DNN -> edge(f, width, EdgeNet.builder(seed, coefficients * frames).dense(16 * width, true)
                    .dense(16 * width, true).dense(classes, false).build());
            case CNN -> edge(f, width, EdgeNet.builder(seed, 1, frames, coefficients).conv(4 * width, 3, 2)
                    .conv(4 * width, 3, 2).dense(8 * width, true).dense(classes, false).build());
            case DS_CNN -> {
                EdgeNet.Builder b = EdgeNet.builder(seed, 1, frames, coefficients).conv(8 * width, 3, 2);
                for (int i = 0; i < 2 + width / 2; i++) {
                    b.separable(8 * width, 3, i == 0 ? 2 : 1);
                }
                yield edge(f, width, b.pool().dense(classes, false).build());
            }
            case GRU, LSTM -> {
                RecurrentSpotter m = new RecurrentSpotter(f == Family.LSTM, 0, 8 * width, frames, coefficients,
                        classes, seed);
                yield new Model(f, m, m.parameterCount(), m.macs(), width);
            }
            case CRNN -> {
                RecurrentSpotter m = new RecurrentSpotter(false, 2 * width, 8 * width, frames, coefficients, classes,
                        seed);
                yield new Model(f, m, m.parameterCount(), m.macs(), width);
            }
        };
    }

    private static Model edge(Family f, int width, EdgeNet n) {
        return new Model(f, n, n.parameterCount(), n.macs(), width);
    }

    /**
     * The widest model of a family within a budget (or the narrowest if none fits).
     *
     * @param f the family
     * @param budget the budget
     * @param frames frames
     * @param coefficients coefficients
     * @param classes classes
     * @param seed the seed
     * @return the model
     */
    public static Model sized(Family f, Budget budget, int frames, int coefficients, int classes, long seed) {
        Model best = build(f, 1, frames, coefficients, classes, seed);
        for (int w = 2; w <= 16; w++) {
            Model m = build(f, w, frames, coefficients, classes, seed);
            if (m.weightBytes() > budget.weightBytes() || m.macs() > budget.macs()) {
                break;
            }
            best = m;
        }
        return best;
    }

    /**
     * Whether a model fits a budget.
     *
     * @param m the model
     * @param b the budget
     * @return whether it fits
     */
    public static boolean fits(Model m, Budget b) {
        return m.weightBytes() <= b.weightBytes() && m.macs() <= b.macs();
    }

    /**
     * Trains any model with Adam and cross-entropy.
     *
     * @param model the model
     * @param x inputs {@code [n, 1, frames, coefficients]} (flattened for a DNN by the model)
     * @param y class ids
     * @param steps steps
     * @param lr the learning rate
     * @param seed the seed
     * @return the last loss
     */
    public static float train(Layer model, Tensor x, Tensor y, int steps, float lr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(lr);
        int n = x.shape().dim(0);
        List<Var> p = model.parameters();
        float last = 0;
        for (int s = 0; s < steps; s++) {
            int from = r.nextInt(Math.max(1, n - 32));
            int to = Math.min(n, from + 32);
            p.forEach(Var::zeroGrad);
            Var loss = Functional.crossEntropy(model.forward(input(model, x.slice(0, from, to)), true),
                    y.slice(0, from, to));
            last = loss.value().item();
            loss.backward();
            opt.step(p);
        }
        p.forEach(Var::zeroGrad);
        return last;
    }

    /**
     * Shapes inputs for a model: flattened rows for a fully connected network.
     *
     * @param model the model
     * @param x inputs {@code [n, 1, frames, coefficients]}
     * @return the model's input
     */
    public static Var input(Layer model, Tensor x) {
        if (model instanceof EdgeNet e && e.inputShape().length == 1) {
            return Var.of(x.reshape(x.shape().dim(0), e.inputShape()[0]));
        }
        return Var.of(x);
    }

    /**
     * Shapes inputs for a model, as a tensor.
     *
     * @param model the model
     * @param x inputs {@code [n, 1, frames, coefficients]}
     * @return the model's input
     */
    public static Tensor inputTensor(Layer model, Tensor x) {
        return input(model, x).value();
    }

    /**
     * Float accuracy of any model.
     *
     * @param model the model
     * @param x inputs
     * @param y class ids
     * @return the share correct
     */
    public static double accuracy(Layer model, Tensor x, Tensor y) {
        float[] z = Var.noGrad(() -> model.forward(input(model, x), false)).value().toArray();
        float[] ids = y.toArray();
        int k = z.length / ids.length;
        int ok = 0;
        for (int i = 0; i < ids.length; i++) {
            int best = 0;
            for (int c = 1; c < k; c++) {
                if (z[i * k + c] > z[i * k + best]) {
                    best = c;
                }
            }
            ok += best == (int) ids[i] ? 1 : 0;
        }
        return ok / (double) ids.length;
    }

    /**
     * A recurrent keyword spotter: an optional per-frame convolution over the coefficients
     * (CRNN), a GRU or LSTM over the frames, and a linear classifier on the last state.
     */
    public static final class RecurrentSpotter implements Layer {
        private final int frames;
        private final int coefficients;
        private final int convChannels;
        private final int units;
        private final boolean lstm;
        private final Var convW;
        private final Var convB;
        private final Layer rnn;
        private final Var head;
        private final Var headB;

        /**
         * Creates the model.
         *
         * @param lstm LSTM cells instead of GRU
         * @param convChannels channels of the CRNN convolution (0 for none)
         * @param units recurrent units
         * @param frames frames
         * @param coefficients coefficients per frame
         * @param classes classes
         * @param seed the seed
         */
        public RecurrentSpotter(boolean lstm, int convChannels, int units, int frames, int coefficients, int classes,
                long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            this.frames = frames;
            this.coefficients = coefficients;
            this.convChannels = convChannels;
            this.units = units;
            this.lstm = lstm;
            int in = coefficients;
            if (convChannels > 0) {
                convW = Var.param(Tensor.normal(Shape.of(convChannels, 1, 3, 3), 0, (float) Math.sqrt(2.0 / 9), r));
                convB = Var.param(Tensor.zeros(Shape.of(convChannels)));
                in = convChannels * (coefficients / 2);
            } else {
                convW = null;
                convB = null;
            }
            Recurrent.Spec spec = lstm ? Recurrent.lstm(units) : Recurrent.gru(units);
            rnn = spec.create(Shape.of(frames, in), r);
            head = Var.param(Tensor.normal(Shape.of(units, classes), 0, (float) Math.sqrt(1.0 / units), r));
            headB = Var.param(Tensor.zeros(Shape.of(classes)));
        }

        private int rnnInputs() {
            return convChannels > 0 ? convChannels * (coefficients / 2) : coefficients;
        }

        @Override
        public Var forward(Var x) {
            int n = x.shape().dim(0);
            Var seq;
            if (convW != null) {
                Var h = Functional.conv(x, convW, convB, new int[] {1, 2}, new int[] {1, 1}, new int[] {1, 1},
                        new int[] {1, 1}, 1).relu();                                     // [n, c, frames, coef/2]
                seq = h.permute(0, 2, 1, 3).reshape(n, frames, rnnInputs());
            } else {
                seq = x.reshape(n, frames, coefficients);
            }
            return rnn.forward(seq).matmul(head).add(headB);
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>();
            if (convW != null) {
                p.add(convW);
                p.add(convB);
            }
            p.addAll(rnn.parameters());
            p.add(head);
            p.add(headB);
            return p;
        }

        /** @return the number of parameters */
        public long parameterCount() {
            return parameters().stream().mapToLong(v -> v.value().size()).sum();
        }

        /** @return multiply-adds per inference */
        public long macs() {
            int gates = lstm ? 4 : 3;
            long m = (long) frames * gates * (rnnInputs() + units) * units + (long) units * head.shape().dim(1);
            if (convW != null) {
                m += (long) convChannels * 9 * frames * (coefficients / 2);
            }
            return m;
        }
    }
}
