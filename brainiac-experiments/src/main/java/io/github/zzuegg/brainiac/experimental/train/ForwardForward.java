package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The Forward-Forward algorithm (Hinton 2022,
 * <a href="https://arxiv.org/abs/2212.13345">arXiv:2212.13345</a>): every layer is trained
 * by its own local objective, with no backward pass between layers. Each layer should have
 * high <em>goodness</em> (mean squared activity) on positive data, a sample with its true
 * label embedded, and low goodness on negative data, a sample with a wrong label (or
 * another corruption from a {@link NegativeData} strategy). Layer inputs are normalised to
 * unit length so a layer cannot just pass the previous layer's goodness on.
 *
 * <p>To classify, the network is run once per candidate label and the label with the most
 * accumulated goodness wins.
 */
public final class ForwardForward implements LocalLearner {

    /** Makes the negative examples for a positive batch. */
    @FunctionalInterface
    public interface NegativeData {

        /**
         * @param inputs the positive inputs, {@code [batch, features]}
         * @param labels their true class ids
         * @param model the model being trained, for strategies that use its predictions
         * @return the negative inputs and the labels to embed with them
         */
        Tensor[] make(Tensor inputs, Tensor labels, ForwardForward model);

        /**
         * The same inputs with a uniformly random wrong label (Hinton's supervised setting).
         *
         * @param seed the random seed
         * @return the strategy
         */
        static NegativeData wrongLabel(long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            return (x, y, m) -> {
                float[] ids = y.toArray();
                for (int i = 0; i < ids.length; i++) {
                    ids[i] = (Math.round(ids[i]) + 1 + r.nextInt(m.classes - 1)) % m.classes;
                }
                return new Tensor[] {x, Tensor.of(y.shape(), ids)};
            };
        }

        /**
         * The same inputs with the wrong label the model currently finds most plausible:
         * harder negatives that sharpen the decision boundary.
         *
         * @return the strategy
         */
        static NegativeData hardestLabel() {
            return (x, y, m) -> {
                float[] scores = m.predict(x).toArray();
                float[] ids = y.toArray();
                for (int i = 0; i < ids.length; i++) {
                    int truth = Math.round(ids[i]);
                    int best = -1;
                    for (int c = 0; c < m.classes; c++) {
                        if (c != truth && (best < 0 || scores[i * m.classes + c] > scores[i * m.classes + best])) {
                            best = c;
                        }
                    }
                    ids[i] = best;
                }
                return new Tensor[] {x, Tensor.of(y.shape(), ids)};
            };
        }

        /**
         * Hybrid inputs with the true label: each feature comes from the sample or from a
         * random other sample, by a random mask. Hinton's unsupervised negatives, which have
         * the right short-range statistics but wrong long-range structure.
         *
         * @param seed the random seed
         * @return the strategy
         */
        static NegativeData hybrid(long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            return (x, y, m) -> {
                int n = x.shape().dim(0);
                int f = x.shape().dim(1);
                float[] a = x.toArray();
                float[] out = new float[a.length];
                for (int i = 0; i < n; i++) {
                    int j = r.nextInt(n);
                    for (int k = 0; k < f; k++) {
                        out[i * f + k] = r.nextBoolean() ? a[i * f + k] : a[j * f + k];
                    }
                }
                return new Tensor[] {Tensor.of(x.shape(), out), y};
            };
        }
    }

    private final int classes;
    private final float labelScale;
    private final float threshold;
    private final List<Var> weights = new ArrayList<>();
    private final List<Var> biases = new ArrayList<>();
    private final Optimizer optimizer;
    private final NegativeData negatives;

    private ForwardForward(Builder b) {
        this.classes = b.classes;
        this.labelScale = b.labelScale;
        this.threshold = b.threshold;
        this.optimizer = b.optimizer;
        this.negatives = b.negatives == null ? NegativeData.wrongLabel(b.seed + 1) : b.negatives;
        SplittableRandom r = new SplittableRandom(b.seed);
        int in = b.inputs + classes;
        for (int size : b.hidden) {
            weights.add(Var.param(Tensor.normal(Shape.of(in, size), 0, (float) Math.sqrt(2.0 / in), r)));
            biases.add(Var.param(Tensor.zeros(Shape.of(1, size))));
            in = size;
        }
    }

    /**
     * @param inputs features per sample
     * @param classes the number of classes
     * @return a builder
     */
    public static Builder builder(int inputs, int classes) {
        return new Builder(inputs, classes);
    }

    /** Settings for a {@link ForwardForward} network. */
    public static final class Builder {
        private final int inputs;
        private final int classes;
        private int[] hidden = {64, 64};
        private float labelScale = 1;
        private float threshold = 2;
        private long seed = 1;
        private Optimizer optimizer;
        private NegativeData negatives;

        private Builder(int inputs, int classes) {
            if (classes < 2) {
                throw new IllegalArgumentException("need at least two classes");
            }
            this.inputs = inputs;
            this.classes = classes;
        }

        /**
         * @param sizes the hidden layer sizes
         * @return this builder
         */
        public Builder hidden(int... sizes) {
            this.hidden = sizes.clone();
            return this;
        }

        /**
         * @param scale the value of the embedded label's one-hot entry
         * @return this builder
         */
        public Builder labelScale(float scale) {
            this.labelScale = scale;
            return this;
        }

        /**
         * @param threshold the goodness separating positive from negative
         * @return this builder
         */
        public Builder threshold(float threshold) {
            this.threshold = threshold;
            return this;
        }

        /**
         * @param seed the random seed
         * @return this builder
         */
        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        /**
         * @param optimizer the optimizer for every layer (Adam with 0.01 by default)
         * @return this builder
         */
        public Builder optimizer(Optimizer optimizer) {
            this.optimizer = optimizer;
            return this;
        }

        /**
         * @param negatives how negative data is made (wrong labels by default)
         * @return this builder
         */
        public Builder negatives(NegativeData negatives) {
            this.negatives = negatives;
            return this;
        }

        /** @return the network */
        public ForwardForward build() {
            if (optimizer == null) {
                optimizer = Optimizer.adam(0.01f);
            }
            return new ForwardForward(this);
        }
    }

    /** @return the number of classes */
    public int classes() {
        return classes;
    }

    /** @return every layer's weight and bias */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (int l = 0; l < weights.size(); l++) {
            p.add(weights.get(l));
            p.add(biases.get(l));
        }
        return p;
    }

    private Tensor embed(Tensor x, Tensor labels) {
        return Tensor.concat(1, LearningMath.oneHot(labels, classes).mul(labelScale), x);
    }

    /** Each layer's goodness ({@code [batch, 1]}), with the layer's input detached. */
    private List<Var> goodness(Tensor input) {
        List<Var> out = new ArrayList<>();
        Tensor h = input;
        for (int l = 0; l < weights.size(); l++) {
            Var in = Var.of(normalise(h));
            Var a = in.matmul(weights.get(l)).add(biases.get(l)).relu();
            out.add(a.square().mean(1, true));
            h = a.value();
        }
        return out;
    }

    private static Tensor normalise(Tensor h) {
        return h.div(h.square().mean(1, true).sqrt().add(1e-4f));
    }

    @Override
    public float trainBatch(Tensor inputs, Tensor labels) {
        Tensor[] neg = negatives.make(inputs, labels, this);
        List<Var> pos = goodness(embed(inputs, labels));
        List<Var> negG = goodness(embed(neg[0], neg[1]));
        Var loss = null;
        for (int l = 0; l < pos.size(); l++) {
            // softplus(-(g+ - theta)) + softplus(g- - theta): local to layer l
            Var layer = pos.get(l).neg().add(threshold).softplus().mean()
                    .add(negG.get(l).sub(threshold).softplus().mean());
            loss = loss == null ? layer : loss.add(layer);
        }
        loss.backward();
        float value = loss.value().item() / pos.size();
        optimizer.step(parameters());
        return value;
    }

    /**
     * Goodness summed over layers for every candidate label.
     *
     * @param inputs {@code [batch, features]}
     * @return {@code [batch, classes]} accumulated goodness
     */
    @Override
    public Tensor predict(Tensor inputs) {
        int batch = inputs.shape().dim(0);
        Tensor[] perClass = new Tensor[classes];
        for (int c = 0; c < classes; c++) {
            Tensor ids = Tensor.full(Shape.of(batch), c);
            Tensor total = null;
            for (Var g : goodness(embed(inputs, ids))) {
                total = total == null ? g.value() : total.add(g.value());
            }
            perClass[c] = total;
        }
        return Tensor.concat(1, perClass);
    }
}
