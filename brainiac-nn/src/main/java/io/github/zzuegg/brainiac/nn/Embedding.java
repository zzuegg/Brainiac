package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Looks up a learned vector for every token id: input {@code [steps...]} of whole-number
 * ids in {@code [0, vocabulary)}, output {@code [steps..., dimension]}. Only the rows of
 * ids that appear in a batch receive gradient.
 *
 * <pre>{@code
 * Model model = Model.sequential()
 *         .input(20)                                       // 20 token ids per sample
 *         .add(Embedding.of(5000, 64).paddingIndex(0))     // -> [20, 64]
 *         .lstm(64, false)
 *         .dense(2, Activation.SOFTMAX)
 *         .build();
 * }</pre>
 *
 * <p>Options: a padding id whose vector is always zero and never trained, a maximum norm
 * that rescales longer vectors, scaling by {@code sqrt(dimension)} (as transformers do),
 * the initializer, pretrained vectors, and freezing (not trained, still saved).
 * {@link #nearest} finds the most similar tokens, for inspecting what was learned.
 */
public final class Embedding implements Layer {

    private final int vocabulary;
    private final int dimension;
    private final int paddingIndex;
    private final float maxNorm;
    private final boolean scale;
    private final boolean trainable;
    private final Var weight;

    private Embedding(int vocabulary, int dimension, int paddingIndex, float maxNorm, boolean scale,
                      boolean trainable, Tensor weights) {
        if (vocabulary < 1 || dimension < 1) {
            throw new IllegalArgumentException("vocabulary and dimension must be at least 1");
        }
        if (paddingIndex < -1 || paddingIndex >= vocabulary) {
            throw new IllegalArgumentException("paddingIndex " + paddingIndex + " is not a token id below "
                    + vocabulary);
        }
        if (!weights.shape().equals(Shape.of(vocabulary, dimension))) {
            throw new IllegalArgumentException("weights must have shape [" + vocabulary + ", " + dimension
                    + "], got " + weights.shape());
        }
        this.vocabulary = vocabulary;
        this.dimension = dimension;
        this.paddingIndex = paddingIndex;
        this.maxNorm = maxNorm;
        this.scale = scale;
        this.trainable = trainable;
        if (paddingIndex >= 0) {
            float[] w = weights.toArray();
            Arrays.fill(w, paddingIndex * dimension, (paddingIndex + 1) * dimension, 0f);
            weights = Tensor.of(weights.shape(), w, weights.device());
        }
        this.weight = trainable ? Var.param(weights) : Var.of(weights);
    }

    /**
     * Starts an embedding of {@code vocabulary} tokens into {@code dimension} values each,
     * initialized from {@code N(0, 1)}.
     *
     * @param vocabulary the number of token ids
     * @param dimension the size of each vector
     * @return the spec
     */
    public static Spec of(int vocabulary, int dimension) {
        return new Spec(vocabulary, dimension);
    }

    /**
     * Starts an embedding from pretrained vectors, for example from word2vec or GloVe.
     *
     * @param weights {@code [vocabulary, dimension]}
     * @return the spec; add {@link Spec#frozen()} to keep the vectors fixed
     */
    public static Spec pretrained(Tensor weights) {
        if (weights.rank() != 2) {
            throw new IllegalArgumentException("pretrained vectors must be [vocabulary, dimension], got "
                    + weights.shape());
        }
        Spec s = new Spec(weights.shape().dim(0), weights.shape().dim(1));
        s.weights = weights;
        return s;
    }

    /** Builds an {@link Embedding}; every setter returns the spec. */
    public static final class Spec implements LayerSpec {
        private final int vocabulary;
        private final int dimension;
        private int paddingIndex = -1;
        private float maxNorm;
        private boolean scale;
        private boolean trainable = true;
        private Init init = (shape, fanIn, fanOut, random) -> Tensor.normal(shape, 0, 1, random);
        private Tensor weights;

        private Spec(int vocabulary, int dimension) {
            this.vocabulary = vocabulary;
            this.dimension = dimension;
        }

        /**
         * @param index the token id used for padding: its vector is zero and not trained
         * @return this spec
         */
        public Spec paddingIndex(int index) {
            this.paddingIndex = index;
            return this;
        }

        /**
         * @param norm rescale looked-up vectors longer than this to this length; 0 for none
         * @return this spec
         */
        public Spec maxNorm(float norm) {
            if (norm < 0) {
                throw new IllegalArgumentException("maxNorm must not be negative");
            }
            this.maxNorm = norm;
            return this;
        }

        /**
         * @param on whether to multiply vectors by {@code sqrt(dimension)}, as transformer
         *     inputs usually are
         * @return this spec
         */
        public Spec scaleBySqrtDimension(boolean on) {
            this.scale = on;
            return this;
        }

        /** @return this spec, with vectors that are not trained */
        public Spec frozen() {
            this.trainable = false;
            return this;
        }

        /**
         * @param initializer how to fill the vectors; ignored for pretrained vectors
         * @return this spec
         */
        public Spec init(Init initializer) {
            this.init = Objects.requireNonNull(initializer, "initializer");
            return this;
        }

        @Override
        public Shape outputShape(Shape input) {
            int[] dims = Arrays.copyOf(input.toArray(), input.rank() + 1);
            dims[input.rank()] = dimension;
            return Shape.of(dims);
        }

        @Override
        public Embedding create(Shape input, SplittableRandom random) {
            Tensor w = weights != null ? weights
                    : init.create(Shape.of(vocabulary, dimension), vocabulary, dimension, random);
            return new Embedding(vocabulary, dimension, paddingIndex, maxNorm, scale, trainable, w);
        }
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        Tensor ids = input.value().reshape(Math.toIntExact(s.size()));
        Var out = weight.gather(0, ids);
        if (maxNorm > 0) {
            Var norm = out.square().sum(1, true).sqrt();
            Var factor = norm.add(1e-7f).pow(-1).mul(maxNorm).minimum(1f);
            out = out.mul(factor);
        }
        if (paddingIndex >= 0) {
            Tensor pad = Tensor.full(Shape.of(), paddingIndex, ids.dtype(), ids.device());
            Tensor keep = ids.equalTo(pad).neg().add(1).reshape(ids.shape().dim(0), 1);
            out = out.mul(keep);
        }
        if (scale) {
            out = out.mul((float) Math.sqrt(dimension));
        }
        int[] dims = Arrays.copyOf(s.toArray(), s.rank() + 1);
        dims[s.rank()] = dimension;
        return out.reshape(dims);
    }

    @Override
    public Shape outputShape(Shape input) {
        int[] dims = Arrays.copyOf(input.toArray(), input.rank() + 1);
        dims[input.rank()] = dimension;
        return Shape.of(dims);
    }

    @Override
    public List<Var> parameters() {
        return trainable ? List.of(weight) : List.of();
    }

    @Override
    public List<Var> buffers() {
        return trainable ? List.of() : List.of(weight);
    }

    /** @return the vectors, {@code [vocabulary, dimension]} */
    public Tensor weights() {
        return weight.value();
    }

    /** @return the number of token ids */
    public int vocabulary() {
        return vocabulary;
    }

    /** @return the size of each vector */
    public int dimension() {
        return dimension;
    }

    /**
     * Finds the tokens whose vectors point most nearly the same way as a token's (cosine
     * similarity), for inspecting what the embedding learned.
     *
     * @param token a token id
     * @param count how many to return
     * @return the most similar other token ids, most similar first
     */
    public int[] nearest(int token, int count) {
        float[] w = weight.value().toArray();
        double[] norms = new double[vocabulary];
        for (int v = 0; v < vocabulary; v++) {
            double n = 0;
            for (int d = 0; d < dimension; d++) {
                n += (double) w[v * dimension + d] * w[v * dimension + d];
            }
            norms[v] = Math.sqrt(n) + 1e-12;
        }
        double[] similarity = new double[vocabulary];
        for (int v = 0; v < vocabulary; v++) {
            double dot = 0;
            for (int d = 0; d < dimension; d++) {
                dot += (double) w[token * dimension + d] * w[v * dimension + d];
            }
            similarity[v] = dot / (norms[token] * norms[v]);
        }
        return java.util.stream.IntStream.range(0, vocabulary)
                .filter(v -> v != token && v != paddingIndex)
                .boxed()
                .sorted((a, b) -> Double.compare(similarity[b], similarity[a]))
                .limit(count)
                .mapToInt(Integer::intValue)
                .toArray();
    }

    @Override
    public String toString() {
        return "Embedding(" + vocabulary + " x " + dimension + (paddingIndex >= 0 ? ", pad " + paddingIndex : "")
                + (trainable ? "" : ", frozen") + ")";
    }

    static Config write(Embedding e) {
        return Config.builder().put("vocabulary", e.vocabulary).put("dimension", e.dimension)
                .put("paddingIndex", e.paddingIndex).put("maxNorm", e.maxNorm).put("scale", e.scale)
                .put("trainable", e.trainable).build();
    }

    static Embedding read(Config c) {
        int v = c.getInt("vocabulary");
        int d = c.getInt("dimension");
        return new Embedding(v, d, c.getInt("paddingIndex"), c.getFloat("maxNorm"), c.getBoolean("scale"),
                c.getBoolean("trainable"), Tensor.zeros(Shape.of(v, d)));
    }
}
