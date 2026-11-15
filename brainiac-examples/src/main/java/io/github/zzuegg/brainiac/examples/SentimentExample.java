package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Embedding;
import io.github.zzuegg.brainiac.nn.MaskedPool;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.Node;
import io.github.zzuegg.brainiac.nn.PaddingMask;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.nn.SequencePool;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Benchmark;
import io.github.zzuegg.brainiac.train.Benchmark.Report;
import io.github.zzuegg.brainiac.train.Benchmark.Task;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * Sentiment of short made-up reviews, with word embeddings learned from scratch.
 *
 * <p>Each review is a list of word ids padded with 0 to {@link #LENGTH} words, such as
 * "the plot was not very good" (negative) or "it is dull but the acting is great"
 * (positive: the clause after "but" wins). Counting good and bad words is not enough:
 * the model has to read "not" and "but" in order. Three models are compared:
 * <ul>
 *   <li><b>bag</b>: the mean of the word vectors, which ignores word order;</li>
 *   <li><b>bilstm</b>: a bidirectional LSTM over the vectors, pooled over the real words;</li>
 *   <li><b>transformer</b>: a transformer block that ignores the padding (key mask).</li>
 * </ul>
 * Afterwards it prints the words whose learned vectors lie closest to "good" and "awful".
 *
 * <p>{@code ./gradlew :brainiac-examples:run --args=sentiment}
 */
public final class SentimentExample {

    private SentimentExample() {
    }

    /** Words per review, after padding. */
    public static final int LENGTH = 12;

    /** The words; the id of a word is its index, and 0 is padding. */
    public static final List<String> WORDS = List.of("<pad>",
            "good", "great", "excellent", "wonderful", "fun", "lovely",
            "bad", "awful", "boring", "terrible", "dull", "poor",
            "not", "never",
            "very", "really", "so",
            "but",
            "the", "movie", "film", "plot", "acting", "story", "was", "is", "it", "this", "and");

    /** The number of word ids, including padding. */
    public static final int VOCABULARY = WORDS.size();

    private static final int[] POSITIVE = {1, 2, 3, 4, 5, 6};
    private static final int[] NEGATIVE = {7, 8, 9, 10, 11, 12};
    private static final int[] NEGATORS = {13, 14};
    private static final int[] INTENSIFIERS = {15, 16, 17};
    private static final int BUT = 18;
    private static final int[] SUBJECTS = {20, 21, 22, 23, 24, 27};
    private static final int[] VERBS = {25, 26};

    /** A review and its sentiment. */
    private record Review(List<Integer> words, boolean positive) {
    }

    private static int pick(int[] words, SplittableRandom random) {
        return words[random.nextInt(words.length)];
    }

    /** "[the] subject verb [not] [very] adjective", and whether it is positive. */
    private static Review clause(SplittableRandom random) {
        List<Integer> w = new ArrayList<>();
        if (random.nextBoolean()) {
            w.add(random.nextBoolean() ? 19 : 28);    // the / this
        }
        w.add(pick(SUBJECTS, random));
        w.add(pick(VERBS, random));
        boolean negated = random.nextDouble() < 0.4;
        if (negated) {
            w.add(pick(NEGATORS, random));
        }
        if (random.nextDouble() < 0.3) {
            w.add(pick(INTENSIFIERS, random));
        }
        boolean positive = random.nextBoolean();
        w.add(pick(positive ? POSITIVE : NEGATIVE, random));
        return new Review(w, positive != negated);
    }

    private static Review review(SplittableRandom random) {
        while (true) {
            Review first = clause(random);
            if (random.nextDouble() < 0.35) {
                Review second = clause(random);
                List<Integer> w = new ArrayList<>(first.words());
                w.add(BUT);
                w.addAll(second.words());
                if (w.size() <= LENGTH) {
                    return new Review(w, second.positive());
                }
            } else {
                return first;
            }
        }
    }

    /**
     * Random reviews: word ids padded with 0 at the end, and one-hot targets
     * (negative, positive).
     *
     * @param samples the number of reviews
     * @param seed the random seed
     * @return the reviews
     */
    public static Dataset reviews(int samples, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        float[][] x = new float[samples][LENGTH];
        float[][] y = new float[samples][2];
        for (int s = 0; s < samples; s++) {
            Review r = review(random);
            for (int i = 0; i < r.words().size(); i++) {
                x[s][i] = r.words().get(i);
            }
            y[s][r.positive() ? 1 : 0] = 1;
        }
        return Dataset.of(x, y);
    }

    /**
     * Turns word ids back into text.
     *
     * @param ids word ids, padding ignored
     * @return the words
     */
    public static String text(float[] ids) {
        return Arrays.stream(toInts(ids)).filter(i -> i != 0).mapToObj(WORDS::get).collect(Collectors.joining(" "));
    }

    private static int[] toInts(float[] ids) {
        return java.util.stream.IntStream.range(0, ids.length).map(i -> Math.round(ids[i])).toArray();
    }

    /**
     * The mean of the word vectors (padding left out), then a small dense network. It
     * cannot see word order.
     *
     * @return an untrained model
     */
    public static Model bag() {
        Node ids = Node.input(LENGTH);
        Node mask = ids.layer(new PaddingMask(0));
        Node out = ids.add(Embedding.of(VOCABULARY, 16).paddingIndex(0))
                .merge(MaskedPool.of(SequencePool.Kind.MEAN), mask)
                .dense(32, Activation.RELU)
                .dense(2, Activation.SOFTMAX);
        return Model.graph(ids, out).seed(1).build();
    }

    /**
     * A bidirectional LSTM over the word vectors, then the mean over the real words.
     *
     * @return an untrained model
     */
    public static Model bilstm() {
        Node ids = Node.input(LENGTH);
        Node mask = ids.layer(new PaddingMask(0));
        Node out = ids.add(Embedding.of(VOCABULARY, 16).paddingIndex(0))
                .add(Recurrent.lstm(24).returnSequences(true).bidirectional(Recurrent.Merge.CONCAT))
                .merge(MaskedPool.of(SequencePool.Kind.MEAN), mask)
                .dense(2, Activation.SOFTMAX);
        return Model.graph(ids, out).seed(2).build();
    }

    /**
     * Word vectors scaled by the square root of their size plus positions, two transformer
     * blocks that ignore padding, then the mean over the real words.
     *
     * @return an untrained model
     */
    public static Model transformer() {
        Node ids = Node.input(LENGTH);
        Node mask = ids.layer(new PaddingMask(0));
        Node out = ids.add(Embedding.of(VOCABULARY, 32).paddingIndex(0).scaleBySqrtDimension(true)
                        .init((shape, fanIn, fanOut, random) -> Tensor.normal(shape, 0, 0.18f, random)))
                .positionalEncoding()
                .merge(TransformerBlock.of(4).ffnSize(64).keyMask(true), mask)
                .merge(TransformerBlock.of(4).ffnSize(64).keyMask(true), mask)
                .merge(MaskedPool.of(SequencePool.Kind.MEAN), mask)
                .dense(2, Activation.SOFTMAX);
        return Model.graph(ids, out).seed(3).build();
    }

    /**
     * The benchmark task: 4,000 reviews to train on and 1,000 to test.
     *
     * @return the task
     */
    public static Task task() {
        return Task.classification("sentiment", reviews(4000, 1), reviews(1000, 2), Shape.of(LENGTH), Shape.of(2))
                .describedAs("positive or negative made-up review, with \"not\" and \"but\"")
                .withClassNames(List.of("negative", "positive"))
                .withDefaults(t -> t.optimizer(io.github.zzuegg.brainiac.train.Optimizer.adam(0.003f))
                        .clipGradientNorm(1).batchSize(32).epochs(12));
    }

    /**
     * Trains and compares the three models, then shows what the embedding learned.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        Task task = task();
        float[] sample = task.test().featuresOf(0);
        System.out.println("Example review: \"" + text(sample) + "\" -> "
                + (task.test().targetsOf(0)[1] > 0 ? "positive" : "negative"));
        Map<String, Supplier<Model>> models = new LinkedHashMap<>();
        models.put("bag of words", SentimentExample::bag);
        models.put("bi-LSTM", SentimentExample::bilstm);
        models.put("transformer", SentimentExample::transformer);
        List<Report> reports = new ArrayList<>();
        List<Model> trained = new ArrayList<>();
        Model best = null;
        double bestAccuracy = -1;
        for (Map.Entry<String, Supplier<Model>> e : models.entrySet()) {
            Model model = e.getValue().get();
            Report r = Benchmark.run(task, e.getKey(), model);
            System.out.printf("%-14s %.1f%%%n", e.getKey(), 100 * r.accuracy());
            reports.add(r);
            trained.add(model);
            if (r.accuracy() > bestAccuracy) {
                bestAccuracy = r.accuracy();
                best = model;
            }
        }
        System.out.println();
        System.out.println(Benchmark.table(reports));
        // the bag-of-words model has to put all meaning into the vectors, so its neighbours read best
        Model bag = null;
        for (int i = 0; i < reports.size(); i++) {
            if (reports.get(i).model().equals("bag of words")) {
                bag = trained.get(i);
            }
        }
        for (Model m : List.of(bag, best)) {
            Embedding embedding = (Embedding) m.layers().stream().filter(Embedding.class::isInstance).findFirst()
                    .orElseThrow();
            System.out.println(m == bag ? "Word neighbours in the bag-of-words model:" : "In the best model:");
            for (String word : List.of("good", "awful", "not", "but")) {
                int[] near = embedding.nearest(WORDS.indexOf(word), 4);
                System.out.println("  closest to \"" + word + "\": "
                        + Arrays.stream(near).mapToObj(WORDS::get).collect(Collectors.joining(", ")));
            }
        }
        String[] tries = {"the movie was not bad", "it is great but the plot is dull", "this film is really boring",
            "the acting was never poor but the story is not fun"};
        for (String sentence : tries) {
            float[] ids = new float[LENGTH];
            String[] words = sentence.split(" ");
            for (int i = 0; i < words.length; i++) {
                ids[i] = WORDS.indexOf(words[i]);
            }
            float p = best.predict(Tensor.of(Shape.of(1, LENGTH), ids)).toArray()[1];
            System.out.printf("%-52s %s (%.0f%% positive)%n", "\"" + sentence + "\"", p > 0.5 ? "positive"
                    : "negative", 100 * p);
        }
    }
}
