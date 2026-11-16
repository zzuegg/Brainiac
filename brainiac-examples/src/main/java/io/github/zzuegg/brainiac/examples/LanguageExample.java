package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Embedding;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Benchmark;
import io.github.zzuegg.brainiac.train.Benchmark.Task;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.SplittableRandom;
import java.util.TreeSet;

/**
 * A character-level language model: a small GPT-style transformer that reads Shakespeare
 * one character at a time, learns to predict the next character, and then writes new text.
 * The text is "Tiny Shakespeare" (about 1.1 MB of the plays, public domain), downloaded on
 * first use into {@link #DATA_FILE}. The first 90% is for training and the rest for testing.
 *
 * <p>The loss is reported per character, in nats, so it compares directly with published
 * numbers: a character-level nanoGPT (10.7M parameters, 6 layers of 384) reaches about
 * 1.47 on the test part; a model that only knows how often each letter occurs gets 3.3.
 *
 * <p>{@code ./gradlew :brainiac-examples:run --args="bench shakespeare --device gpu"}
 */
public final class LanguageExample {

    private LanguageExample() {
    }

    /** Where the text is downloaded to, relative to the working directory. */
    public static final Path DATA_FILE = Path.of("data", "tinyshakespeare", "input.txt");

    /** Where the text comes from. */
    public static final String URL =
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt";

    /** Characters the model sees at once. */
    public static final int CONTEXT = 64;

    /**
     * Downloads (if needed) and reads the text.
     *
     * @param file the text file, or where to put it
     * @return the text
     * @throws IOException if it cannot be downloaded or read
     */
    public static String load(Path file) throws IOException {
        if (!Files.exists(file)) {
            Path dir = file.toAbsolutePath().getParent();
            Files.createDirectories(dir);
            System.out.println("downloading " + URL);
            Path part = dir.resolve(file.getFileName() + ".part");
            try (InputStream in = URI.create(URL).toURL().openStream()) {
                Files.copy(in, part, StandardCopyOption.REPLACE_EXISTING);
            }
            Files.move(part, file, StandardCopyOption.REPLACE_EXISTING);
        }
        return Files.readString(file, StandardCharsets.UTF_8);
    }

    /**
     * The characters a text uses, sorted; a character's id is its index here.
     *
     * @param text the text
     * @return the vocabulary
     */
    public static String vocabulary(String text) {
        TreeSet<Character> chars = new TreeSet<>();
        for (int i = 0; i < text.length(); i++) {
            chars.add(text.charAt(i));
        }
        StringBuilder b = new StringBuilder(chars.size());
        chars.forEach(b::append);
        return b.toString();
    }

    /**
     * Cuts a text into windows of {@code length} characters. The features are the
     * characters' ids; the targets are the one-hot ids of the character after each one, so
     * every position is a prediction.
     *
     * @param text the text
     * @param vocabulary the characters, as from {@link #vocabulary}
     * @param length the window length
     * @param stride how far each window starts after the previous one
     * @return features {@code [length]} and targets {@code [length, vocabulary]}
     * @throws IllegalArgumentException if the text has a character missing from the vocabulary
     *     or is too short for one window
     */
    public static Dataset windows(String text, String vocabulary, int length, int stride) {
        if (length < 1 || stride < 1) {
            throw new IllegalArgumentException("length and stride must be at least 1");
        }
        int count = text.length() <= length ? 0 : (text.length() - length - 1) / stride + 1;
        if (count == 0) {
            throw new IllegalArgumentException("a text of " + text.length() + " characters is too short for a window of "
                    + length + " plus the next character");
        }
        int[] ids = ids(text, vocabulary);
        int v = vocabulary.length();
        float[] features = new float[count * length];
        float[] targets = new float[count * length * v];
        for (int w = 0; w < count; w++) {
            int start = w * stride;
            for (int t = 0; t < length; t++) {
                features[w * length + t] = ids[start + t];
                targets[(w * length + t) * v + ids[start + t + 1]] = 1;
            }
        }
        return Dataset.ofRows(features, length, targets, length * v);
    }

    private static int[] ids(String text, String vocabulary) {
        int[] ids = new int[text.length()];
        for (int i = 0; i < ids.length; i++) {
            ids[i] = vocabulary.indexOf(text.charAt(i));
            if (ids[i] < 0) {
                throw new IllegalArgumentException("character '" + text.charAt(i) + "' at " + i
                        + " is not in the vocabulary");
            }
        }
        return ids;
    }

    /**
     * Cross-entropy per character, in nats: the mean over windows and positions.
     *
     * @param length the window length
     * @return the loss
     */
    public static Loss perCharacter(int length) {
        return (p, t) -> Loss.CROSS_ENTROPY.compute(p, t).mul(1f / length);
    }

    /**
     * The language modelling task on a text.
     *
     * @param name the task name
     * @param text the whole text; the first 90% trains, the rest tests
     * @param length the window length
     * @return the task, with training defaults
     */
    public static Task task(String name, String text, int length) {
        String vocabulary = vocabulary(text);
        int cut = (int) (text.length() * 0.9);
        Dataset train = windows(text.substring(0, cut), vocabulary, length, length);
        Dataset test = windows(text.substring(cut), vocabulary, length, length);
        java.util.List<String> names = vocabulary.chars().mapToObj(LanguageExample::printable).toList();
        return Task.classification(name, train, test, Shape.of(length), Shape.of(length, vocabulary.length()))
                .withLoss(perCharacter(length))
                .withClassNames(names);
    }

    /** A character as a readable class name: {@code \n} for a line break, {@code ' '} for a space. */
    private static String printable(int c) {
        return switch (c) {
            case '\n' -> "\\n";
            case '\t' -> "\\t";
            case ' ' -> "' '";
            default -> String.valueOf((char) c);
        };
    }

    /**
     * A small decoder-only transformer: character embeddings, causal blocks with rotary
     * positions, a final norm and a softmax over the characters at every position.
     *
     * @param vocabulary the number of characters
     * @param length the window length
     * @param width the embedding width
     * @param layers the number of transformer blocks
     * @param heads the attention heads per block
     * @return an untrained model: {@code [length]} ids to {@code [length, vocabulary]}
     */
    public static Model model(int vocabulary, int length, int width, int layers, int heads) {
        Model.Builder b = Model.sequential()
                .input(length)
                .add(Embedding.of(vocabulary, width));
        for (int i = 0; i < layers; i++) {
            b.add(TransformerBlock.of(heads).causal(true).rotary(10000).ffnSize(4 * width));
        }
        return b.layer(LayerNorm.of(width))
                .dense(vocabulary, Activation.SOFTMAX)
                .seed(42)
                .build();
    }

    /**
     * Writes text with a trained model, one character at a time: each next character is
     * drawn from the model's prediction, sharpened or flattened by the temperature.
     *
     * @param model a model from {@link #model}, trained
     * @param vocabulary the characters
     * @param prompt the text to continue; at least one character, all in the vocabulary
     * @param count how many characters to add
     * @param temperature below 1 picks likely characters more often, above 1 less often
     * @param seed the random seed
     * @return the prompt followed by the new characters
     */
    public static String generate(Model model, String vocabulary, String prompt, int count, float temperature,
                                  long seed) {
        if (prompt.isEmpty()) {
            throw new IllegalArgumentException("the prompt needs at least one character");
        }
        int length = model.inputShape().dim(0);
        int v = vocabulary.length();
        SplittableRandom random = new SplittableRandom(seed);
        StringBuilder text = new StringBuilder(prompt);
        float[] window = new float[length];
        double[] weights = new double[v];
        for (int n = 0; n < count; n++) {
            int start = Math.max(0, text.length() - length);
            int used = text.length() - start;
            for (int i = 0; i < length; i++) {
                // causal attention: positions after the last real character do not matter
                window[i] = i < used ? vocabulary.indexOf(text.charAt(start + i)) : 0;
            }
            Tensor out = model.predict(Tensor.of(Shape.of(1, length), window, model.device()));
            float[] p = out.toArray();
            out.release();
            int row = (used - 1) * v;
            double total = 0;
            for (int c = 0; c < v; c++) {
                weights[c] = Math.pow(Math.max(p[row + c], 1e-12), 1.0 / temperature);
                total += weights[c];
            }
            double pick = random.nextDouble() * total;
            int c = 0;
            while (c < v - 1 && (pick -= weights[c]) > 0) {
                c++;
            }
            text.append(vocabulary.charAt(c));
        }
        return text.toString();
    }

    /**
     * Trains the default model on Shakespeare and writes a page of new text.
     *
     * @param args ignored
     * @throws IOException if the text cannot be downloaded
     */
    public static void main(String[] args) throws IOException {
        Task task = Tasks.shakespeare();
        Model model = BenchmarkExample.defaultModel(task).to(Device.best());
        System.out.println(model.summary());
        System.out.print(Benchmark.run(task, "transformer", model, t -> t.listener(
                io.github.zzuegg.brainiac.train.Progress.every(1))));
        System.out.println(generate(model, vocabulary(load(DATA_FILE)), "ROMEO:\n", 500, 0.8f, 1));
    }
}
