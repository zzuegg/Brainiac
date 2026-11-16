package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.data.Tokenizer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Embedding;
import io.github.zzuegg.brainiac.nn.KvCache;
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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A language model: a small GPT-style transformer that reads Shakespeare one token at a
 * time, learns to predict the next token, and then writes new text. Tokens come from a
 * {@link Tokenizer}: single characters by default, or word pieces ({@code bpe:1024} and
 * others), which let the same window see three or four times as much text. The text is
 * "Tiny Shakespeare" (about 1.1 MB of the plays, public domain), downloaded on first use
 * into {@link #DATA_FILE}. The first 90% is for training (and for learning the tokenizer)
 * and the rest for testing.
 *
 * <p>Whatever the tokenizer, the loss is reported per character, in nats, so tokenizers
 * compare fairly and with published numbers: a character-level nanoGPT (10.7M parameters,
 * 6 layers of 384) reaches about 1.47 on the test part; a model that only knows how often
 * each letter occurs gets 3.3.
 *
 * <p>{@code ./gradlew :brainiac-examples:run --args="bench shakespeare --device gpu --tokenizers char,bpe:1024"}
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
        return io.github.zzuegg.brainiac.data.CharacterTokenizer.train(text).characters();
    }

    /**
     * Cuts token ids into windows of {@code length}. The features are the ids; the targets
     * are the id of the token after each one, so every position is a prediction.
     *
     * @param ids the text as token ids
     * @param length the window length
     * @param stride how far each window starts after the previous one
     * @return features {@code [length]} and targets {@code [length]}, both token ids
     * @throws IllegalArgumentException if there are too few ids for one window
     */
    public static Dataset windows(int[] ids, int length, int stride) {
        if (length < 1 || stride < 1) {
            throw new IllegalArgumentException("length and stride must be at least 1");
        }
        int count = ids.length <= length ? 0 : (ids.length - length - 1) / stride + 1;
        if (count == 0) {
            throw new IllegalArgumentException(ids.length + " tokens are too few for a window of " + length
                    + " plus the next token");
        }
        float[] features = new float[count * length];
        float[] targets = new float[count * length];
        for (int w = 0; w < count; w++) {
            int start = w * stride;
            for (int t = 0; t < length; t++) {
                features[w * length + t] = ids[start + t];
                targets[w * length + t] = ids[start + t + 1];
            }
        }
        return Dataset.ofRows(features, length, targets, length);
    }

    /**
     * Cross-entropy per character, in nats: the loss per token times the tokens per character.
     *
     * @param length the window length
     * @param charactersPerToken how much text a token covers on average; 1 for characters
     * @return the loss
     */
    public static Loss perCharacter(int length, double charactersPerToken) {
        float scale = (float) (1 / (length * charactersPerToken));
        return (p, t) -> Loss.CROSS_ENTROPY.compute(p, t).mul(scale);
    }

    private static final Map<String, Tokenizer> TOKENIZERS = new ConcurrentHashMap<>();

    /**
     * The tokenizer a spec gives on the training part of a text: built once and kept.
     *
     * @param spec as for {@link Tokenizer#train}, such as {@code char} or {@code bpe:1024}
     * @param text the whole text; the tokenizer learns from the first 90%
     * @return the tokenizer
     */
    public static Tokenizer tokenizer(String spec, String text) {
        String key = spec + "@" + text.length() + "/" + text.hashCode();
        return TOKENIZERS.computeIfAbsent(key, k -> Tokenizer.train(spec, text.substring(0, cut(text))));
    }

    private static int cut(String text) {
        return (int) (text.length() * 0.9);
    }

    /**
     * The character-level language modelling task on a text.
     *
     * @param name the task name
     * @param text the whole text; the first 90% trains, the rest tests
     * @param length the window length
     * @return the task, with training defaults
     */
    public static Task task(String name, String text, int length) {
        return task(name, text, length, "char");
    }

    /**
     * The language modelling task on a text, with tokens from a tokenizer.
     *
     * @param name the task name
     * @param text the whole text; the first 90% trains (the model and the tokenizer), the
     *     rest tests
     * @param length the window length, in tokens
     * @param tokenizerSpec as for {@link Tokenizer#train}, such as {@code char} or
     *     {@code bpe:1024}
     * @return the task; the targets are token ids
     */
    public static Task task(String name, String text, int length, String tokenizerSpec) {
        Tokenizer tokenizer = tokenizer(tokenizerSpec, text);
        int cut = cut(text);
        String testText = text.substring(cut);
        int[] testIds = tokenizer.encode(testText);
        Dataset train = windows(tokenizer.encode(text.substring(0, cut)), length, length);
        Dataset test = windows(testIds, length, length);
        double charsPerToken = (double) testText.length() / testIds.length;
        List<String> names = new ArrayList<>(tokenizer.size());
        for (int id = 0; id < tokenizer.size(); id++) {
            names.add(printable(tokenizer.decode(id)));
        }
        return Task.classification(name, train, test, Shape.of(length), Shape.of(length, tokenizer.size()))
                .withLoss(perCharacter(length, charsPerToken))
                .withClassNames(names);
    }

    /** A token as a readable class name: {@code \n} for a line break, {@code ' '} for a space. */
    private static String printable(String token) {
        if (token.equals(" ")) {
            return "' '";
        }
        String s = token.replace("\n", "\\n").replace("\t", "\\t");
        return s.startsWith(" ") || s.endsWith(" ") ? "'" + s + "'" : s;
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
     * Writes text with a trained character model; see
     * {@link #generate(Model, Tokenizer, String, int, float, long)}.
     *
     * @param model a model from {@link #model}, trained on characters
     * @param vocabulary the characters
     * @param prompt the text to continue; at least one character
     * @param count how many characters to add
     * @param temperature below 1 picks likely characters more often, above 1 less often
     * @param seed the random seed
     * @return the prompt followed by the new characters
     */
    public static String generate(Model model, String vocabulary, String prompt, int count, float temperature,
                                  long seed) {
        return generate(model, new io.github.zzuegg.brainiac.data.CharacterTokenizer(vocabulary), prompt, count,
                temperature, seed);
    }

    /**
     * Writes text with a trained model, one token at a time: each next token is drawn from
     * the model's prediction, sharpened or flattened by the temperature.
     *
     * @param model a model from {@link #model}, trained
     * @param tokenizer the tokenizer it was trained with
     * @param prompt the text to continue; at least one token
     * @param count how many tokens to add
     * @param temperature below 1 picks likely tokens more often, above 1 less often
     * @param seed the random seed
     * @return the prompt followed by the new text
     */
    public static String generate(Model model, Tokenizer tokenizer, String prompt, int count, float temperature,
                                  long seed) {
        int[] start = tokenizer.encode(prompt);
        if (start.length == 0) {
            throw new IllegalArgumentException("the prompt needs at least one token");
        }
        int length = model.inputShape().dim(0);
        int v = model.outputShape().dim(-1);
        SplittableRandom random = new SplittableRandom(seed);
        int[] ids = java.util.Arrays.copyOf(start, start.length + count);
        int size = start.length;
        double[] weights = new double[v];
        // keys and values of earlier tokens are kept, so each new token is one cheap step
        KvCache cache = KvCache.of(model);
        Tensor out = cache.append(java.util.Arrays.copyOfRange(start, Math.max(0, size - length), size));
        for (int n = 0; n < count; n++) {
            float[] p = out.toArray();
            out.release();
            ids[size] = sample(p, 0, v, temperature, random, weights);
            if (n + 1 < count) {
                out = cache.append(ids[size]);
            }
            size++;
        }
        return prompt + tokenizer.decode(java.util.Arrays.copyOfRange(ids, start.length, size));
    }

    /** Draws a class from probabilities {@code p[offset .. offset + v)} at a temperature. */
    static int sample(float[] p, int offset, int v, float temperature, SplittableRandom random, double[] weights) {
        double total = 0;
        for (int c = 0; c < v; c++) {
            weights[c] = Math.pow(Math.max(p[offset + c], 1e-12), 1.0 / temperature);
            total += weights[c];
        }
        double pick = random.nextDouble() * total;
        int c = 0;
        while (c < v - 1 && (pick -= weights[c]) > 0) {
            c++;
        }
        return c;
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
        System.out.println(generate(model, tokenizer("char", load(DATA_FILE)), "ROMEO:\n", 500, 0.8f, 1));
    }
}
