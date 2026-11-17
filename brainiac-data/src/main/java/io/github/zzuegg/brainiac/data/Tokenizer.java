package io.github.zzuegg.brainiac.data;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Turns text into whole-number token ids for a model, and ids back into text. A tokenizer
 * is not a neural network: it is a fixed table, built once from a text by counting, and
 * the model's embedding layer then learns what each token means.
 *
 * <p>The kinds trade vocabulary size against sequence length:
 * <ul>
 * <li>{@code char}: one token per character. A tiny vocabulary, long sequences.
 * <li>{@code byte}: one token per UTF-8 byte, always 256 tokens; any text works.
 * <li>{@code word}: the most frequent words (with their leading space) as tokens; other
 *     words fall back to bytes.
 * <li>{@code bpe}: byte-pair encoding (GPT): merges the most frequent neighbouring pair.
 * <li>{@code wordpiece}: like BPE, but merges the pair that is most frequent relative to its
 *     parts (BERT); encodes by taking the longest known piece first.
 * <li>{@code unigram}: starts from many candidate pieces and prunes those the text needs
 *     least under a probabilistic model (SentencePiece, T5); encodes the most likely split.
 * </ul>
 * Every kind except {@code char} starts from the 256 bytes, so no text is ever unknown.
 *
 * <pre>{@code
 * Tokenizer t = Tokenizer.train("bpe:1024", text);
 * int[] ids = t.encode("To be, or not to be");
 * String back = t.decode(ids);
 * }</pre>
 */
public interface Tokenizer {

    /** The kinds {@link #train} knows. */
    List<String> KINDS = List.of("char", "byte", "word", "bpe", "wordpiece", "unigram");

    /**
     * Splits a text into token ids.
     *
     * @param text the text
     * @return the ids, each in {@code [0, size())}
     */
    int[] encode(String text);

    /**
     * Turns token ids back into text.
     *
     * @param ids the ids
     * @return the text
     */
    String decode(int... ids);

    /** @return the number of distinct tokens */
    int size();

    /** @return the kind, one of {@link #KINDS} */
    String kind();

    /**
     * Writes the tokenizer to a text file that {@link #load} reads back.
     *
     * @param path the file
     * @throws IOException if it cannot be written
     */
    void save(Path path) throws IOException;

    /**
     * Builds a tokenizer from a text.
     *
     * @param spec the kind, with the vocabulary size where it applies: {@code char},
     *     {@code byte}, {@code word:4096}, {@code bpe:1024}, {@code wordpiece:1024},
     *     {@code unigram:1024}
     * @param text the text to learn from
     * @return the tokenizer
     * @throws IllegalArgumentException for an unknown kind or a missing size
     */
    static Tokenizer train(String spec, String text) {
        String[] parts = spec.trim().toLowerCase(java.util.Locale.ROOT).split(":");
        String kind = parts[0];
        int size = -1;
        if (parts.length > 1) {
            try {
                size = Integer.parseInt(parts[1]);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("tokenizer size must be a number, got " + spec);
            }
        }
        if (size < 0 && !kind.equals("char") && !kind.equals("byte")) {
            throw new IllegalArgumentException(kind + " needs a vocabulary size, like " + kind + ":1024");
        }
        return switch (kind) {
            case "char" -> CharacterTokenizer.train(text);
            case "byte" -> ByteTokenizer.INSTANCE;
            case "word" -> WordTokenizer.train(text, size);
            case "bpe" -> BytePairTokenizer.train(text, size);
            case "wordpiece" -> WordPieceTokenizer.train(text, size);
            case "unigram" -> UnigramTokenizer.train(text, size);
            default -> throw new IllegalArgumentException("unknown tokenizer " + kind + "; known: " + KINDS);
        };
    }

    /**
     * Reads a tokenizer written by {@link #save}.
     *
     * @param path the file
     * @return the tokenizer
     * @throws IOException if it cannot be read or is not a tokenizer file
     */
    static Tokenizer load(Path path) throws IOException {
        List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
        if (lines.isEmpty() || !lines.get(0).startsWith(Pretokenizer.HEADER)) {
            throw new IOException(path + " is not a tokenizer file");
        }
        String kind = lines.get(0).split(" ")[1];
        List<String> body = Pretokenizer.body(lines, path);
        try {
            return switch (kind) {
                case "char" -> CharacterTokenizer.read(body);
                case "byte" -> ByteTokenizer.INSTANCE;
                case "word" -> WordTokenizer.read(body);
                case "bpe" -> BytePairTokenizer.read(body);
                case "wordpiece" -> WordPieceTokenizer.read(body);
                case "unigram" -> UnigramTokenizer.read(body);
                default -> {
                    Reader reader = Pretokenizer.READERS.get(kind);
                    if (reader == null) {
                        throw new IOException(path + " has an unknown tokenizer kind " + kind);
                    }
                    yield reader.read(body);
                }
            };
        } catch (RuntimeException e) {
            throw new IOException(path + ": " + e.getMessage(), e);
        }
    }

    /** Reads the body of a tokenizer file, the lines after the header. */
    @FunctionalInterface
    interface Reader {
        /**
         * @param lines the lines {@link #write} was given
         * @return the tokenizer
         * @throws IOException if the lines do not describe one
         */
        Tokenizer read(List<String> lines) throws IOException;
    }

    /**
     * Makes {@link #load} read files of a tokenizer kind defined outside this module. Its
     * {@link #save} writes them with {@link #write}.
     *
     * @param kind the kind, as {@link #kind()} returns it; not a built-in one
     * @param reader reads the lines back
     */
    static void register(String kind, Reader reader) {
        if (KINDS.contains(kind) || kind.isEmpty() || kind.contains(" ")) {
            throw new IllegalArgumentException("cannot register tokenizer kind '" + kind + "'");
        }
        Pretokenizer.READERS.put(kind, java.util.Objects.requireNonNull(reader, "reader"));
    }

    /**
     * Writes a tokenizer file in the format {@link #load} reads: a header with the kind,
     * then the lines, which must not contain line breaks.
     *
     * @param path the file
     * @param kind the kind
     * @param lines the tokenizer's own data
     * @throws IOException if it cannot be written
     */
    static void write(Path path, String kind, List<String> lines) throws IOException {
        Pretokenizer.write(path, kind, lines);
    }

    /**
     * Cuts a text into the chunks the built-in tokenizers never merge across: words with
     * their leading space, numbers, punctuation runs and spaces, as in GPT-2.
     *
     * @param text the text
     * @return the chunks, in order; they join to the text
     */
    static List<String> pretokenize(String text) {
        return Pretokenizer.chunks(text);
    }

    /**
     * How many characters of a text each token covers on average: 1 for {@code char},
     * about 3 to 4 for a BPE vocabulary of a few thousand on English.
     *
     * @param text a sample text
     * @return characters per token
     */
    default double charactersPerToken(String text) {
        int tokens = encode(text).length;
        return tokens == 0 ? 1 : (double) text.length() / tokens;
    }
}
