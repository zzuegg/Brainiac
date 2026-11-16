package io.github.zzuegg.brainiac.data;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Whole words as tokens: the most frequent words of the training text (with their leading
 * space), numbers, punctuation runs and spaces. A word outside the vocabulary is written
 * byte by byte, so nothing is unknown, but rare words cost many tokens.
 */
public final class WordTokenizer implements Tokenizer {

    private final Pieces pieces;

    private WordTokenizer(Pieces pieces) {
        this.pieces = pieces;
    }

    /**
     * Picks the most frequent words.
     *
     * @param text the training text
     * @param vocabularySize the number of tokens wanted, at least 256 (the byte fallback)
     * @return the tokenizer
     */
    public static WordTokenizer train(String text, int vocabularySize) {
        if (vocabularySize < 256) {
            throw new IllegalArgumentException("the vocabulary needs at least the 256 bytes, got " + vocabularySize);
        }
        List<Map.Entry<String, Integer>> words = new ArrayList<>(Pretokenizer.counts(text).entrySet());
        words.sort(Map.Entry.<String, Integer>comparingByValue().reversed());
        List<byte[]> longer = new ArrayList<>();
        for (Map.Entry<String, Integer> e : words) {
            if (256 + longer.size() >= vocabularySize || e.getValue() < 2) {
                break;
            }
            byte[] b = Pretokenizer.utf8(e.getKey());
            if (b.length > 1) {
                longer.add(b);
            }
        }
        return new WordTokenizer(new Pieces(longer));
    }

    @Override
    public int[] encode(String text) {
        List<String> chunks = Pretokenizer.chunks(text);
        int[] out = new int[text.length() * 4];
        int n = 0;
        for (String chunk : chunks) {
            byte[] b = Pretokenizer.utf8(chunk);
            int id = pieces.find(b, 0, b.length);
            if (id >= 0) {
                out[n++] = id;
            } else {
                for (byte x : b) {
                    out[n++] = x & 0xff;
                }
            }
        }
        return java.util.Arrays.copyOf(out, n);
    }

    @Override
    public String decode(int... ids) {
        return pieces.decode(ids);
    }

    @Override
    public int size() {
        return pieces.bytes.length;
    }

    @Override
    public String kind() {
        return "word";
    }

    @Override
    public void save(Path path) throws IOException {
        Pretokenizer.write(path, kind(), Pretokenizer.hexLines(pieces.longer()));
    }

    static WordTokenizer read(List<String> lines) {
        return new WordTokenizer(Pieces.read(lines));
    }

    @Override
    public String toString() {
        return "WordTokenizer(" + size() + " tokens)";
    }
}
