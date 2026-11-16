package io.github.zzuegg.brainiac.data;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * WordPiece, as in BERT. Training merges pieces like BPE, but picks the pair with the
 * highest {@code count(ab) / (count(a) count(b))}: pairs whose parts seldom occur apart,
 * such as {@code "q" + "u"}, go first. Encoding takes the longest known piece at each
 * position, left to right. Pieces are bytes, and a word's first piece holds its leading
 * space, so no {@code ##} continuation marks are needed.
 */
public final class WordPieceTokenizer implements Tokenizer {

    private final Pieces pieces;
    private final Map<String, int[]> cache = new ConcurrentHashMap<>();

    private WordPieceTokenizer(Pieces pieces) {
        this.pieces = pieces;
    }

    /**
     * Learns pieces from a text.
     *
     * @param text the training text
     * @param vocabularySize the number of tokens wanted, at least 256
     * @return the tokenizer
     */
    public static WordPieceTokenizer train(String text, int vocabularySize) {
        List<byte[]> learned = BytePairTokenizer.pieces(BytePairTokenizer.learn(text, vocabularySize, true));
        Set<String> seen = new HashSet<>();
        List<byte[]> unique = new ArrayList<>(learned.size());
        for (byte[] p : learned) {
            if (seen.add(Pretokenizer.key(p))) {
                unique.add(p);
            }
        }
        return new WordPieceTokenizer(new Pieces(unique));
    }

    @Override
    public int[] encode(String text) {
        List<int[]> parts = new ArrayList<>();
        int total = 0;
        for (String chunk : Pretokenizer.chunks(text)) {
            int[] ids = cache.computeIfAbsent(chunk, this::encodeChunk);
            parts.add(ids);
            total += ids.length;
        }
        int[] out = new int[total];
        int at = 0;
        for (int[] p : parts) {
            System.arraycopy(p, 0, out, at, p.length);
            at += p.length;
        }
        return out;
    }

    private int[] encodeChunk(String chunk) {
        byte[] b = Pretokenizer.utf8(chunk);
        int[] out = new int[b.length];
        int n = 0;
        int i = 0;
        while (i < b.length) {
            for (int end = Math.min(b.length, i + pieces.longest); end > i; end--) {
                int id = pieces.find(b, i, end);
                if (id >= 0) {
                    out[n++] = id;
                    i = end;
                    break;
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
        return "wordpiece";
    }

    @Override
    public void save(Path path) throws IOException {
        Pretokenizer.write(path, kind(), Pretokenizer.hexLines(pieces.longer()));
    }

    static WordPieceTokenizer read(List<String> lines) {
        return new WordPieceTokenizer(Pieces.read(lines));
    }

    @Override
    public String toString() {
        return "WordPieceTokenizer(" + size() + " tokens)";
    }
}
