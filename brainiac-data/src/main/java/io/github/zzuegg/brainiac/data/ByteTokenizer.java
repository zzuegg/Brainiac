package io.github.zzuegg.brainiac.data;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;

/** One token per UTF-8 byte: 256 tokens, nothing to learn, and any text works. */
public final class ByteTokenizer implements Tokenizer {

    /** The only instance. */
    public static final ByteTokenizer INSTANCE = new ByteTokenizer();

    private ByteTokenizer() {
    }

    @Override
    public int[] encode(String text) {
        byte[] b = text.getBytes(StandardCharsets.UTF_8);
        int[] ids = new int[b.length];
        for (int i = 0; i < b.length; i++) {
            ids[i] = b[i] & 0xff;
        }
        return ids;
    }

    @Override
    public String decode(int... ids) {
        byte[] b = new byte[ids.length];
        for (int i = 0; i < ids.length; i++) {
            if (ids[i] < 0 || ids[i] > 255) {
                throw new IllegalArgumentException("token " + ids[i] + " is outside [0, 256)");
            }
            b[i] = (byte) ids[i];
        }
        return new String(b, StandardCharsets.UTF_8);
    }

    @Override
    public int size() {
        return 256;
    }

    @Override
    public String kind() {
        return "byte";
    }

    @Override
    public void save(Path path) throws IOException {
        Pretokenizer.write(path, kind(), List.of());
    }

    @Override
    public String toString() {
        return "ByteTokenizer";
    }
}
