package io.github.zzuegg.brainiac.data;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A token corpus too large for memory: shards of token ids on disk, memory-mapped and read
 * as training windows. Each shard is a flat little-endian array of 16-bit (vocabularies up
 * to 65,536) or 32-bit token ids, the format nanoGPT and llm.c use; {@code shards.json}
 * lists the shards and their token width.
 *
 * <pre>{@code
 * // once: tokenize text into shards of 100M tokens
 * try (TokenShards.Writer w = TokenShards.writer(Path.of("data/web"), 100_000_000, tok.size())) {
 *     for (String doc : documents) w.add(tok.encode(doc), endOfText);
 * }
 * // training: windows of 1025 tokens (1024 inputs, each with its next token as target)
 * try (TokenShards data = TokenShards.open(Path.of("data/web"))) {
 *     int[][] windows = data.next(16, 1024);       // in order; data.position() resumes
 *     int[][] sample = data.random(16, 1024, rng); // or anywhere
 * }
 * }</pre>
 */
public final class TokenShards implements AutoCloseable {

    private final Arena arena;
    private final List<MemorySegment> shards;
    private final long[] starts;   // first token of each shard
    private final int bytes;
    private final long total;
    private long position;

    private TokenShards(Arena arena, List<MemorySegment> shards, int bytes) {
        this.arena = arena;
        this.shards = shards;
        this.bytes = bytes;
        this.starts = new long[shards.size() + 1];
        for (int i = 0; i < shards.size(); i++) {
            starts[i + 1] = starts[i] + shards.get(i).byteSize() / bytes;
        }
        this.total = starts[shards.size()];
    }

    /**
     * Opens a directory that {@link #writer} wrote.
     *
     * @param dir the directory with {@code shards.json}
     * @return the corpus
     */
    public static TokenShards open(Path dir) {
        try {
            String index = Files.readString(dir.resolve("shards.json"), StandardCharsets.UTF_8);
            Matcher bits = Pattern.compile("\"bits\"\\s*:\\s*(\\d+)").matcher(index);
            if (!bits.find()) {
                throw new IllegalArgumentException("shards.json has no \"bits\"");
            }
            List<Path> files = new ArrayList<>();
            Matcher names = Pattern.compile("\"([^\"]+\\.bin)\"").matcher(index);
            while (names.find()) {
                files.add(dir.resolve(names.group(1)));
            }
            return open(files, Integer.parseInt(bits.group(1)));
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Opens raw token files, such as nanoGPT's {@code train.bin}.
     *
     * @param files the shards, in order
     * @param bits 16 or 32
     * @return the corpus
     */
    public static TokenShards open(List<Path> files, int bits) {
        if (bits != 16 && bits != 32) {
            throw new IllegalArgumentException("tokens are 16 or 32 bits, not " + bits);
        }
        Arena arena = Arena.ofShared();
        List<MemorySegment> maps = new ArrayList<>();
        try {
            for (Path f : files) {
                try (FileChannel ch = FileChannel.open(f, StandardOpenOption.READ)) {
                    long size = ch.size();
                    if (size % (bits / 8) != 0) {
                        throw new IllegalArgumentException(f + " is not a whole number of " + bits + "-bit tokens");
                    }
                    maps.add(ch.map(FileChannel.MapMode.READ_ONLY, 0, size, arena));
                }
            }
        } catch (IOException | RuntimeException e) {
            arena.close();
            if (e instanceof IOException io) {
                throw new UncheckedIOException(io);
            }
            throw (RuntimeException) e;
        }
        return new TokenShards(arena, List.copyOf(maps), bits / 8);
    }

    /** @return the number of tokens in all shards */
    public long size() {
        return total;
    }

    /**
     * Reads one token.
     *
     * @param index its position in the whole corpus
     * @return the token id
     */
    public int token(long index) {
        if (index < 0 || index >= total) {
            throw new IndexOutOfBoundsException(index + " of " + total);
        }
        int s = shardOf(index);
        long off = (index - starts[s]) * bytes;
        MemorySegment m = shards.get(s);
        return bytes == 2 ? Short.toUnsignedInt(m.get(ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN),
                off)) : m.get(ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN), off);
    }

    private int shardOf(long index) {
        int lo = 0;
        int hi = shards.size() - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) >>> 1;
            if (starts[mid] <= index) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        return lo;
    }

    /**
     * Copies a run of tokens, across shard boundaries if needed.
     *
     * @param start the first token's position
     * @param out where to put them
     */
    public void read(long start, int[] out) {
        for (int i = 0; i < out.length; i++) {
            out[i] = token(start + i);
        }
    }

    /**
     * The next windows in order: window {@code i} starts where window {@code i - 1}'s inputs
     * end, so its first input is the last target of the one before. At the end of the
     * corpus it starts again from the beginning.
     *
     * @param batch the number of windows
     * @param context the inputs per window; each window holds {@code context + 1} tokens
     * @return {@code [batch][context + 1]} tokens: inputs are {@code [0, context)}, targets
     *     {@code [1, context]}
     */
    public int[][] next(int batch, int context) {
        if (context + 1 > total) {
            throw new IllegalArgumentException("windows of " + (context + 1) + " tokens, the corpus has " + total);
        }
        int[][] out = new int[batch][context + 1];
        for (int b = 0; b < batch; b++) {
            if (position + context + 1 > total) {
                position = 0;
            }
            read(position, out[b]);
            position += context;
        }
        return out;
    }

    /**
     * Windows from random places.
     *
     * @param batch the number of windows
     * @param context the inputs per window
     * @param random where the starts come from
     * @return {@code [batch][context + 1]} tokens
     */
    public int[][] random(int batch, int context, RandomGenerator random) {
        int[][] out = new int[batch][context + 1];
        for (int b = 0; b < batch; b++) {
            read(random.nextLong(total - context), out[b]);
        }
        return out;
    }

    /** @return where {@link #next} reads next, to save with a checkpoint */
    public long position() {
        return position;
    }

    /**
     * Moves where {@link #next} reads, to resume from a checkpoint.
     *
     * @param position the token position
     */
    public void seek(long position) {
        if (position < 0 || position > total) {
            throw new IndexOutOfBoundsException(position + " of " + total);
        }
        this.position = position;
    }

    @Override
    public void close() {
        arena.close();
    }

    // ---------------------------------------------------------------- writing

    /**
     * Starts writing shards.
     *
     * @param dir the directory, created if needed
     * @param tokensPerShard how many tokens each shard holds before the next one starts
     * @param vocabularySize the vocabulary size: up to 65,536 tokens take 16 bits, more 32
     * @return the writer; close it to finish
     */
    public static Writer writer(Path dir, long tokensPerShard, int vocabularySize) {
        return new Writer(dir, tokensPerShard, vocabularySize <= 65536 ? 16 : 32);
    }

    /** Appends tokens to shards. */
    public static final class Writer implements AutoCloseable {
        private final Path dir;
        private final long perShard;
        private final int bits;
        private final List<String> names = new ArrayList<>();
        private final List<Long> counts = new ArrayList<>();
        private java.io.OutputStream out;
        private long inShard;
        private long total;
        private final byte[] buffer = new byte[1 << 16];
        private int buffered;

        Writer(Path dir, long perShard, int bits) {
            if (perShard < 1) {
                throw new IllegalArgumentException("tokensPerShard < 1");
            }
            this.dir = dir;
            this.perShard = perShard;
            this.bits = bits;
            try {
                Files.createDirectories(dir);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        /**
         * Appends tokens.
         *
         * @param tokens the token ids
         * @return this writer
         */
        public Writer add(int... tokens) {
            try {
                for (int t : tokens) {
                    if (t < 0 || bits == 16 && t > 0xFFFF) {
                        throw new IllegalArgumentException("token " + t + " does not fit in " + bits + " bits");
                    }
                    if (out == null || inShard == perShard) {
                        startShard();
                    }
                    int need = bits / 8;
                    if (buffered + need > buffer.length) {
                        out.write(buffer, 0, buffered);
                        buffered = 0;
                    }
                    for (int k = 0; k < need; k++) {
                        buffer[buffered++] = (byte) (t >>> (8 * k));
                    }
                    inShard++;
                    total++;
                }
                return this;
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }

        /**
         * Appends a document followed by a separator such as the end-of-text token.
         *
         * @param tokens the document's tokens
         * @param separator the token after it
         * @return this writer
         */
        public Writer add(int[] tokens, int separator) {
            add(tokens);
            return add(separator);
        }

        /** @return the tokens written so far */
        public long size() {
            return total;
        }

        private void startShard() throws IOException {
            finishShard();
            String name = String.format("shard-%05d.bin", names.size());
            names.add(name);
            counts.add(0L);
            out = Files.newOutputStream(dir.resolve(name));
            inShard = 0;
        }

        private void finishShard() throws IOException {
            if (out != null) {
                out.write(buffer, 0, buffered);
                buffered = 0;
                out.close();
                counts.set(counts.size() - 1, inShard);
                out = null;
            }
        }

        /** Finishes the last shard and writes {@code shards.json}. */
        @Override
        public void close() {
            try {
                finishShard();
                StringBuilder json = new StringBuilder("{\n  \"bits\": ").append(bits).append(",\n  \"tokens\": ")
                        .append(total).append(",\n  \"shards\": [");
                for (int i = 0; i < names.size(); i++) {
                    json.append(i == 0 ? "\n    " : ",\n    ").append("{\"file\": \"").append(names.get(i))
                            .append("\", \"tokens\": ").append(counts.get(i)).append('}');
                }
                json.append("\n  ]\n}\n");
                Files.writeString(dir.resolve("shards.json"), json, StandardCharsets.UTF_8);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
    }
}
