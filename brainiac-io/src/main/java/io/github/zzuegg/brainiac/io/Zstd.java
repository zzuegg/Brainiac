package io.github.zzuegg.brainiac.io;

import java.util.Arrays;

/**
 * Decompresses Zstandard frames (RFC 8878), as Parquet and other formats store them. Only
 * decoding, and no dictionaries.
 */
final class Zstd {

    private static final int MAGIC = 0xFD2FB528;

    private static final short[] LL_DEFAULT = {4, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            3, 2, 1, 1, 1, 1, 1, -1, -1, -1, -1};
    private static final short[] ML_DEFAULT = {1, 4, 3, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1};
    private static final short[] OF_DEFAULT = {1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            -1, -1, -1, -1, -1};

    private static final int[] LL_BASE = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 28,
            32, 40, 48, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
    private static final int[] LL_BITS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 6,
            7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    private static final int[] ML_BASE = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39, 41, 43, 47, 51, 59, 67, 83, 99, 131, 259, 515,
            1027, 2051, 4099, 8195, 16387, 32771, 65539};
    private static final int[] ML_BITS = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    private Zstd() {
    }

    /** An FSE decoding table. */
    private record Fse(int log, int[] symbol, int[] bits, int[] base) {

        static Fse of(short[] norm, int maxSymbol, int log) {
            int size = 1 << log;
            int[] symbol = new int[size];
            int[] bits = new int[size];
            int[] base = new int[size];
            int[] next = new int[maxSymbol + 1];
            int high = size - 1;
            for (int s = 0; s <= maxSymbol; s++) {
                if (norm[s] == -1) {
                    symbol[high--] = s;
                    next[s] = 1;
                } else {
                    next[s] = norm[s];
                }
            }
            int pos = 0;
            int step = (size >> 1) + (size >> 3) + 3;
            for (int s = 0; s <= maxSymbol; s++) {
                for (int i = 0; i < norm[s]; i++) {
                    symbol[pos] = s;
                    do {
                        pos = (pos + step) & (size - 1);
                    } while (pos > high);
                }
            }
            if (pos != 0) {
                throw new IllegalArgumentException("broken zstd FSE table");
            }
            for (int u = 0; u < size; u++) {
                int s = symbol[u];
                int state = next[s]++;
                int nb = log - (31 - Integer.numberOfLeadingZeros(state));
                bits[u] = nb;
                base[u] = (state << nb) - size;
            }
            return new Fse(log, symbol, bits, base);
        }

        static Fse rle(int s) {
            return new Fse(0, new int[] {s}, new int[] {0}, new int[] {0});
        }
    }

    /** Reads bits backwards from the end of a range, as FSE and Huffman streams are written. */
    private static final class Backward {
        private final byte[] in;
        private final int start;
        private final long total;
        private long consumed;

        Backward(byte[] in, int start, int end) {
            if (end <= start || in[end - 1] == 0) {
                throw new IllegalArgumentException("broken zstd bitstream");
            }
            this.in = in;
            this.start = start;
            total = (long) (end - start) * 8;
            consumed = 8 - (31 - Integer.numberOfLeadingZeros(in[end - 1] & 0xFF));
        }

        int read(int n) {
            if (n == 0) {
                return 0;
            }
            long pos = total - consumed - n;
            consumed += n;
            long v = 0;
            long byteAt = Math.floorDiv(pos, 8);
            for (int i = 0; i < 8; i++) {
                long b = byteAt + i;
                if (b >= 0 && b < total / 8) {
                    v |= (long) (in[start + (int) b] & 0xFF) << 8 * i;
                }
            }
            return (int) (v >>> Math.floorMod(pos, 8) & ((1L << n) - 1));
        }

        boolean overflowed() {
            return consumed > total;
        }

        boolean finished() {
            return consumed == total;
        }
    }

    /** State kept across the blocks of a frame. */
    private static final class Frame {
        byte[] out;
        int size;
        int[] huffSymbol;
        int[] huffBits;
        int huffLog;
        Fse ll;
        Fse of;
        Fse ml;
        final int[] reps = {1, 4, 8};

        void put(byte b) {
            if (size == out.length) {
                out = Arrays.copyOf(out, Math.max(16, out.length * 2));
            }
            out[size++] = b;
        }
    }

    static byte[] decompress(byte[] in, int expected) {
        Frame f = new Frame();
        f.out = new byte[Math.max(expected, 16)];
        int p = 0;
        while (p < in.length) {
            int magic = le(in, p, 4);
            if ((magic & 0xFFFFFFF0) == 0x184D2A50) {   // skippable frame
                p += 8 + le(in, p + 4, 4);
                continue;
            }
            if (magic != MAGIC) {
                throw new IllegalArgumentException("not a zstd frame");
            }
            p += 4;
            int descriptor = in[p++] & 0xFF;
            int fcsFlag = descriptor >>> 6;
            boolean single = (descriptor & 0x20) != 0;
            boolean checksum = (descriptor & 4) != 0;
            int dictFlag = descriptor & 3;
            if (!single) {
                p++;   // window descriptor
            }
            int dictBytes = new int[] {0, 1, 2, 4}[dictFlag];
            if (dictBytes > 0 && le(in, p, dictBytes) != 0) {
                throw new IllegalArgumentException("zstd dictionaries are not supported");
            }
            p += dictBytes;
            p += fcsFlag == 0 ? (single ? 1 : 0) : 1 << fcsFlag;
            f.reps[0] = 1;
            f.reps[1] = 4;
            f.reps[2] = 8;
            f.huffSymbol = null;
            f.ll = f.of = f.ml = null;
            boolean last = false;
            while (!last) {
                int header = le(in, p, 3);
                p += 3;
                last = (header & 1) != 0;
                int type = header >>> 1 & 3;
                int size = header >>> 3;
                switch (type) {
                    case 0 -> {
                        for (int i = 0; i < size; i++) {
                            f.put(in[p + i]);
                        }
                        p += size;
                    }
                    case 1 -> {
                        for (int i = 0; i < size; i++) {
                            f.put(in[p]);
                        }
                        p++;
                    }
                    case 2 -> {
                        block(in, p, p + size, f);
                        p += size;
                    }
                    default -> throw new IllegalArgumentException("broken zstd block");
                }
            }
            if (checksum) {
                p += 4;
            }
        }
        if (f.size != expected) {
            throw new IllegalArgumentException("zstd data holds " + f.size + " bytes, expected " + expected);
        }
        return f.out.length == f.size ? f.out : Arrays.copyOf(f.out, f.size);
    }

    private static void block(byte[] in, int p, int end, Frame f) {
        // literals
        int b0 = in[p] & 0xFF;
        int type = b0 & 3;
        int format = b0 >>> 2 & 3;
        byte[] literals;
        if (type == 0 || type == 1) {
            int size;
            if (format == 0 || format == 2) {
                size = b0 >>> 3;
                p += 1;
            } else if (format == 1) {
                size = (b0 >>> 4) + ((in[p + 1] & 0xFF) << 4);
                p += 2;
            } else {
                size = (b0 >>> 4) + ((in[p + 1] & 0xFF) << 4) + ((in[p + 2] & 0xFF) << 12);
                p += 3;
            }
            if (type == 0) {
                literals = Arrays.copyOfRange(in, p, p + size);
                p += size;
            } else {
                literals = new byte[size];
                Arrays.fill(literals, in[p]);
                p += 1;
            }
        } else {
            int regen;
            int compressed;
            boolean four = format != 0;
            int b1 = in[p + 1] & 0xFF;
            int b2 = in[p + 2] & 0xFF;
            if (format <= 1) {
                regen = b0 >>> 4 | (b1 & 0x3F) << 4;
                compressed = b1 >>> 6 | b2 << 2;
                p += 3;
            } else if (format == 2) {
                int b3 = in[p + 3] & 0xFF;
                regen = b0 >>> 4 | b1 << 4 | (b2 & 3) << 12;
                compressed = b2 >>> 2 | b3 << 6;
                p += 4;
            } else {
                int b3 = in[p + 3] & 0xFF;
                int b4 = in[p + 4] & 0xFF;
                regen = b0 >>> 4 | b1 << 4 | (b2 & 0x3F) << 12;
                compressed = b2 >>> 6 | b3 << 2 | b4 << 10;
                p += 5;
            }
            int streams = p;
            if (type == 2) {
                streams = huffmanTable(in, p, f);
            } else if (f.huffSymbol == null) {
                throw new IllegalArgumentException("broken zstd literals (no Huffman table to repeat)");
            }
            literals = new byte[regen];
            int streamsEnd = p + compressed;
            if (!four) {
                huffman(in, streams, streamsEnd, f, literals, 0, regen);
            } else {
                int s1 = le(in, streams, 2);
                int s2 = le(in, streams + 2, 2);
                int s3 = le(in, streams + 4, 2);
                int at = streams + 6;
                int per = (regen + 3) / 4;
                huffman(in, at, at + s1, f, literals, 0, per);
                huffman(in, at + s1, at + s1 + s2, f, literals, per, per);
                huffman(in, at + s1 + s2, at + s1 + s2 + s3, f, literals, 2 * per, per);
                huffman(in, at + s1 + s2 + s3, streamsEnd, f, literals, 3 * per, regen - 3 * per);
            }
            p = streamsEnd;
        }
        // sequences
        int count;
        int c0 = in[p++] & 0xFF;
        if (c0 < 128) {
            count = c0;
        } else if (c0 < 255) {
            count = (c0 - 128 << 8) + (in[p++] & 0xFF);
        } else {
            count = (in[p] & 0xFF) + ((in[p + 1] & 0xFF) << 8) + 0x7F00;
            p += 2;
        }
        int lit = 0;
        if (count > 0) {
            int modes = in[p++] & 0xFF;
            int[] at = {p};
            f.ll = table(in, at, modes >>> 6, f.ll, LL_DEFAULT, 6, 35, 9);
            f.of = table(in, at, modes >>> 4 & 3, f.of, OF_DEFAULT, 5, 31, 8);
            f.ml = table(in, at, modes >>> 2 & 3, f.ml, ML_DEFAULT, 6, 52, 9);
            Backward bits = new Backward(in, at[0], end);
            int ll = bits.read(f.ll.log);
            int of = bits.read(f.of.log);
            int ml = bits.read(f.ml.log);
            for (int s = 0; s < count; s++) {
                int ofCode = f.of.symbol[of];
                int mlCode = f.ml.symbol[ml];
                int llCode = f.ll.symbol[ll];
                long offsetValue = (1L << ofCode) + bits.read(ofCode);
                int matchLength = ML_BASE[mlCode] + bits.read(ML_BITS[mlCode]);
                int literalLength = LL_BASE[llCode] + bits.read(LL_BITS[llCode]);
                int offset;
                if (offsetValue > 3) {
                    offset = (int) (offsetValue - 3);
                    f.reps[2] = f.reps[1];
                    f.reps[1] = f.reps[0];
                    f.reps[0] = offset;
                } else {
                    int index = (int) offsetValue - 1 + (literalLength == 0 ? 1 : 0);
                    if (index == 0) {
                        offset = f.reps[0];
                    } else {
                        offset = index == 3 ? f.reps[0] - 1 : f.reps[index];
                        if (index != 1) {
                            f.reps[2] = f.reps[1];
                        }
                        f.reps[1] = f.reps[0];
                        f.reps[0] = offset;
                    }
                }
                if (s < count - 1) {
                    ll = f.ll.base[ll] + bits.read(f.ll.bits[ll]);
                    ml = f.ml.base[ml] + bits.read(f.ml.bits[ml]);
                    of = f.of.base[of] + bits.read(f.of.bits[of]);
                }
                if (lit + literalLength > literals.length) {
                    throw new IllegalArgumentException("broken zstd sequence (literals)");
                }
                for (int i = 0; i < literalLength; i++) {
                    f.put(literals[lit++]);
                }
                if (offset <= 0 || offset > f.size) {
                    throw new IllegalArgumentException("broken zstd sequence (offset " + offset + ")");
                }
                for (int i = 0; i < matchLength; i++) {
                    f.put(f.out[f.size - offset]);
                }
            }
        }
        while (lit < literals.length) {
            f.put(literals[lit++]);
        }
    }

    /** A sequence code table: predefined, RLE, described, or the previous one. */
    private static Fse table(byte[] in, int[] at, int mode, Fse previous, short[] defaults, int defaultLog,
            int maxSymbol, int maxLog) {
        return switch (mode) {
            case 0 -> Fse.of(defaults, defaults.length - 1, defaultLog);
            case 1 -> Fse.rle(in[at[0]++] & 0xFF);
            case 2 -> {
                short[] norm = new short[maxSymbol + 1];
                int[] info = counts(in, at[0], norm, maxSymbol);
                at[0] = info[0];
                if (info[1] > maxLog) {
                    throw new IllegalArgumentException("broken zstd table (log " + info[1] + ")");
                }
                yield Fse.of(norm, info[2], info[1]);
            }
            default -> {
                if (previous == null) {
                    throw new IllegalArgumentException("broken zstd sequences (no table to repeat)");
                }
                yield previous;
            }
        };
    }

    /**
     * Reads FSE normalized counts from a forward bitstream.
     *
     * @return the position after them, the table log and the largest symbol
     */
    private static int[] counts(byte[] in, int start, short[] norm, int maxSymbol) {
        long bitPos = (long) start * 8;
        int log = (int) bitsAt(in, bitPos, 4) + 5;
        bitPos += 4;
        int remaining = (1 << log) + 1;
        int threshold = 1 << log;
        int nbBits = log + 1;
        int symbol = 0;
        boolean previous0 = false;
        while (remaining > 1 && symbol <= maxSymbol) {
            if (previous0) {
                int repeat;
                do {
                    repeat = (int) bitsAt(in, bitPos, 2);
                    bitPos += 2;
                    for (int i = 0; i < repeat; i++) {
                        norm[symbol++] = 0;
                    }
                } while (repeat == 3);
                if (symbol > maxSymbol) {
                    break;
                }
            }
            int max = 2 * threshold - 1 - remaining;
            long peek = bitsAt(in, bitPos, nbBits);
            int count;
            if ((peek & (threshold - 1)) < max) {
                count = (int) (peek & (threshold - 1));
                bitPos += nbBits - 1;
            } else {
                count = (int) (peek & (2L * threshold - 1));
                if (count >= threshold) {
                    count -= max;
                }
                bitPos += nbBits;
            }
            count--;
            remaining -= Math.abs(count);
            norm[symbol++] = (short) count;
            previous0 = count == 0;
            while (remaining < threshold) {
                nbBits--;
                threshold >>= 1;
            }
        }
        if (remaining != 1) {
            throw new IllegalArgumentException("broken zstd FSE counts");
        }
        return new int[] {(int) ((bitPos + 7) / 8), log, symbol - 1};
    }

    private static long bitsAt(byte[] in, long bitPos, int n) {
        long v = 0;
        int byteAt = (int) (bitPos >>> 3);
        for (int i = 0; i < 8 && byteAt + i < in.length; i++) {
            v |= (long) (in[byteAt + i] & 0xFF) << 8 * i;
        }
        return v >>> (bitPos & 7) & ((1L << n) - 1);
    }

    /** Reads a Huffman tree description; returns where the streams start. */
    private static int huffmanTable(byte[] in, int p, Frame f) {
        int header = in[p++] & 0xFF;
        int[] weights = new int[256];
        int n;
        if (header < 128) {
            // weights compressed with FSE: two interleaved states
            short[] norm = new short[256];
            int[] info = counts(in, p, norm, 255);
            if (info[1] > 6) {
                throw new IllegalArgumentException("broken zstd Huffman weights");
            }
            Fse t = Fse.of(norm, info[2], info[1]);
            Backward bits = new Backward(in, info[0], p + header);
            int s1 = bits.read(t.log);
            int s2 = bits.read(t.log);
            n = 0;
            while (true) {
                weights[n++] = t.symbol[s1];
                s1 = t.base[s1] + bits.read(t.bits[s1]);
                if (bits.overflowed()) {
                    weights[n++] = t.symbol[s2];
                    break;
                }
                weights[n++] = t.symbol[s2];
                s2 = t.base[s2] + bits.read(t.bits[s2]);
                if (bits.overflowed()) {
                    weights[n++] = t.symbol[s1];
                    break;
                }
                if (n > 254) {
                    throw new IllegalArgumentException("broken zstd Huffman weights");
                }
            }
            p += header;
        } else {
            n = header - 127;
            for (int i = 0; i < n; i++) {
                int b = in[p + i / 2] & 0xFF;
                weights[i] = i % 2 == 0 ? b >>> 4 : b & 15;
            }
            p += (n + 1) / 2;
        }
        long sum = 0;
        for (int i = 0; i < n; i++) {
            if (weights[i] > 0) {
                sum += 1L << weights[i] - 1;
            }
        }
        int maxBits = 64 - Long.numberOfLeadingZeros(sum);
        long rest = (1L << maxBits) - sum;
        if (rest == 0 || Long.bitCount(rest) != 1) {
            throw new IllegalArgumentException("broken zstd Huffman weights");
        }
        weights[n++] = 64 - Long.numberOfLeadingZeros(rest);
        int size = 1 << maxBits;
        int[] symbol = new int[size];
        int[] bits = new int[size];
        int[] rankStart = new int[maxBits + 2];
        int[] rankCount = new int[maxBits + 2];
        for (int i = 0; i < n; i++) {
            rankCount[weights[i]]++;
        }
        int pos = 0;
        for (int w = 1; w <= maxBits; w++) {
            rankStart[w] = pos;
            pos += rankCount[w] << w - 1;
        }
        for (int s = 0; s < n; s++) {
            int w = weights[s];
            if (w == 0) {
                continue;
            }
            int len = 1 << w - 1;
            for (int i = 0; i < len; i++) {
                symbol[rankStart[w] + i] = s;
                bits[rankStart[w] + i] = maxBits + 1 - w;
            }
            rankStart[w] += len;
        }
        f.huffSymbol = symbol;
        f.huffBits = bits;
        f.huffLog = maxBits;
        return p;
    }

    private static void huffman(byte[] in, int start, int end, Frame f, byte[] out, int at, int count) {
        if (count == 0) {
            return;
        }
        Backward bits = new Backward(in, start, end);
        for (int i = 0; i < count; i++) {
            long before = bits.consumed;
            int peek = bits.read(f.huffLog);
            bits.consumed = before + f.huffBits[peek];
            out[at + i] = (byte) f.huffSymbol[peek];
        }
        if (!bits.finished()) {
            throw new IllegalArgumentException("broken zstd literals stream");
        }
    }

    private static int le(byte[] in, int p, int n) {
        int v = 0;
        for (int i = 0; i < n; i++) {
            v |= (in[p + i] & 0xFF) << 8 * i;
        }
        return v;
    }
}
