package io.github.zzuegg.brainiac.io;

/** Decompresses raw Snappy blocks (as Parquet stores them). */
final class Snappy {

    private Snappy() {
    }

    static byte[] decompress(byte[] in, int expected) {
        int p = 0;
        long length = 0;
        int shift = 0;
        int b;
        do {
            b = in[p++] & 0xFF;
            length |= (long) (b & 0x7F) << shift;
            shift += 7;
        } while ((b & 0x80) != 0);
        if (length != expected) {
            throw new IllegalArgumentException("snappy block holds " + length + " bytes, expected " + expected);
        }
        byte[] out = new byte[(int) length];
        int o = 0;
        while (p < in.length) {
            int tag = in[p++] & 0xFF;
            int kind = tag & 3;
            if (kind == 0) {
                int n = tag >>> 2;
                if (n >= 60) {
                    int bytes = n - 59;
                    n = 0;
                    for (int i = 0; i < bytes; i++) {
                        n |= (in[p++] & 0xFF) << 8 * i;
                    }
                }
                n++;
                System.arraycopy(in, p, out, o, n);
                p += n;
                o += n;
                continue;
            }
            int n;
            int offset;
            if (kind == 1) {
                n = (tag >>> 2 & 7) + 4;
                offset = (tag >>> 5) << 8 | in[p++] & 0xFF;
            } else if (kind == 2) {
                n = (tag >>> 2) + 1;
                offset = in[p] & 0xFF | (in[p + 1] & 0xFF) << 8;
                p += 2;
            } else {
                n = (tag >>> 2) + 1;
                offset = in[p] & 0xFF | (in[p + 1] & 0xFF) << 8 | (in[p + 2] & 0xFF) << 16 | (in[p + 3] & 0xFF) << 24;
                p += 4;
            }
            if (offset <= 0 || offset > o) {
                throw new IllegalArgumentException("broken snappy block");
            }
            for (int i = 0; i < n; i++) {
                out[o] = out[o - offset];
                o++;
            }
        }
        if (o != out.length) {
            throw new IllegalArgumentException("snappy block ends early");
        }
        return out;
    }
}
