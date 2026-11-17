package io.github.zzuegg.brainiac.io;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * The parts of a SentencePiece {@code .model} file (a {@code ModelProto} protocol buffer)
 * that tokenizing needs, read without a protobuf library.
 */
record SentencePieceModel(List<String> pieces, double[] scores, int[] types, boolean bpe, boolean byteFallback,
        int unkId, int bosId, int eosId, int padId, boolean addDummyPrefix, boolean removeExtraWhitespaces,
        boolean escapeWhitespaces, String normalizer) {

    static final int NORMAL = 1;
    static final int UNKNOWN = 2;
    static final int CONTROL = 3;
    static final int USER_DEFINED = 4;
    static final int UNUSED = 5;
    static final int BYTE = 6;

    /** A forward-only protobuf reader. */
    private static final class Proto {
        final ByteBuffer b;

        Proto(ByteBuffer b) {
            this.b = b.order(ByteOrder.LITTLE_ENDIAN);
        }

        long varint() {
            long v = 0;
            for (int shift = 0; shift < 64; shift += 7) {
                int x = b.get() & 0xFF;
                v |= (long) (x & 0x7F) << shift;
                if (x < 0x80) {
                    return v;
                }
            }
            throw new IllegalArgumentException("bad protobuf varint");
        }

        ByteBuffer bytes() {
            int n = Math.toIntExact(varint());
            ByteBuffer slice = b.slice(b.position(), n);
            b.position(b.position() + n);
            return slice;
        }

        void skip(int wire) {
            switch (wire) {
                case 0 -> varint();
                case 1 -> b.position(b.position() + 8);
                case 2 -> bytes();
                case 5 -> b.position(b.position() + 4);
                default -> throw new IllegalArgumentException("bad protobuf wire type " + wire);
            }
        }
    }

    private static String string(ByteBuffer b) {
        byte[] a = new byte[b.remaining()];
        b.get(a);
        return new String(a, StandardCharsets.UTF_8);
    }

    static SentencePieceModel parse(byte[] file) {
        Proto p = new Proto(ByteBuffer.wrap(file));
        List<String> pieces = new ArrayList<>();
        List<Double> scores = new ArrayList<>();
        List<Integer> types = new ArrayList<>();
        int modelType = 1;
        boolean byteFallback = false;
        int unk = 0;
        int bos = 1;
        int eos = 2;
        int pad = -1;
        boolean dummy = true;
        boolean extra = true;
        boolean escape = true;
        String normalizer = "nmt_nfkc";
        while (p.b.hasRemaining()) {
            long key = p.varint();
            int field = (int) (key >>> 3);
            int wire = (int) (key & 7);
            if (field == 1 && wire == 2) {
                Proto sp = new Proto(p.bytes());
                String piece = "";
                float score = 0;
                int type = NORMAL;
                while (sp.b.hasRemaining()) {
                    long k = sp.varint();
                    int f = (int) (k >>> 3);
                    int w = (int) (k & 7);
                    if (f == 1 && w == 2) {
                        piece = string(sp.bytes());
                    } else if (f == 2 && w == 5) {
                        score = sp.b.getFloat();
                    } else if (f == 3 && w == 0) {
                        type = (int) sp.varint();
                    } else {
                        sp.skip(w);
                    }
                }
                pieces.add(piece);
                scores.add((double) score);
                types.add(type);
            } else if (field == 2 && wire == 2) {
                Proto t = new Proto(p.bytes());
                while (t.b.hasRemaining()) {
                    long k = t.varint();
                    int f = (int) (k >>> 3);
                    int w = (int) (k & 7);
                    if (w != 0) {
                        t.skip(w);
                        continue;
                    }
                    long v = t.varint();
                    switch (f) {
                        case 3 -> modelType = (int) v;
                        case 35 -> byteFallback = v != 0;
                        case 40 -> unk = (int) v;
                        case 41 -> bos = (int) v;
                        case 42 -> eos = (int) v;
                        case 43 -> pad = (int) v;
                        default -> {
                        }
                    }
                }
            } else if (field == 3 && wire == 2) {
                Proto n = new Proto(p.bytes());
                while (n.b.hasRemaining()) {
                    long k = n.varint();
                    int f = (int) (k >>> 3);
                    int w = (int) (k & 7);
                    if (f == 1 && w == 2) {
                        normalizer = string(n.bytes());
                    } else if (w == 0 && (f == 3 || f == 4 || f == 5)) {
                        boolean v = n.varint() != 0;
                        switch (f) {
                            case 3 -> dummy = v;
                            case 4 -> extra = v;
                            default -> escape = v;
                        }
                    } else {
                        n.skip(w);
                    }
                }
            } else {
                p.skip(wire);
            }
        }
        if (modelType != 1 && modelType != 2) {
            throw new IllegalArgumentException("SentencePiece model type " + modelType
                    + " is not supported; unigram (1) and BPE (2) are");
        }
        double[] s = scores.stream().mapToDouble(Double::doubleValue).toArray();
        int[] ty = types.stream().mapToInt(Integer::intValue).toArray();
        return new SentencePieceModel(List.copyOf(pieces), s, ty, modelType == 2, byteFallback, unk, bos, eos, pad,
                dummy, extra, escape, normalizer);
    }
}
