package io.github.zzuegg.brainiac.nn.onnx;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/**
 * A reader of the protocol buffers wire format over a memory segment: varints, fixed-width
 * numbers and length-delimited fields, enough to walk an ONNX file without its schema
 * compiler. Length-delimited fields come back as slices of the same segment, so tensors in
 * a mapped file are never copied until read.
 */
final class Protobuf {

    static final int VARINT = 0;
    static final int FIXED64 = 1;
    static final int BYTES = 2;
    static final int FIXED32 = 5;

    private static final ValueLayout.OfInt INT = ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
    private static final ValueLayout.OfLong LONG = ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private final MemorySegment m;
    private long at;
    private int field;
    private int wire;

    Protobuf(MemorySegment m) {
        this.m = m;
    }

    /** Moves to the next field; false at the end of the message. */
    boolean next() {
        if (at >= m.byteSize()) {
            return false;
        }
        long key = varint();
        field = (int) (key >>> 3);
        wire = (int) (key & 7);
        return true;
    }

    int field() {
        return field;
    }

    int wire() {
        return wire;
    }

    long varint() {
        long v = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            byte b = m.get(ValueLayout.JAVA_BYTE, at++);
            v |= (long) (b & 0x7F) << shift;
            if (b >= 0) {
                return v;
            }
        }
        throw new IllegalArgumentException("malformed varint");
    }

    int fixed32() {
        int v = m.get(INT, at);
        at += 4;
        return v;
    }

    long fixed64() {
        long v = m.get(LONG, at);
        at += 8;
        return v;
    }

    float float32() {
        return Float.intBitsToFloat(fixed32());
    }

    MemorySegment bytes() {
        long n = varint();
        MemorySegment s = m.asSlice(at, n);
        at += n;
        return s;
    }

    String string() {
        return new String(bytes().toArray(ValueLayout.JAVA_BYTE), StandardCharsets.UTF_8);
    }

    Protobuf message() {
        return new Protobuf(bytes());
    }

    /** Skips the current field's value. */
    void skip() {
        switch (wire) {
            case VARINT -> varint();
            case FIXED64 -> at += 8;
            case BYTES -> {
                long n = varint(); // read before adding: at += varint() would use the old position
                at += n;
            }
            case FIXED32 -> at += 4;
            default -> throw new IllegalArgumentException("unsupported wire type " + wire);
        }
    }

    /** Reads a repeated int64 field's values, packed or one at a time, into {@code out}. */
    void longs(java.util.List<Long> out) {
        if (wire == BYTES) {
            Protobuf p = message();
            while (p.at < p.m.byteSize()) {
                out.add(p.varint());
            }
        } else {
            out.add(varint());
        }
    }

    /** Reads a repeated float field's values, packed or one at a time. */
    void floats(java.util.List<Float> out) {
        if (wire == BYTES) {
            MemorySegment s = bytes();
            for (long i = 0; i < s.byteSize(); i += 4) {
                out.add(Float.intBitsToFloat(s.get(INT, i)));
            }
        } else {
            out.add(float32());
        }
    }

    /** Reads a repeated double field's values, packed or one at a time. */
    void doubles(java.util.List<Double> out) {
        if (wire == BYTES) {
            MemorySegment s = bytes();
            for (long i = 0; i < s.byteSize(); i += 8) {
                out.add(Double.longBitsToDouble(s.get(LONG, i)));
            }
        } else {
            out.add(Double.longBitsToDouble(fixed64()));
        }
    }
}
