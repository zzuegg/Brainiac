package io.github.zzuegg.brainiac.data;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A vocabulary of byte strings: ids 0 to 255 are the single bytes, so every text can be
 * written with it, and longer pieces follow.
 */
final class Pieces {

    final byte[][] bytes;
    final Map<String, Integer> index;
    final int longest;

    Pieces(List<byte[]> longer) {
        bytes = new byte[256 + longer.size()][];
        index = new HashMap<>(bytes.length * 2);
        int max = 1;
        for (int b = 0; b < 256; b++) {
            bytes[b] = new byte[] {(byte) b};
            index.put(Pretokenizer.key(bytes[b]), b);
        }
        for (int i = 0; i < longer.size(); i++) {
            byte[] p = longer.get(i);
            if (p.length < 2) {
                throw new IllegalArgumentException("piece " + (256 + i) + " must be longer than one byte");
            }
            if (index.put(Pretokenizer.key(p), 256 + i) != null) {
                throw new IllegalArgumentException("piece " + (256 + i) + " appears twice");
            }
            bytes[256 + i] = p.clone();
            max = Math.max(max, p.length);
        }
        longest = max;
    }

    /** The bytes of each piece longer than one byte, in id order. */
    List<byte[]> longer() {
        List<byte[]> out = new ArrayList<>(bytes.length - 256);
        for (int i = 256; i < bytes.length; i++) {
            out.add(bytes[i]);
        }
        return out;
    }

    /** The id of the bytes {@code [from, to)} of a chunk, or -1. */
    int find(byte[] chunk, int from, int to) {
        Integer id = index.get(Pretokenizer.key(chunk, from, to));
        return id == null ? -1 : id;
    }

    String decode(int... ids) {
        ByteArrayOutputStream out = new ByteArrayOutputStream(ids.length * 4);
        for (int id : ids) {
            if (id < 0 || id >= bytes.length) {
                throw new IllegalArgumentException("token " + id + " is outside [0, " + bytes.length + ")");
            }
            out.writeBytes(bytes[id]);
        }
        return out.toString(StandardCharsets.UTF_8);
    }

    static Pieces read(List<String> hexLines) {
        List<byte[]> longer = new ArrayList<>(hexLines.size());
        for (String line : hexLines) {
            longer.add(Pretokenizer.unhex(line.trim().split(" ")[0]));
        }
        return new Pieces(longer);
    }
}
