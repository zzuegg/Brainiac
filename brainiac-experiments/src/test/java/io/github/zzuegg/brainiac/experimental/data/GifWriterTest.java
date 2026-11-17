package io.github.zzuegg.brainiac.experimental.data;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

class GifWriterTest {

    /** An independent GIF decoder, enough to read back what the writer produces. */
    static List<byte[]> decode(byte[] gif, int[] size) throws java.io.IOException {
        ByteArrayInputStream in = new ByteArrayInputStream(gif);
        byte[] header = in.readNBytes(6);
        assertEquals("GIF89a", new String(header, java.nio.charset.StandardCharsets.US_ASCII));
        size[0] = in.read() | in.read() << 8;
        size[1] = in.read() | in.read() << 8;
        int flags = in.read();
        in.read();
        in.read();
        if ((flags & 0x80) != 0) {
            in.skipNBytes(3L * (1 << ((flags & 7) + 1)));
        }
        List<byte[]> frames = new ArrayList<>();
        while (true) {
            int b = in.read();
            if (b == 0x3b) {
                return frames;
            }
            if (b == 0x21) {
                in.read();
                for (int n = in.read(); n > 0; n = in.read()) {
                    in.skipNBytes(n);
                }
                continue;
            }
            assertEquals(0x2c, b);
            in.skipNBytes(4);
            int w = in.read() | in.read() << 8;
            int h = in.read() | in.read() << 8;
            in.read();
            int minCode = in.read();
            ByteArrayOutputStream data = new ByteArrayOutputStream();
            for (int n = in.read(); n > 0; n = in.read()) {
                data.writeBytes(in.readNBytes(n));
            }
            frames.add(unLzw(data.toByteArray(), minCode, w * h));
        }
    }

    static byte[] unLzw(byte[] data, int minCode, int pixels) {
        int clear = 1 << minCode;
        int end = clear + 1;
        byte[] out = new byte[pixels];
        int pos = 0;
        List<byte[]> table = new ArrayList<>();
        int size = minCode + 1;
        int bitPos = 0;
        byte[] prev = null;
        while (true) {
            int code = 0;
            for (int i = 0; i < size; i++, bitPos++) {
                code |= ((data[bitPos >> 3] >> (bitPos & 7)) & 1) << i;
            }
            if (code == clear) {
                table.clear();
                for (int i = 0; i < clear + 2; i++) {
                    table.add(new byte[] {(byte) i});
                }
                size = minCode + 1;
                prev = null;
                continue;
            }
            if (code == end) {
                break;
            }
            byte[] entry;
            if (code < table.size()) {
                entry = table.get(code);
                if (prev != null) {
                    table.add(concat(prev, entry[0]));
                }
            } else {
                entry = concat(prev, prev[0]);
                table.add(entry);
            }
            if (table.size() == 1 << size && size < 12) {
                size++;
            }
            System.arraycopy(entry, 0, out, pos, entry.length);
            pos += entry.length;
            prev = entry;
        }
        assertEquals(pixels, pos);
        return out;
    }

    private static byte[] concat(byte[] a, byte b) {
        byte[] c = java.util.Arrays.copyOf(a, a.length + 1);
        c[a.length] = b;
        return c;
    }

    @Test
    void framesRoundTripThroughLzw() throws java.io.IOException {
        SplittableRandom r = new SplittableRandom(1);
        GifWriter gif = GifWriter.grayscale(40, 30);
        List<byte[]> written = new ArrayList<>();
        for (int f = 0; f < 3; f++) {
            byte[] px = new byte[40 * 30];
            for (int i = 0; i < px.length; i++) {
                // noise (forces dictionary resets) mixed with flat runs (long codes)
                px[i] = (byte) (f == 1 ? r.nextInt(256) : (i / 37) % 7);
            }
            written.add(px);
            gif.frame(px);
        }
        int[] size = new int[2];
        List<byte[]> read = decode(gif.toBytes(), size);
        assertEquals(40, size[0]);
        assertEquals(30, size[1]);
        assertEquals(3, read.size());
        for (int f = 0; f < 3; f++) {
            assertArrayEquals(written.get(f), read.get(f));
        }
    }

    @Test
    void scalesAndMapsValuesOntoThePalette() throws java.io.IOException {
        GifWriter gif = GifWriter.withPalette(2, 1, 0x000000, 0xffffff).scale(3);
        gif.frame(new float[] {-1, 1}, -1, 1);
        int[] size = new int[2];
        byte[] px = decode(gif.toBytes(), size).get(0);
        assertEquals(6, size[0]);
        assertEquals(3, size[1]);
        assertArrayEquals(new byte[] {0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1}, px);
    }
}
