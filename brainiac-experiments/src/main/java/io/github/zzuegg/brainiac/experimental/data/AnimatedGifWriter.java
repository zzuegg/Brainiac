package io.github.zzuegg.brainiac.experimental.data;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A minimal animated GIF encoder with no dependencies: indexed frames over one global
 * palette of up to 256 colours, LZW-compressed, with a per-frame delay and the NETSCAPE2.0
 * looping extension so the animation repeats. Enough to turn training snapshots into a
 * movie any browser can play.
 *
 * <p>Format: CompuServe, "Graphics Interchange Format, Version 89a" (1990),
 * <a href="https://www.w3.org/Graphics/GIF/spec-gif89a.txt">spec-gif89a.txt</a>; the
 * variable-width LZW code follows Welch 1984, "A Technique for High-Performance Data
 * Compression", <a href="https://doi.org/10.1109/MC.1984.1659158">doi:10.1109/MC.1984.1659158</a>.
 */
public final class AnimatedGifWriter {

    private final int width;
    private final int height;
    private final int[] palette;
    private final int colourBits;
    private final List<byte[]> frames = new ArrayList<>();
    private final List<Integer> delays = new ArrayList<>();
    private int loops;

    private AnimatedGifWriter(int width, int height, int[] palette) {
        if (width < 1 || height < 1 || width > 0xFFFF || height > 0xFFFF) {
            throw new IllegalArgumentException("bad size " + width + "x" + height);
        }
        if (palette.length < 1 || palette.length > 256) {
            throw new IllegalArgumentException("a palette has 1 to 256 colours");
        }
        this.width = width;
        this.height = height;
        int bits = 1;
        while (1 << bits < palette.length) {
            bits++;
        }
        this.colourBits = bits;
        this.palette = java.util.Arrays.copyOf(palette, 1 << bits);
    }

    /**
     * Starts an animation.
     *
     * @param width the width in pixels
     * @param height the height in pixels
     * @param palette the colours as {@code 0xRRGGBB}, at most 256
     * @return the writer
     */
    public static AnimatedGifWriter of(int width, int height, int... palette) {
        return new AnimatedGifWriter(width, height, palette.clone());
    }

    /**
     * Sets how often the animation repeats.
     *
     * @param count 0 to loop forever (the default)
     * @return this
     */
    public AnimatedGifWriter loops(int count) {
        this.loops = count;
        return this;
    }

    /**
     * Adds a frame.
     *
     * @param pixels palette indices, row by row, {@code width * height} of them
     * @param delayCentiseconds how long it shows, in hundredths of a second
     * @return this
     */
    public AnimatedGifWriter frame(byte[] pixels, int delayCentiseconds) {
        if (pixels.length != width * height) {
            throw new IllegalArgumentException("a frame has " + width * height + " pixels, not " + pixels.length);
        }
        for (byte p : pixels) {
            if ((p & 0xFF) >= palette.length) {
                throw new IllegalArgumentException("colour index " + (p & 0xFF) + " outside the palette");
            }
        }
        frames.add(pixels.clone());
        delays.add(delayCentiseconds);
        return this;
    }

    /** @return the number of frames */
    public int frames() {
        return frames.size();
    }

    /**
     * Encodes the animation.
     *
     * @return the GIF file's bytes
     */
    public byte[] toBytes() {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try {
            write(out);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return out.toByteArray();
    }

    /**
     * Writes the animation to a file.
     *
     * @param file the file
     * @throws IOException if writing fails
     */
    public void write(Path file) throws IOException {
        Files.write(file, toBytes());
    }

    /**
     * Writes the animation to a stream.
     *
     * @param out the stream
     * @throws IOException if writing fails
     */
    public void write(OutputStream out) throws IOException {
        if (frames.isEmpty()) {
            throw new IllegalStateException("no frames");
        }
        out.write("GIF89a".getBytes(java.nio.charset.StandardCharsets.US_ASCII));
        short16(out, width);
        short16(out, height);
        out.write(0x80 | (colourBits - 1) << 4 | (colourBits - 1));   // global table, its size
        out.write(0);   // background colour
        out.write(0);   // aspect ratio
        for (int c : palette) {
            out.write(c >> 16 & 0xFF);
            out.write(c >> 8 & 0xFF);
            out.write(c & 0xFF);
        }
        if (frames.size() > 1) {
            out.write(new byte[] {0x21, (byte) 0xFF, 11});
            out.write("NETSCAPE2.0".getBytes(java.nio.charset.StandardCharsets.US_ASCII));
            out.write(new byte[] {3, 1});
            short16(out, loops);
            out.write(0);
        }
        int minCode = Math.max(2, colourBits);
        for (int f = 0; f < frames.size(); f++) {
            out.write(new byte[] {0x21, (byte) 0xF9, 4, 0});   // graphic control: no transparency
            short16(out, delays.get(f));
            out.write(new byte[] {0, 0});
            out.write(0x2C);   // image descriptor
            short16(out, 0);
            short16(out, 0);
            short16(out, width);
            short16(out, height);
            out.write(0);
            out.write(minCode);
            byte[] data = lzw(frames.get(f), minCode);
            for (int i = 0; i < data.length; i += 255) {
                int n = Math.min(255, data.length - i);
                out.write(n);
                out.write(data, i, n);
            }
            out.write(0);
        }
        out.write(0x3B);
    }

    private static void short16(OutputStream out, int v) throws IOException {
        out.write(v & 0xFF);
        out.write(v >> 8 & 0xFF);
    }

    /** Variable-width LZW as GIF uses it: codes packed least significant bit first. */
    static byte[] lzw(byte[] pixels, int minCode) {
        int clear = 1 << minCode;
        int end = clear + 1;
        Bits bits = new Bits();
        Map<Integer, Integer> table = new HashMap<>();
        int size = minCode + 1;
        int next = end + 1;
        bits.put(clear, size);
        int prefix = pixels[0] & 0xFF;
        for (int i = 1; i < pixels.length; i++) {
            int k = pixels[i] & 0xFF;
            int key = prefix << 8 | k;
            Integer code = table.get(key);
            if (code != null) {
                prefix = code;
                continue;
            }
            bits.put(prefix, size);
            if (next == 4096) {
                bits.put(clear, size);
                table.clear();
                size = minCode + 1;
                next = end + 1;
            } else {
                if (next >= 1 << size) {
                    size++;
                }
                table.put(key, next++);
            }
            prefix = k;
        }
        bits.put(prefix, size);
        bits.put(end, size);
        return bits.toBytes();
    }

    private static final class Bits {
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();
        private int acc;
        private int count;

        void put(int code, int size) {
            acc |= code << count;
            count += size;
            while (count >= 8) {
                out.write(acc & 0xFF);
                acc >>>= 8;
                count -= 8;
            }
        }

        byte[] toBytes() {
            if (count > 0) {
                out.write(acc & 0xFF);
                acc = 0;
                count = 0;
            }
            return out.toByteArray();
        }
    }
}
