package io.github.zzuegg.brainiac.experimental.data;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A minimal animated GIF89a encoder with no dependencies: one global palette of up to 256
 * colours, LZW-compressed frames, a per-frame delay and an endless loop. Meant for watching
 * models work: diffusion denoising, cellular automata growing, text crystallizing, a
 * Gaussian-splat fit converging.
 *
 * <pre>{@code
 * GifWriter gif = GifWriter.grayscale(8, 8).scale(8).delay(8);
 * for (float[] frame : frames) {
 *     gif.frame(frame, -1, 1);           // maps -1 to black and +1 to white
 * }
 * gif.write(Path.of("denoise.gif"));
 * }</pre>
 */
public final class GifWriter {
    private final int width;
    private final int height;
    private final int[] palette;
    private final List<byte[]> frames = new ArrayList<>();
    private int scale = 1;
    private int delay = 10;
    private boolean loop = true;

    private GifWriter(int width, int height, int[] palette) {
        if (width < 1 || height < 1) {
            throw new IllegalArgumentException("size must be positive, got " + width + "x" + height);
        }
        if (palette.length < 2 || palette.length > 256) {
            throw new IllegalArgumentException("a palette has 2..256 colours, got " + palette.length);
        }
        this.width = width;
        this.height = height;
        this.palette = palette.clone();
    }

    /**
     * Creates a writer with a 256-level grey palette.
     *
     * @param width frame width in pixels, before scaling
     * @param height frame height in pixels, before scaling
     * @return the writer
     */
    public static GifWriter grayscale(int width, int height) {
        int[] p = new int[256];
        for (int i = 0; i < 256; i++) {
            p[i] = i << 16 | i << 8 | i;
        }
        return new GifWriter(width, height, p);
    }

    /**
     * Creates a writer with a given palette.
     *
     * @param width frame width in pixels, before scaling
     * @param height frame height in pixels, before scaling
     * @param rgb the colours as {@code 0xRRGGBB}, 2..256 of them
     * @return the writer
     */
    public static GifWriter withPalette(int width, int height, int... rgb) {
        return new GifWriter(width, height, rgb);
    }

    /**
     * Sets an integer upscaling factor (nearest neighbour), since model outputs are tiny.
     *
     * @param factor the factor, at least 1
     * @return this writer
     */
    public GifWriter scale(int factor) {
        if (factor < 1) {
            throw new IllegalArgumentException("scale must be at least 1");
        }
        this.scale = factor;
        return this;
    }

    /**
     * Sets the delay between frames.
     *
     * @param hundredths the delay in hundredths of a second
     * @return this writer
     */
    public GifWriter delay(int hundredths) {
        this.delay = hundredths;
        return this;
    }

    /**
     * Sets whether the animation repeats forever (the default) or plays once.
     *
     * @param loop true to repeat
     * @return this writer
     */
    public GifWriter loop(boolean loop) {
        this.loop = loop;
        return this;
    }

    /**
     * Adds a frame of palette indices.
     *
     * @param indices {@code width * height} indices into the palette, row-major
     * @return this writer
     */
    public GifWriter frame(byte[] indices) {
        if (indices.length != width * height) {
            throw new IllegalArgumentException("a frame needs " + width * height + " pixels, got " + indices.length);
        }
        for (byte b : indices) {
            if ((b & 0xff) >= palette.length) {
                throw new IllegalArgumentException("index " + (b & 0xff) + " is outside the palette");
            }
        }
        frames.add(indices.clone());
        return this;
    }

    /**
     * Adds a frame of values, mapped linearly onto the palette ({@code min} to the first
     * colour, {@code max} to the last) and clamped.
     *
     * @param values {@code width * height} values, row-major
     * @param min the value shown as the first colour
     * @param max the value shown as the last colour
     * @return this writer
     */
    public GifWriter frame(float[] values, float min, float max) {
        byte[] idx = new byte[values.length];
        int top = palette.length - 1;
        for (int i = 0; i < values.length; i++) {
            float v = (values[i] - min) / (max - min);
            idx[i] = (byte) Math.round(Math.max(0, Math.min(1, Float.isNaN(v) ? 0 : v)) * top);
        }
        return frame(idx);
    }

    /** @return the number of frames added so far */
    public int frames() {
        return frames.size();
    }

    /**
     * Encodes the animation.
     *
     * @return the GIF file's bytes
     */
    public byte[] toBytes() {
        if (frames.isEmpty()) {
            throw new IllegalStateException("add at least one frame");
        }
        int w = width * scale;
        int h = height * scale;
        int bits = 1;
        while (1 << bits < palette.length) {
            bits++;
        }
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.writeBytes("GIF89a".getBytes(java.nio.charset.StandardCharsets.US_ASCII));
        short16(out, w);
        short16(out, h);
        out.write(0x80 | (bits - 1) << 4 | (bits - 1));   // global table, colour resolution, size
        out.write(0);   // background colour index
        out.write(0);   // pixel aspect ratio
        for (int i = 0; i < 1 << bits; i++) {
            int c = i < palette.length ? palette[i] : 0;
            out.write(c >> 16 & 0xff);
            out.write(c >> 8 & 0xff);
            out.write(c & 0xff);
        }
        if (loop) {
            out.write(0x21);
            out.write(0xff);
            out.write(11);
            out.writeBytes("NETSCAPE2.0".getBytes(java.nio.charset.StandardCharsets.US_ASCII));
            out.write(3);
            out.write(1);
            short16(out, 0);   // repeat forever
            out.write(0);
        }
        for (byte[] f : frames) {
            out.write(0x21);   // graphic control extension
            out.write(0xf9);
            out.write(4);
            out.write(0);
            short16(out, delay);
            out.write(0);
            out.write(0);
            out.write(0x2c);   // image descriptor
            short16(out, 0);
            short16(out, 0);
            short16(out, w);
            short16(out, h);
            out.write(0);
            int minCode = Math.max(2, bits);
            out.write(minCode);
            subBlocks(out, lzw(upscale(f, w, h), minCode));
        }
        out.write(0x3b);
        return out.toByteArray();
    }

    /**
     * Encodes the animation and writes it to a file.
     *
     * @param path the file
     */
    public void write(Path path) {
        try {
            Files.write(path, toBytes());
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private byte[] upscale(byte[] f, int w, int h) {
        if (scale == 1) {
            return f;
        }
        byte[] big = new byte[w * h];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                big[y * w + x] = f[(y / scale) * width + x / scale];
            }
        }
        return big;
    }

    /** Variable-length-code LZW as GIF uses it: LSB-first codes, clear and end codes. */
    static byte[] lzw(byte[] pixels, int minCode) {
        int clear = 1 << minCode;
        int end = clear + 1;
        BitWriter bw = new BitWriter();
        Map<Integer, Integer> dict = new HashMap<>();
        int size = minCode + 1;
        int next = end + 1;
        bw.write(clear, size);
        int prefix = pixels[0] & 0xff;
        for (int i = 1; i < pixels.length; i++) {
            int k = pixels[i] & 0xff;
            int key = prefix << 8 | k;
            Integer code = dict.get(key);
            if (code != null) {
                prefix = code;
                continue;
            }
            bw.write(prefix, size);
            if (next < 4096) {
                dict.put(key, next++);
                if (next > 1 << size && size < 12) {
                    size++;
                }
            } else {
                bw.write(clear, size);
                dict.clear();
                size = minCode + 1;
                next = end + 1;
            }
            prefix = k;
        }
        bw.write(prefix, size);
        bw.write(end, size);
        return bw.toBytes();
    }

    private static void subBlocks(ByteArrayOutputStream out, byte[] data) {
        for (int i = 0; i < data.length; i += 255) {
            int n = Math.min(255, data.length - i);
            out.write(n);
            out.write(data, i, n);
        }
        out.write(0);
    }

    private static void short16(ByteArrayOutputStream out, int v) {
        out.write(v & 0xff);
        out.write(v >> 8 & 0xff);
    }

    private static final class BitWriter {
        private final ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        private int buffer;
        private int count;

        void write(int code, int bits) {
            buffer |= code << count;
            count += bits;
            while (count >= 8) {
                bytes.write(buffer & 0xff);
                buffer >>>= 8;
                count -= 8;
            }
        }

        byte[] toBytes() {
            if (count > 0) {
                bytes.write(buffer & 0xff);
                buffer = 0;
                count = 0;
            }
            return bytes.toByteArray();
        }
    }
}
