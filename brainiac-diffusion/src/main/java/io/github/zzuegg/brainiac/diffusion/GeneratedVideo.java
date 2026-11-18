package io.github.zzuegg.brainiac.diffusion;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.zip.CRC32;
import java.util.zip.Deflater;

/**
 * A generated video: its frames as 8-bit RGB images and the rate they play at. It writes
 * itself as an animated PNG (plays in every browser) or an uncompressed AVI (plays in video
 * players), with no codec libraries.
 *
 * @param frames the frames, all of one size
 * @param fps frames per second (Wan: 16)
 * @param seed the seed that makes this video again with the same request
 */
public record GeneratedVideo(List<GeneratedImage> frames, int fps, long seed) {

    /** Checks the frames. */
    public GeneratedVideo {
        frames = List.copyOf(frames);
        if (frames.isEmpty() || fps < 1) {
            throw new IllegalArgumentException("a video needs frames and a positive rate");
        }
        GeneratedImage first = frames.get(0);
        for (GeneratedImage f : frames) {
            if (f.width() != first.width() || f.height() != first.height()) {
                throw new IllegalArgumentException("frames of different sizes");
            }
        }
    }

    /** @return the width in pixels */
    public int width() {
        return frames.get(0).width();
    }

    /** @return the height in pixels */
    public int height() {
        return frames.get(0).height();
    }

    /** @return the length in seconds */
    public double seconds() {
        return frames.size() / (double) fps;
    }

    // ------------------------------------------------------------------------------- APNG

    /**
     * Encodes the video as an animated PNG that loops forever.
     *
     * @return the file's bytes
     */
    public byte[] apng() {
        int w = width();
        int h = height();
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.writeBytes(new byte[] {(byte) 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'});
        chunk(out, "IHDR", ByteBuffer.allocate(13).putInt(w).putInt(h).put((byte) 8).put((byte) 2).put((byte) 0)
                .put((byte) 0).put((byte) 0).array());
        chunk(out, "acTL", ByteBuffer.allocate(8).putInt(frames.size()).putInt(0).array());
        int sequence = 0;
        for (int i = 0; i < frames.size(); i++) {
            ByteBuffer control = ByteBuffer.allocate(26).putInt(sequence++).putInt(w).putInt(h).putInt(0).putInt(0)
                    .putShort((short) 1).putShort((short) fps).put((byte) 0).put((byte) 0);
            chunk(out, "fcTL", control.array());
            byte[] data = deflate(frames.get(i));
            if (i == 0) {
                chunk(out, "IDAT", data);
            } else {
                chunk(out, "fdAT", ByteBuffer.allocate(4 + data.length).putInt(sequence++).put(data).array());
            }
        }
        chunk(out, "IEND", new byte[0]);
        return out.toByteArray();
    }

    private static byte[] deflate(GeneratedImage f) {
        int w = f.width();
        int h = f.height();
        byte[] raw = new byte[(w * 3 + 1) * h];
        byte[] rgb = f.rgb();
        for (int y = 0; y < h; y++) {
            int row = y * (w * 3 + 1);
            raw[row] = 1;   // the "sub" filter: each byte minus the one a pixel to the left
            for (int x = 0; x < w * 3; x++) {
                int left = x >= 3 ? rgb[y * w * 3 + x - 3] : 0;
                raw[row + 1 + x] = (byte) (rgb[y * w * 3 + x] - left);
            }
        }
        Deflater deflater = new Deflater(6);
        deflater.setInput(raw);
        deflater.finish();
        ByteArrayOutputStream z = new ByteArrayOutputStream();
        byte[] buf = new byte[1 << 16];
        while (!deflater.finished()) {
            z.write(buf, 0, deflater.deflate(buf));
        }
        deflater.end();
        return z.toByteArray();
    }

    private static void chunk(ByteArrayOutputStream out, String type, byte[] data) {
        byte[] t = type.getBytes(StandardCharsets.US_ASCII);
        out.writeBytes(ByteBuffer.allocate(4).putInt(data.length).array());
        out.writeBytes(t);
        out.writeBytes(data);
        CRC32 crc = new CRC32();
        crc.update(t);
        crc.update(data);
        out.writeBytes(ByteBuffer.allocate(4).putInt((int) crc.getValue()).array());
    }

    // ------------------------------------------------------------------------------- AVI

    /**
     * Encodes the video as an AVI file of uncompressed 24-bit frames, which any video player
     * opens; large, as uncompressed video is (about 1.2 MB per 832x480 frame).
     *
     * @return the file's bytes
     */
    public byte[] avi() {
        int w = width();
        int h = height();
        int stride = (w * 3 + 3) & ~3;
        int frameBytes = stride * h;
        int n = frames.size();
        Riff riff = new Riff();
        riff.begin("RIFF", "AVI ");
        riff.begin("LIST", "hdrl");
        // the main header
        riff.chunk("avih", le(56).putInt(1_000_000 / fps).putInt(frameBytes * fps).putInt(0).putInt(0x10)
                .putInt(n).putInt(0).putInt(1).putInt(frameBytes).putInt(w).putInt(h).putInt(0).putInt(0)
                .putInt(0).putInt(0).array());
        riff.begin("LIST", "strl");
        ByteBuffer strh = le(56).put("vids".getBytes(StandardCharsets.US_ASCII)).putInt(0).putInt(0)
                .putShort((short) 0).putShort((short) 0).putInt(0).putInt(1).putInt(fps).putInt(0).putInt(n)
                .putInt(frameBytes).putInt(-1).putInt(0).putShort((short) 0).putShort((short) 0)
                .putShort((short) w).putShort((short) h);
        riff.chunk("strh", strh.array());
        // BITMAPINFOHEADER: bottom-up rows of blue, green, red
        riff.chunk("strf", le(40).putInt(40).putInt(w).putInt(h).putShort((short) 1).putShort((short) 24).putInt(0)
                .putInt(frameBytes).putInt(0).putInt(0).putInt(0).putInt(0).array());
        riff.end();
        riff.end();
        riff.begin("LIST", "movi");
        int moviStart = riff.size() - 4;   // offsets in the index count from the "movi" tag
        int[] offsets = new int[n];
        byte[] frame = new byte[frameBytes];
        for (int i = 0; i < n; i++) {
            byte[] rgb = frames.get(i).rgb();
            for (int y = 0; y < h; y++) {
                int src = (h - 1 - y) * w * 3;
                int dst = y * stride;
                for (int x = 0; x < w; x++) {
                    frame[dst + 3 * x] = rgb[src + 3 * x + 2];
                    frame[dst + 3 * x + 1] = rgb[src + 3 * x + 1];
                    frame[dst + 3 * x + 2] = rgb[src + 3 * x];
                }
            }
            offsets[i] = riff.size() - moviStart;
            riff.chunk("00db", frame);
        }
        riff.end();
        ByteBuffer index = le(16 * n);
        for (int i = 0; i < n; i++) {
            index.put("00db".getBytes(StandardCharsets.US_ASCII)).putInt(0x10).putInt(offsets[i]).putInt(frameBytes);
        }
        riff.chunk("idx1", index.array());
        riff.end();
        return riff.bytes();
    }

    private static ByteBuffer le(int size) {
        return ByteBuffer.allocate(size).order(ByteOrder.LITTLE_ENDIAN);
    }

    /** Writes RIFF chunks and lists, filling in their sizes when they end. */
    private static final class Riff {
        private final ByteArrayOutputStream out = new ByteArrayOutputStream();
        private final java.util.ArrayDeque<Integer> open = new java.util.ArrayDeque<>();
        private byte[] done;

        void begin(String id, String type) {
            out.writeBytes(id.getBytes(StandardCharsets.US_ASCII));
            open.push(out.size());
            out.writeBytes(new byte[4]);
            out.writeBytes(type.getBytes(StandardCharsets.US_ASCII));
        }

        void chunk(String id, byte[] data) {
            out.writeBytes(id.getBytes(StandardCharsets.US_ASCII));
            out.writeBytes(le(4).putInt(data.length).array());
            out.writeBytes(data);
            if (data.length % 2 != 0) {
                out.write(0);
            }
        }

        /** Ends the innermost list; its size is patched in {@link #bytes()}. */
        void end() {
            int at = open.pop();
            sizes.add(new int[] {at, out.size() - at - 4});
        }

        private final java.util.List<int[]> sizes = new java.util.ArrayList<>();

        int size() {
            return out.size();
        }

        byte[] bytes() {
            if (done == null) {
                done = out.toByteArray();
                for (int[] s : sizes) {
                    ByteBuffer.wrap(done, s[0], 4).order(ByteOrder.LITTLE_ENDIAN).putInt(s[1]);
                }
            }
            return done;
        }
    }
}
