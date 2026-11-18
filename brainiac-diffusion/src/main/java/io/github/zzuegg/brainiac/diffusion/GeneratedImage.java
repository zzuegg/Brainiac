package io.github.zzuegg.brainiac.diffusion;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.zip.CRC32;
import java.util.zip.Deflater;

/**
 * An image as 8-bit RGB, row by row from the top, three bytes per pixel.
 *
 * @param width the width in pixels
 * @param height the height in pixels
 * @param rgb the pixels, {@code width * height * 3} bytes
 * @param seed the seed that makes this image again with the same request (or -1 for a preview)
 */
public record GeneratedImage(int width, int height, byte[] rgb, long seed) {

    /**
     * Validates the size.
     *
     * @param width the width in pixels
     * @param height the height in pixels
     * @param rgb the pixels
     * @param seed the seed
     */
    public GeneratedImage {
        if ((long) width * height * 3 != rgb.length) {
            throw new IllegalArgumentException(width + "x" + height + " RGB needs " + (long) width * height * 3
                    + " bytes, got " + rgb.length);
        }
    }

    /**
     * Returns one pixel.
     *
     * @param x the column
     * @param y the row
     * @return {@code 0xRRGGBB}
     */
    public int pixel(int x, int y) {
        int i = (y * width + x) * 3;
        return (rgb[i] & 0xff) << 16 | (rgb[i + 1] & 0xff) << 8 | rgb[i + 2] & 0xff;
    }

    /**
     * Encodes the image as a PNG file.
     *
     * @return the PNG bytes
     */
    public byte[] png() {
        byte[] raw = new byte[(width * 3 + 1) * height];
        for (int y = 0; y < height; y++) {
            raw[y * (width * 3 + 1)] = 0;   // no filter
            System.arraycopy(rgb, y * width * 3, raw, y * (width * 3 + 1) + 1, width * 3);
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
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.writeBytes(new byte[] {(byte) 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'});
        ByteBuffer header = ByteBuffer.allocate(13).putInt(width).putInt(height)
                .put((byte) 8).put((byte) 2).put((byte) 0).put((byte) 0).put((byte) 0);
        chunk(out, "IHDR", header.array());
        chunk(out, "IDAT", z.toByteArray());
        chunk(out, "IEND", new byte[0]);
        return out.toByteArray();
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
}
