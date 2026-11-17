package io.github.zzuegg.brainiac.audio;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * WAV files: decoding PCM (8, 16, 24 and 32 bit), IEEE float (32 and 64 bit) and
 * extensible-format files to mono {@link Audio}, and encoding 16-bit PCM.
 *
 * <pre>{@code
 * Audio clip = Wav.read(Path.of("speech.wav")).resample(16_000);
 * Files.write(Path.of("out.wav"), Wav.encode(clip));
 * }</pre>
 */
public final class Wav {

    private static final int PCM = 1;
    private static final int FLOAT = 3;
    private static final int EXTENSIBLE = 0xFFFE;

    private Wav() {
    }

    /**
     * Reads a WAV file, mixing its channels to mono.
     *
     * @param file the file
     * @return its audio, at the file's sample rate
     * @throws UncheckedIOException if it cannot be read
     * @throws IllegalArgumentException if it is not a WAV file Brainiac can decode
     */
    public static Audio read(Path file) {
        try {
            return decode(Files.readAllBytes(file));
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read " + file + ": " + e.getMessage(), e);
        }
    }

    /**
     * Decodes a WAV file's bytes, mixing its channels to mono.
     *
     * @param bytes the whole file
     * @return its audio, at the file's sample rate
     * @throws IllegalArgumentException if it is not a WAV file Brainiac can decode
     */
    public static Audio decode(byte[] bytes) {
        ByteBuffer b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
        if (bytes.length < 12 || !tag(b, 0).equals("RIFF") || !tag(b, 8).equals("WAVE")) {
            throw new IllegalArgumentException("not a WAV file (no RIFF/WAVE header); convert the audio to WAV first");
        }
        int format = -1;
        int channels = 0;
        int rate = 0;
        int bits = 0;
        int dataStart = -1;
        int dataLength = 0;
        int pos = 12;
        while (pos + 8 <= bytes.length) {
            String id = tag(b, pos);
            long size = Integer.toUnsignedLong(b.getInt(pos + 4));
            int body = pos + 8;
            if (id.equals("fmt ")) {
                if (size < 16) {
                    throw new IllegalArgumentException("WAV fmt chunk too short: " + size + " bytes");
                }
                format = b.getShort(body) & 0xFFFF;
                channels = b.getShort(body + 2) & 0xFFFF;
                rate = b.getInt(body + 4);
                bits = b.getShort(body + 14) & 0xFFFF;
                if (format == EXTENSIBLE && size >= 40) {
                    format = b.getShort(body + 24) & 0xFFFF;   // the sub-format GUID starts with the code
                }
            } else if (id.equals("data")) {
                dataStart = body;
                // streams written before their length is known say 0 or 0xFFFFFFFF
                long available = bytes.length - (long) body;
                dataLength = (int) (size == 0 || size > available ? available : size);
                break;
            }
            pos = Math.toIntExact(body + size + (size & 1));
        }
        if (format < 0) {
            throw new IllegalArgumentException("WAV file has no fmt chunk");
        }
        if (dataStart < 0) {
            throw new IllegalArgumentException("WAV file has no data chunk");
        }
        if (channels < 1 || rate <= 0) {
            throw new IllegalArgumentException("WAV file has " + channels + " channels at " + rate + " Hz");
        }
        boolean pcm = format == PCM && (bits == 8 || bits == 16 || bits == 24 || bits == 32);
        boolean flt = format == FLOAT && (bits == 32 || bits == 64);
        if (!pcm && !flt) {
            throw new IllegalArgumentException("unsupported WAV encoding (format " + format + ", " + bits
                    + " bits); use PCM or 32-bit float WAV");
        }
        int width = bits / 8;
        int frames = dataLength / (width * channels);
        float[] out = new float[frames];
        for (int f = 0; f < frames; f++) {
            double sum = 0;
            for (int c = 0; c < channels; c++) {
                int at = dataStart + (f * channels + c) * width;
                sum += sample(b, at, bits, flt);
            }
            out[f] = (float) (sum / channels);
        }
        return new Audio(out, rate);
    }

    private static double sample(ByteBuffer b, int at, int bits, boolean flt) {
        if (flt) {
            return bits == 32 ? b.getFloat(at) : b.getDouble(at);
        }
        return switch (bits) {
            case 8 -> ((b.get(at) & 0xFF) - 128) / 128.0;
            case 16 -> b.getShort(at) / 32768.0;
            case 24 -> ((b.get(at) & 0xFF) | (b.get(at + 1) & 0xFF) << 8 | b.get(at + 2) << 16) / 8388608.0;
            default -> b.getInt(at) / 2147483648.0;
        };
    }

    private static String tag(ByteBuffer b, int at) {
        char[] c = new char[4];
        for (int i = 0; i < 4; i++) {
            c[i] = (char) (b.get(at + i) & 0xFF);
        }
        return new String(c);
    }

    /**
     * Encodes mono audio as a 16-bit PCM WAV file; samples outside {@code [-1, 1]} are clipped.
     *
     * @param audio the audio
     * @return the file's bytes
     */
    public static byte[] encode(Audio audio) {
        float[] s = audio.samples();
        int data = s.length * 2;
        ByteBuffer b = ByteBuffer.allocate(44 + data).order(ByteOrder.LITTLE_ENDIAN);
        b.put("RIFF".getBytes(java.nio.charset.StandardCharsets.US_ASCII)).putInt(36 + data);
        b.put("WAVE".getBytes(java.nio.charset.StandardCharsets.US_ASCII));
        b.put("fmt ".getBytes(java.nio.charset.StandardCharsets.US_ASCII)).putInt(16);
        b.putShort((short) PCM).putShort((short) 1).putInt(audio.sampleRate()).putInt(audio.sampleRate() * 2);
        b.putShort((short) 2).putShort((short) 16);
        b.put("data".getBytes(java.nio.charset.StandardCharsets.US_ASCII)).putInt(data);
        for (float v : s) {
            float c = Math.max(-1f, Math.min(1f, v));
            b.putShort((short) Math.round(c * 32767f));
        }
        return b.array();
    }

    /**
     * Writes mono audio as a 16-bit PCM WAV file.
     *
     * @param audio the audio
     * @param file where to write it
     * @throws UncheckedIOException if it cannot be written
     */
    public static void write(Audio audio, Path file) {
        try {
            Files.write(file, encode(audio));
        } catch (IOException e) {
            throw new UncheckedIOException("cannot write " + file + ": " + e.getMessage(), e);
        }
    }
}
