package io.github.zzuegg.brainiac.data;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Reads and writes WAV files: PCM with 8, 16, 24 or 32 bits and 32-bit float samples, any
 * number of channels. Samples are floats in {@code [-1, 1]}.
 */
public final class Wav {

    /**
     * Audio as one array of samples per channel.
     *
     * @param sampleRate samples per second
     * @param channels the samples of each channel, all the same length
     */
    public record Audio(int sampleRate, float[][] channels) {

        /** @return the samples per channel */
        public int length() {
            return channels.length == 0 ? 0 : channels[0].length;
        }

        /** @return the channels averaged into one */
        public float[] mono() {
            float[] out = new float[length()];
            for (float[] c : channels) {
                for (int i = 0; i < out.length; i++) {
                    out[i] += c[i] / channels.length;
                }
            }
            return out;
        }
    }

    private Wav() {
    }

    /**
     * Reads a WAV file.
     *
     * @param file the file
     * @return the audio
     * @throws IOException if it cannot be read or is not a supported WAV file
     */
    public static Audio read(Path file) throws IOException {
        ByteBuffer b = ByteBuffer.wrap(Files.readAllBytes(file)).order(ByteOrder.LITTLE_ENDIAN);
        if (b.remaining() < 12 || b.getInt(0) != 0x46464952 || b.getInt(8) != 0x45564157) {   // RIFF, WAVE
            throw new IOException(file + " is not a WAV file");
        }
        int pos = 12;
        int format = -1;
        int channels = 0;
        int rate = 0;
        int bits = 0;
        while (pos + 8 <= b.limit()) {
            int id = b.getInt(pos);
            int size = b.getInt(pos + 4);
            int body = pos + 8;
            if (id == 0x20746d66) {   // "fmt "
                format = b.getShort(body) & 0xffff;
                channels = b.getShort(body + 2) & 0xffff;
                rate = b.getInt(body + 4);
                bits = b.getShort(body + 14) & 0xffff;
                if (format == 0xfffe && size >= 26) {   // extensible: the real format follows
                    format = b.getShort(body + 24) & 0xffff;
                }
            } else if (id == 0x61746164) {   // "data"
                if (format < 0) {
                    throw new IOException(file + ": data before format");
                }
                int length = Math.min(size, b.limit() - body);
                return new Audio(rate, samples(b, body, length, format, channels, bits, file));
            }
            pos = body + size + (size & 1);
        }
        throw new IOException(file + " has no audio data");
    }

    private static float[][] samples(ByteBuffer b, int at, int length, int format, int channels, int bits, Path file)
            throws IOException {
        int bytes = bits / 8;
        boolean pcm = format == 1;
        boolean flt = format == 3 && bits == 32;
        if (!(pcm && (bits == 8 || bits == 16 || bits == 24 || bits == 32)) && !flt || channels < 1) {
            throw new IOException(file + ": unsupported WAV format " + format + " with " + bits + " bits");
        }
        int frames = length / (bytes * channels);
        float[][] out = new float[channels][frames];
        for (int f = 0; f < frames; f++) {
            for (int c = 0; c < channels; c++) {
                int p = at + (f * channels + c) * bytes;
                out[c][f] = flt ? b.getFloat(p) : switch (bits) {
                    case 8 -> ((b.get(p) & 0xff) - 128) / 128f;
                    case 16 -> b.getShort(p) / 32768f;
                    case 24 -> ((b.get(p) & 0xff) | (b.get(p + 1) & 0xff) << 8 | b.get(p + 2) << 16) / 8388608f;
                    default -> b.getInt(p) / 2147483648f;
                };
            }
        }
        return out;
    }

    /**
     * Writes 16-bit PCM.
     *
     * @param audio the audio; samples are clamped to {@code [-1, 1]}
     * @param file where to write
     * @throws IOException if it cannot be written
     */
    public static void write(Audio audio, Path file) throws IOException {
        int channels = audio.channels().length;
        int frames = audio.length();
        int data = frames * channels * 2;
        ByteBuffer b = ByteBuffer.allocate(44 + data).order(ByteOrder.LITTLE_ENDIAN);
        b.putInt(0x46464952).putInt(36 + data).putInt(0x45564157);
        b.putInt(0x20746d66).putInt(16).putShort((short) 1).putShort((short) channels).putInt(audio.sampleRate())
                .putInt(audio.sampleRate() * channels * 2).putShort((short) (channels * 2)).putShort((short) 16);
        b.putInt(0x61746164).putInt(data);
        for (int f = 0; f < frames; f++) {
            for (int c = 0; c < channels; c++) {
                b.putShort((short) Math.round(Math.clamp(audio.channels()[c][f], -1f, 1f) * 32767));
            }
        }
        Files.write(file, b.array());
    }
}
