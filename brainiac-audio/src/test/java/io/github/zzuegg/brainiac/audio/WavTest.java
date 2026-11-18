package io.github.zzuegg.brainiac.audio;

import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class WavTest {

    /** A WAV file of interleaved frames written by hand. */
    private static byte[] wav(int format, int channels, int rate, int bits, byte[] data) {
        ByteBuffer b = ByteBuffer.allocate(44 + data.length + 12).order(ByteOrder.LITTLE_ENDIAN);
        b.put("RIFF".getBytes()).putInt(36 + data.length + 12).put("WAVE".getBytes());
        b.put("LIST".getBytes()).putInt(4).put("INFO".getBytes());   // a chunk to skip
        b.put("fmt ".getBytes()).putInt(16).putShort((short) format).putShort((short) channels).putInt(rate)
                .putInt(rate * channels * bits / 8).putShort((short) (channels * bits / 8)).putShort((short) bits);
        b.put("data".getBytes()).putInt(data.length).put(data);
        return b.array();
    }

    @Test
    void roundTrips16Bit() {
        float[] s = {0, 0.5f, -0.5f, 1, -1, 0.25f};
        Audio a = Wav.decode(Wav.encode(new Audio(s, 22_050)));
        assertEquals(22_050, a.sampleRate());
        for (int i = 0; i < s.length; i++) {
            assertEquals(s[i], a.samples()[i], 1e-4);
        }
    }

    @Test
    void mixesStereo24BitAndReadsFloat() {
        ByteBuffer d = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN);
        // frame 0: left 0.5, right -0.5; frame 1: left 0.25, right 0.25 (24-bit)
        for (int v : new int[] {4194304, -4194304, 2097152, 2097152}) {
            d.put((byte) v).put((byte) (v >> 8)).put((byte) (v >> 16));
        }
        Audio a = Wav.decode(wav(1, 2, 8000, 24, d.array()));
        assertEquals(2, a.samples().length);
        assertEquals(0, a.samples()[0], 1e-6);
        assertEquals(0.25, a.samples()[1], 1e-6);
        ByteBuffer f = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putFloat(0.75f).putFloat(-0.125f);
        Audio fl = Wav.decode(wav(3, 1, 48_000, 32, f.array()));
        assertEquals(0.75f, fl.samples()[0]);
        assertEquals(-0.125f, fl.samples()[1]);
        assertEquals(16_000, fl.resample(16_000).sampleRate());
    }

    @Test
    void explainsWhatItCannotRead() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> Wav.decode("OggS....".getBytes()));
        assertTrue(e.getMessage().contains("not a WAV file"));
        assertThrows(IllegalArgumentException.class, () -> Wav.decode(wav(85, 1, 8000, 16, new byte[4])));
    }

    @Test
    void splitsSentences() {
        assertEquals(List.of("Hello there.", "How are you?", "Fine, 3.5 cats"),
                VitsSpeech.sentences("Hello there. How are you?\nFine, 3.5 cats"));
        assertTrue(WhisperTranscriber.compressionRatio("the cat ".repeat(40)) > 2.4);
        assertTrue(WhisperTranscriber.compressionRatio("The quick brown fox jumps over the lazy dog.") < 2.4);
    }
}
