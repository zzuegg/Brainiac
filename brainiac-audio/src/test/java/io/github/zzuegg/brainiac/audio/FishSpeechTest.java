package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.tensor.DType;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

import static io.github.zzuegg.brainiac.audio.Qwen3TtsTest.ints2;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Fish Audio S2-Pro against tiny random models and the reference outputs of make_fish.py. */
class FishSpeechTest {

    private static Map<String, Object> expected;
    private static FishSpeech tts;

    @BeforeAll
    static void load() throws Exception {
        Path dir = WhisperTest.resource("fish");
        expected = Json.object(Json.read(dir.resolve("expected.json")));
        tts = FishSpeech.load(dir, TestDevice.get(), DType.FLOAT32);
    }

    @AfterAll
    static void close() {
        tts.close();
    }

    private static final FishSpeech.Sampling GREEDY = new FishSpeech.Sampling(0.7f, 0.7f, 1, 10);

    private static void close(float[] want, float[] got, double tol, String what) {
        assertEquals(want.length, got.length, what + " size");
        for (int i = 0; i < want.length; i++) {
            assertFalse(Float.isNaN(got[i]), what + " " + i + " is NaN");
            assertEquals(want[i], got[i], tol, what + " " + i);
        }
    }

    @Test
    void codecEncodesAsTheReference() {
        int[][] want = ints2(expected.get("codec_codes"));
        int[][] got = tts.codec().encode(WhisperTest.floats(expected.get("codec_wave")));
        assertEquals(want.length, got.length, "codebooks");
        int differ = 0;
        for (int q = 0; q < want.length; q++) {
            assertEquals(want[q].length, got[q].length, "frames");
            for (int f = 0; f < want[q].length; f++) {
                differ += want[q][f] == got[q][f] ? 0 : 1;
            }
        }
        // a near tie between codewords may go either way in float32
        assertTrue(differ <= 2, differ + " codes differ");
    }

    @Test
    void codecDecodesAsTheReference() {
        int[][] codes = ints2(expected.get("decode_codes"));
        close(WhisperTest.floats(expected.get("decode_audio")), tts.codec().decode(codes), 2e-3, "sample");
    }

    @Test
    void samplingFiltersAsTheReference() {
        float[] logits = WhisperTest.floats(expected.get("sample_logits"));
        List<Object> probs = Json.array(expected.get("sample_probs"));
        float[][] settings = {{0.7f, 0.7f, 30}, {1.0f, 0.9f, 30}, {0.7f, 1.0f, 5}};
        for (int i = 0; i < settings.length; i++) {
            float[] got = FishSpeech.probs(logits, settings[i][0], settings[i][1], (int) settings[i][2]);
            close(WhisperTest.floats(probs.get(i)), got, 1e-5, "case " + i);
        }
    }

    private static FishSpeech.Reference reference() {
        return new FishSpeech.Reference("Guten Morgen!", ints2(expected.get("ref_codes")));
    }

    @Test
    void promptsAreTheReferences() {
        assertArrayEquals(ints2(expected.get("plain_prompt")), tts.prompt(null, List.of("Hello world."), List.of()));
        assertArrayEquals(ints2(expected.get("clone_prompt")),
                tts.prompt(reference(), List.of("Hello world."), List.of()));
        int[][] first = transpose(ints2(expected.get("clone_frames")));
        assertArrayEquals(ints2(expected.get("next_prompt")),
                tts.prompt(reference(), List.of("Hello world.", "Second part."), List.<int[][]>of(first)));
    }

    private static int[][] transpose(int[][] frames) {
        int[][] out = new int[frames[0].length][frames.length];
        for (int f = 0; f < frames.length; f++) {
            for (int q = 0; q < frames[f].length; q++) {
                out[q][f] = frames[f][q];
            }
        }
        return out;
    }

    @Test
    void framesAreTheReferences() {
        for (String c : new String[] {"plain", "clone", "next"}) {
            int[][] got = tts.generate(ints2(expected.get(c + "_prompt")), GREEDY, new SplittableRandom(1), () -> false);
            assertArrayEquals(ints2(expected.get(c + "_frames")), got, c);
        }
    }

    @Test
    void speaksAndClones() {
        tts.sampling(new FishSpeech.Sampling(0.7f, 0.7f, 30, 12));
        try {
            Audio a = tts.synthesize("Hello world. Second part.", SpeechOptions.defaults().seed(2), null);
            assertEquals(44_100, a.sampleRate());
            assertTrue(a.samples().length > 0, "samples");
            tts.addVoice("buzz", new Audio(WhisperTest.floats(expected.get("codec_wave")), 44_100), "Guten Morgen!");
            Audio b = tts.synthesize("Hi.", SpeechOptions.defaults().voice("buzz").seed(1), null);
            assertTrue(b.samples().length > 0, "cloned samples");
            for (float s : b.samples()) {
                assertFalse(Float.isNaN(s), "NaN");
            }
            tts.removeVoice("buzz");
        } finally {
            tts.sampling(FishSpeech.Sampling.defaults());
        }
    }
}
