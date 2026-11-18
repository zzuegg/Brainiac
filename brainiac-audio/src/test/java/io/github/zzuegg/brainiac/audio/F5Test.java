package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.tensor.DType;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** F5-TTS and Vocos against the f5_tts and vocos packages with deterministic noise (see make_f5.py). */
class F5Test {

    private static Map<String, Object> expected;

    @BeforeAll
    static void read() throws Exception {
        expected = Json.object(Json.read(WhisperTest.resource("f5").resolve("expected.json")));
    }

    private static F5Speech load() {
        return F5Speech.load(WhisperTest.resource("f5"), WhisperTest.resource("vocos"), TestDevice.get(),
                DType.FLOAT32);
    }

    @Test
    void textIsSpelledAsF5Does() {
        assertEquals(strings(expected.get("chars")),
                F5Tts.spell(expected.get("refText") + "  " + expected.get("genText")));
        for (Object o : Json.array(expected.get("textCases"))) {
            Map<String, Object> c = Json.object(o);
            assertEquals(String.join("", strings(c.get("chars"))), String.join("", F5Tts.spell((String) c.get("text"))),
                    (String) c.get("text"));
        }
    }

    @Test
    void textIsChunkedAsF5Does() {
        for (Object o : Json.array(expected.get("chunkCases"))) {
            Map<String, Object> c = Json.object(o);
            assertEquals(strings(c.get("chunks")), F5Speech.chunk((String) c.get("text"),
                    ((Number) c.get("max")).intValue()));
        }
    }

    @Test
    void referenceMelMatches() {
        try (F5Speech tts = load()) {
            float[][] mel = tts.mel(WhisperTest.floats(expected.get("reference")));
            float[][] ref = WhisperTest.matrix(expected.get("referenceMel"));      // [mels][frames]
            assertEquals(ref[0].length, mel.length, "frames");
            for (int f = 0; f < mel.length; f++) {
                for (int m = 0; m < ref.length; m++) {
                    assertEquals(ref[m][f], mel[f][m], 2e-3f, "frame " + f + " band " + m);
                }
            }
        }
    }

    @Test
    void networkMatchesWithAndWithoutConditions() {
        Map<String, Object> fw = Json.object(expected.get("forward"));
        try (F5Speech tts = load()) {
            int[] ids = Json.array(fw.get("ids")).stream().mapToInt(v -> ((Number) v).intValue()).toArray();
            float[][][] v = tts.model().velocity(WhisperTest.matrix(fw.get("x")), WhisperTest.matrix(fw.get("cond")),
                    ids, ((Number) fw.get("time")).floatValue());
            close(WhisperTest.matrix(fw.get("pred")), v[0], "conditioned");
            close(WhisperTest.matrix(fw.get("nullPred")), v[1], "unconditioned");
        }
    }

    @Test
    void vocosMatches() {
        Map<String, Object> vc = Json.object(expected.get("vocos"));
        try (F5Speech tts = load()) {
            float[] audio = tts.vocoder().decode(WhisperTest.matrix(vc.get("mel")));
            closeWave(WhisperTest.floats(vc.get("audio")), audio);
        }
        try (Vocos same = Vocos.load(WhisperTest.resource("vocos-same"), TestDevice.get(), DType.FLOAT32)) {
            assertTrue(!same.config().center());
            closeWave(WhisperTest.floats(expected.get("vocosSame")), same.decode(WhisperTest.matrix(vc.get("mel"))));
        }
    }

    @Test
    void clonedSpeechMatches() {
        try (F5Speech tts = load()) {
            int steps = ((Number) expected.get("steps")).intValue();
            Audio ref = new Audio(WhisperTest.floats(expected.get("referenceLoaded")), 24_000);
            Audio out = tts.generate(ref, (String) expected.get("refText"), (String) expected.get("genText"), 1,
                    new F5Speech.Sampling(steps, 2f, -1f), i -> KokoroTest.pseudoNoise(), () -> false, null);
            closeWave(WhisperTest.floats(expected.get("audio")), out.samples());
        }
    }

    @Test
    void aFolderOfModelsIsNotAnF5Model() {
        Path f5 = WhisperTest.resource("f5");
        assertTrue(F5Speech.isF5(f5));
        // the fixtures folder holds f5/ and vocos/ side by side: a collection, not a model
        org.junit.jupiter.api.Assertions.assertFalse(F5Speech.isF5(f5.getParent()));
    }

    @Test
    void cloningIsOfferedAndNeedsAReference() {
        try (F5Speech tts = load()) {
            assertTrue(tts.canClone());
            IllegalArgumentException e = org.junit.jupiter.api.Assertions.assertThrows(IllegalArgumentException.class,
                    () -> tts.synthesize("Hello."));
            assertTrue(e.getMessage().contains("reference"), e.getMessage());
            assertEquals("Hi there. ", F5Speech.referenceText("Hi there"));
            assertEquals("Hi there. ", F5Speech.referenceText("Hi there."));
        }
    }

    private static void close(float[][] want, float[][] got, String what) {
        assertEquals(want.length, got.length, what + " frames");
        double scale = 0;
        for (float[] r : want) {
            for (float v : r) {
                scale = Math.max(scale, Math.abs(v));
            }
        }
        for (int i = 0; i < want.length; i++) {
            for (int j = 0; j < want[i].length; j++) {
                assertEquals(want[i][j], got[i][j], 1e-3 * scale + 1e-4, what + " [" + i + "][" + j + "]");
            }
        }
    }

    private static void closeWave(float[] want, float[] got) {
        assertEquals(want.length, got.length, "samples");
        double scale = 0;
        for (float v : want) {
            scale = Math.max(scale, Math.abs(v));
        }
        for (int i = 0; i < want.length; i++) {
            assertEquals(want[i], got[i], 2e-3 * scale, "sample " + i);
        }
    }

    private static List<String> strings(Object o) {
        return Json.array(o).stream().map(String.class::cast).toList();
    }
}
