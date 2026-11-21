package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.tensor.DType;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Map;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/** Qwen3-TTS against tiny random models and the reference outputs of make_qwen3tts.py. */
class Qwen3TtsTest {

    static int[][] ints2(Object list) {
        return Json.array(list).stream()
                .map(l -> Json.array(l).stream().mapToInt(v -> ((Number) v).intValue()).toArray())
                .toArray(int[][]::new);
    }

    static int[] ints(Object list) {
        return Json.array(list).stream().mapToInt(v -> ((Number) v).intValue()).toArray();
    }

    @Test
    void codecDecodesInChunksAsTheReference() throws Exception {
        Path dir = WhisperTest.resource("qwen3tts/speech_tokenizer");
        Map<String, Object> expected = Json.object(Json.read(dir.resolve("expected.json")));
        int[][] codes = ints2(expected.get("codes"));
        float[] want = WhisperTest.floats(expected.get("audio"));
        try (Qwen3TtsCodec codec = Qwen3TtsCodec.load(dir, TestDevice.get(), DType.FLOAT32)) {
            float[] got = codec.decode(codes);
            assertEquals(want.length, got.length, "samples");
            for (int i = 0; i < want.length; i++) {
                assertEquals(want[i], got[i], 2e-3, "sample " + i);
            }
        }
    }

    @Test
    void codecEncodesAsMimi() throws Exception {
        Path dir = WhisperTest.resource("qwen3tts/speech_tokenizer");
        Map<String, Object> expected = Json.object(Json.read(dir.resolve("expected.json")));
        int[][] want = ints2(expected.get("encoded"));
        try (Qwen3TtsCodec codec = Qwen3TtsCodec.load(dir, TestDevice.get(), DType.FLOAT32)) {
            int[][] got = codec.encode(WhisperTest.floats(expected.get("wave")));
            assertEquals(want.length, got.length, "frames");
            int differ = 0;
            for (int f = 0; f < want.length; f++) {
                assertEquals(want[f][0], got[f][0], "semantic code of frame " + f);
                for (int q = 1; q < want[f].length; q++) {
                    differ += want[f][q] == got[f][q] ? 0 : 1;
                }
            }
            // a near tie in a later residual stage may go either way in float32
            assertEquals(0, differ, 0.002 * want.length * (want[0].length - 1), "acoustic codes that differ");
        }
    }

    static float[][] floats2(Object list) {
        return Json.array(list).stream().map(WhisperTest::floats).toArray(float[][]::new);
    }

    @Test
    void speakerEncoderReadsTheVoiceAsTheReference() throws Exception {
        Path dir = WhisperTest.resource("qwen3tts");
        Map<String, Object> expected = Json.object(Json.read(dir.resolve("expected.json")));
        float[] voice = WhisperTest.floats(expected.get("voice"));
        float[][] mel = floats2(expected.get("mel"));
        float[] xvector = WhisperTest.floats(expected.get("xvector"));
        try (Qwen3Speech tts = Qwen3Speech.load(dir, TestDevice.get(), DType.FLOAT32)) {
            Qwen3SpeakerEncoder enc = tts.speakerEncoder();
            float[] got = Var.noGrad(() -> enc.mel(voice)).value().toArray();
            assertEquals(mel.length * mel[0].length, got.length, "mel size");
            for (int f = 0; f < mel.length; f++) {
                for (int m = 0; m < mel[f].length; m++) {
                    assertEquals(mel[f][m], got[f * mel[f].length + m], 2e-3, "mel " + f + "," + m);
                }
            }
            float[] x = enc.embed(voice);
            assertEquals(xvector.length, x.length, "embedding size");
            for (int i = 0; i < x.length; i++) {
                assertEquals(xvector[i], x[i], 2e-3 * Math.max(1, Math.abs(xvector[i])), "embedding " + i);
            }
        }
    }

    @Test
    void talkerWritesTheReferenceFrames() throws Exception {
        Path dir = WhisperTest.resource("qwen3tts");
        Map<String, Object> e = Json.object(Json.read(dir.resolve("expected.json")));
        int[] text = ints(e.get("text_ids"));
        int[] ref = ints(e.get("ref_ids"));
        int[] instruct = ints(e.get("instruct_ids"));
        int[][] refCode = ints2(e.get("ref_code"));
        float[] xv = WhisperTest.floats(e.get("xvector"));
        Map<String, Object> cases = Json.object(e.get("cases"));
        Map<String, Qwen3Speech.Prompt> prompts = new java.util.LinkedHashMap<>();
        prompts.put("custom", new Qwen3Speech.Prompt(text, instruct, "alice", "english", null, null, null, true));
        prompts.put("dialect", new Qwen3Speech.Prompt(text, null, "bob", "auto", null, null, null, true));
        prompts.put("design", new Qwen3Speech.Prompt(text, instruct, null, "auto", null, null, null, true));
        prompts.put("streaming", new Qwen3Speech.Prompt(text, null, "alice", "chinese", null, null, null, false));
        prompts.put("xvector", new Qwen3Speech.Prompt(text, null, null, "english", xv, null, null, false));
        prompts.put("icl", new Qwen3Speech.Prompt(text, null, null, "english", xv, ref, refCode, false));
        prompts.put("icl_long_text", new Qwen3Speech.Prompt(text, null, null, "english", xv, ref,
                java.util.Arrays.copyOf(refCode, 2), false));
        prompts.put("icl_nonstreaming", new Qwen3Speech.Prompt(text, null, null, "english", xv, ref, refCode, true));
        try (Qwen3Speech tts = Qwen3Speech.load(dir, TestDevice.get(), DType.FLOAT32)) {
            for (Map.Entry<String, Qwen3Speech.Prompt> c : prompts.entrySet()) {
                int[][] want = ints2(cases.get(c.getKey()));
                int[][] got = tts.generate(c.getValue(), Qwen3Speech.Sampling.greedy(24), new SplittableRandom(1),
                        () -> false);
                assertEquals(want.length, got.length, c.getKey() + ": frames");
                for (int f = 0; f < want.length; f++) {
                    assertArrayEquals(want[f], got[f], c.getKey() + ": frame " + f);
                }
            }
        }
    }

    @Test
    void samplingIsReproducibleAndClonedAudioLosesTheReference() throws Exception {
        Path dir = WhisperTest.resource("qwen3tts");
        Map<String, Object> e = Json.object(Json.read(dir.resolve("expected.json")));
        int[][] refCode = ints2(e.get("ref_code"));
        Qwen3Speech.Prompt p = new Qwen3Speech.Prompt(ints(e.get("text_ids")), null, null, "english",
                WhisperTest.floats(e.get("xvector")), ints(e.get("ref_ids")), refCode, false);
        try (Qwen3Speech tts = Qwen3Speech.load(dir, TestDevice.get(), DType.FLOAT32)) {
            Qwen3Speech.Sampling s = new Qwen3Speech.Sampling(0.9f, 50, 1f, 1.05f, 0.9f, 50, 1f, 16);
            int[][] a = tts.generate(p, s, new SplittableRandom(7), () -> false);
            int[][] b = tts.generate(p, s, new SplittableRandom(7), () -> false);
            assertEquals(a.length, b.length);
            for (int f = 0; f < a.length; f++) {
                assertArrayEquals(a[f], b[f], "frame " + f);
            }
            float[] alone = tts.codec().decode(a);
            float[] after = tts.decode(a, refCode);
            int hop = tts.codec().hop();
            assertEquals(alone.length, after.length, hop, "samples once the reference is cut off");
        }
    }
}
