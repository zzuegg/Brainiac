package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.tensor.DType;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static io.github.zzuegg.brainiac.audio.Qwen3TtsTest.floats2;
import static io.github.zzuegg.brainiac.audio.Qwen3TtsTest.ints;
import static io.github.zzuegg.brainiac.audio.Qwen3TtsTest.ints2;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Chatterbox against tiny random models and the reference outputs of make_chatterbox.py. */
class ChatterboxTest {

    private static Map<String, Object> expected;
    private static ChatterboxSpeech tts;
    private static float[] voice16;

    @BeforeAll
    static void load() throws Exception {
        Path dir = WhisperTest.resource("chatterbox");
        expected = Json.object(Json.read(dir.resolve("expected.json")));
        voice16 = floats("voice16");
        tts = ChatterboxSpeech.load(dir, TestDevice.get(), DType.FLOAT32);
    }

    @AfterAll
    static void close() {
        tts.close();
    }

    private static float[] floats(String key) {
        return WhisperTest.floats(expected.get(key));
    }

    /** Rows of a nested list, flattened. */
    private static float[] flat(String key) {
        float[][] rows = floats2(expected.get(key));
        float[] out = new float[rows.length * rows[0].length];
        for (int r = 0; r < rows.length; r++) {
            System.arraycopy(rows[r], 0, out, r * rows[0].length, rows[0].length);
        }
        return out;
    }

    /** Close to within {@code tol} of the larger of 1 and the largest wanted magnitude. */
    private static void close(float[] want, float[] got, double tol, String what) {
        assertEquals(want.length, got.length, what + " size");
        double scale = 1;
        for (float w : want) {
            scale = Math.max(scale, Math.abs(w));
        }
        for (int i = 0; i < want.length; i++) {
            assertFalse(Float.isNaN(got[i]), what + " " + i + " is NaN");
            assertEquals(want[i], got[i], tol * scale, what + " " + i);
        }
    }

    @Test
    void textTokensAreTheTokenizers() throws Exception {
        Map<String, Object> t = Json.object(Json.read(WhisperTest.resource("chatterbox/tokenizer_expected.json")));
        List<Object> texts = Json.array(t.get("texts"));
        int[][] ids = ints2(t.get("ids"));
        for (int i = 0; i < ids.length; i++) {
            int[] want = new int[ids[i].length + 2];
            want[0] = 15;
            System.arraycopy(ids[i], 0, want, 1, ids[i].length);
            assertArrayEquals(want, tts.textTokens((String) texts.get(i)), (String) texts.get(i));
        }
    }

    @Test
    void punctuationIsNormalizedAsTheReference() {
        assertEquals("Hello there.", ChatterboxSpeech.normalize("hello there"));
        assertEquals("Wait, what?", ChatterboxSpeech.normalize("Wait , what?"));
        assertEquals("It's fine.", ChatterboxSpeech.normalize("It’s  fine"));
        assertEquals("You need to add some text for me to talk.", ChatterboxSpeech.normalize(""));
    }

    @Test
    void voiceEncoderEmbedsAsTheReference() {
        ChatterboxVoiceEncoder ve = tts.voiceEncoder();
        int[] span = ChatterboxVoiceEncoder.trim(voice16);
        assertEquals(((Number) expected.get("ve_trimmed")).intValue(), span[1] - span[0], "trimmed samples");
        float[] trimmed = Arrays.copyOfRange(voice16, span[0], span[1]);
        float[] mel = Var.noGrad(() -> ve.mel(trimmed)).value().toArray();
        float[] want = flat("ve_mel");
        close(Arrays.copyOf(want, mel.length), mel, 2e-4, "mel");
        close(floats("ve_embedding"), ve.embed(voice16), 2e-3, "embedding");
    }

    @Test
    void speechTokenizerReadsCodesAsTheReference() {
        S3SpeechTokenizer s3 = tts.s3();
        close(flat("s3_mel"), Var.noGrad(() -> s3.mel(voice16)).value().toArray(), 1e-3, "log-mel");
        int[] want = ints(expected.get("s3_codes"));
        int[] got = s3.encode(voice16, 0);
        assertEquals(want.length, got.length, "codes");
        int differ = 0;
        for (int i = 0; i < want.length; i++) {
            differ += want[i] == got[i] ? 0 : 1;
        }
        // a value near a rounding boundary may go either way in float32
        assertTrue(differ <= 1, differ + " of " + want.length + " codes differ");
        assertEquals(ints(expected.get("s3_codes_prompt")).length, s3.encode(voice16, 12).length, "prompt codes");
    }

    @Test
    void camPlusEmbedsAsTheReference() {
        float[][] fb = CamPlusEncoder.fbank(voice16);
        float[] got = new float[fb.length * fb[0].length];
        for (int t = 0; t < fb.length; t++) {
            System.arraycopy(fb[t], 0, got, t * fb[0].length, fb[0].length);
        }
        close(flat("fbank"), got, 2e-4, "fbank");
        close(floats("xvector"), tts.camPlus().embed(voice16), 2e-3, "x-vector");
    }

    @Test
    void referenceMelIsS3Gens() {
        float[] want = flat("prompt_mel");                                  // [80, frames]
        float[] got = tts.mel24(floats("voice24"));                         // [frames, 80]
        int frames = want.length / 80;
        float[] t = new float[got.length];
        for (int f = 0; f < frames; f++) {
            for (int m = 0; m < 80; m++) {
                t[m * frames + f] = got[f * 80 + m];
            }
        }
        close(want, t, 2e-4, "mel");
    }

    @Test
    void t3ConditionsAsTheReference() {
        Var cond = tts.t3().cond(floats("ve_embedding"), ints(expected.get("t3_prompt_tokens")), 0.5f);
        close(flat("t3_cond"), cond.value().toArray(), 1e-3, "cond");
    }

    @Test
    void t3WritesTheReferencesTokens() {
        ChatterboxT3 t3 = tts.t3();
        float[] speaker = floats("ve_embedding");
        int[] prompt = ints(expected.get("t3_prompt_tokens"));
        int[] text = ints(expected.get("t3_text_ids"));
        int[] guided = t3.generate(text, t3.cond(speaker, prompt, 0.5f), ChatterboxT3.Sampling.greedy(20, 0.5f, 1.2f),
                null, () -> false);
        assertArrayEquals(ints(expected.get("t3_speech")), guided, "with guidance");
        int[] plain = t3.generate(text, t3.cond(speaker, prompt, 1.2f), ChatterboxT3.Sampling.greedy(20, 0f, 1f),
                null, () -> false);
        assertArrayEquals(ints(expected.get("t3_speech_exaggerated_nocfg")), plain, "exaggerated, no guidance");
    }

    @Test
    void flowMakesTheReferencesMel() {
        S3Flow flow = tts.flow();
        int[] prompt = ints(expected.get("flow_prompt_tokens"));
        int[] tokens = ints(expected.get("flow_tokens"));
        int[] all = new int[prompt.length + tokens.length];
        System.arraycopy(prompt, 0, all, 0, prompt.length);
        System.arraycopy(tokens, 0, all, prompt.length, tokens.length);
        float[] promptMel = flat("prompt_mel");                            // [80, 65]
        int melFrames = promptMel.length / 80;
        int frames = 2 * prompt.length;
        float[] feat = new float[frames * 80];
        for (int f = 0; f < frames; f++) {
            for (int m = 0; m < 80; m++) {
                feat[f * 80 + m] = promptMel[m * melFrames + f];
            }
        }
        Var.noGrad(() -> {
            Var mu = flow.encode(all);
            close(flat("flow_mu"), mu.value().toArray(), 2e-3, "mu");
            Var mel = flow.decode(mu, floats("xvector"), feat, frames, flat("flow_noise"), 10);
            close(flat("flow_mel"), mel.value().toArray(), 5e-3, "mel");
            return null;
        });
    }

    @Test
    void hiftVocodesAsTheReference() {
        HiftVocoder hift = tts.hift();
        float[] melData = flat("flow_mel");
        Var mel = Var.of(io.github.zzuegg.brainiac.tensor.Tensor.of(
                io.github.zzuegg.brainiac.tensor.Shape.of(80, melData.length / 80), melData, TestDevice.get()));
        Var.noGrad(() -> {
            close(floats("f0"), hift.pitch(mel), 1e-3, "f0");
            float[] f0 = floats("f0");
            float[] quiet = hift.source(f0, new float[floats("source_phases").length], null);
            float[] want = floats("source_quiet");
            float[] every7 = new float[want.length];
            for (int i = 0; i < every7.length; i++) {
                every7[i] = quiet[7 * i];
            }
            close(want, every7, 1e-4, "source");
            float[] wave = hift.decode(mel, floats("hift_source"));
            HiftVocoder.fadeIn(wave);
            close(floats("hift_audio"), wave, 2e-3, "audio");
            return null;
        });
    }

    @Test
    void speaksAndClones() {
        Audio speech = tts.synthesize("Hello world.", SpeechOptions.defaults().seed(3), null);
        assertEquals(24_000, speech.sampleRate());
        assertTrue(speech.samples().length > 0, "samples");
        for (float s : speech.samples()) {
            assertFalse(Float.isNaN(s), "NaN in the audio");
        }
        tts.addVoice("buzz", new Audio(floats("voice24"), 24_000));
        assertTrue(tts.voices().stream().anyMatch(v -> v.id().equals("buzz")), "the new voice is listed");
        Audio cloned = tts.synthesize("Hi.", SpeechOptions.defaults().voice("buzz").seed(1), null);
        assertTrue(cloned.samples().length > 0, "cloned samples");
        tts.removeVoice("buzz");
    }
}
