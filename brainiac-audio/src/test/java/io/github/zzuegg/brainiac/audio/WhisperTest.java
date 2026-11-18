package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Whisper's front end and network against PyTorch and transformers (see make_whisper.py). */
class WhisperTest {

    static Path resource(String name) {
        try {
            return Path.of(WhisperTest.class.getResource(name).toURI());
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    static float[] floats(Object list) {
        List<Object> l = Json.array(list);
        float[] out = new float[l.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ((Number) l.get(i)).floatValue();
        }
        return out;
    }

    static float[][] matrix(Object list) {
        List<Object> l = Json.array(list);
        float[][] out = new float[l.size()][];
        for (int i = 0; i < out.length; i++) {
            out[i] = floats(l.get(i));
        }
        return out;
    }

    @Test
    void logMelMatchesWhisper() throws Exception {
        Map<String, Object> ref = Json.object(Json.read(resource("mel.json")));
        float[] audio = floats(ref.get("audio"));
        float[][] want = matrix(ref.get("mel80"));
        float[][] got = WhisperFeatures.logMel(audio, 80);
        assertEquals(want.length, got.length);
        assertEquals(want[0].length, got[0].length);
        double worst = 0;
        for (int m = 0; m < want.length; m++) {
            for (int t = 0; t < want[m].length; t++) {
                worst = Math.max(worst, Math.abs(want[m][t] - got[m][t]));
            }
        }
        assertTrue(worst < 1e-3, "worst difference " + worst);
        float[][] want128 = matrix(ref.get("mel128_first"));
        float[][] got128 = WhisperFeatures.logMel(audio, 128);
        for (int m = 0; m < 128; m++) {
            for (int t = 0; t < want128[m].length; t++) {
                assertEquals(want128[m][t], got128[m][t], 1e-3, "mel " + m + " frame " + t);
            }
        }
    }

    @Test
    void encoderAndDecoderMatchTransformers() throws Exception {
        Device device = TestDevice.get();
        Map<String, Object> ref = Json.object(Json.read(resource("mel.json")));
        float[][] mel = matrix(ref.get("mel80"));
        float[][] window = new float[80][80];
        for (int m = 0; m < 80; m++) {
            System.arraycopy(mel[m], 0, window[m], 0, 80);
        }
        Map<String, Object> expected = Json.object(Json.read(resource("whisper/expected.json")));
        Whisper model = Whisper.load(resource("whisper"), device, DType.FLOAT32);
        Tensor audio = model.encode(window);
        float[][] enc = matrix(expected.get("encoder"));
        float[] got = audio.to(Device.cpu()).toArray();
        int width = enc[0].length;
        for (int p = 0; p < enc.length; p++) {
            for (int j = 0; j < width; j++) {
                assertEquals(enc[p][j], got[p * width + j], 2e-3, "encoder position " + p + " feature " + j);
            }
        }
        int[] ids = Json.array(expected.get("ids")).stream().mapToInt(o -> ((Number) o).intValue()).toArray();
        float[][] logits = matrix(expected.get("logits"));
        try (Whisper.Decoder d = model.decoder(audio)) {
            float[][] all = d.nextAll(ids);
            for (int s = 0; s < ids.length; s++) {
                for (int v = 0; v < logits[s].length; v++) {
                    assertEquals(logits[s][v], all[s][v], 3e-3, "step " + s + " token " + v);
                }
            }
            // the same, a token at a time through the cache
            d.reset();
            for (int s = 0; s < ids.length; s++) {
                float[] step = d.next(ids[s]);
                for (int v = 0; v < logits[s].length; v++) {
                    assertEquals(logits[s][v], step[v], 3e-3, "cached step " + s + " token " + v);
                }
            }
        }
        model.close();
    }
}
