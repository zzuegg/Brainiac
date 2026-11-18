package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.tensor.DType;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/** VITS against transformers' VitsModel with the noise off (see make_vits.py). */
class VitsTest {

    @ParameterizedTest
    @ValueSource(strings = {"vits", "vits-deterministic", "vits-speakers"})
    void waveformMatchesTransformers(String name) throws Exception {
        Path dir = WhisperTest.resource(name);
        Map<String, Object> expected = Json.object(Json.read(dir.resolve("expected.json")));
        VitsTokenizer tok = VitsTokenizer.load(dir);
        int[] ids = tok.encode((String) expected.get("text"));
        int[] want = Json.array(expected.get("ids")).stream().mapToInt(o -> ((Number) o).intValue()).toArray();
        assertArrayEquals(want, ids);
        Vits model = Vits.load(dir, TestDevice.get(), DType.FLOAT32);
        int speaker = model.config().speakers() > 1 ? 1 : 0;
        float[] wave = model.synthesize(ids, new Vits.Sampling(0, 0, 1, speaker, 0));
        float[] ref = WhisperTest.floats(expected.get("waveform"));
        assertEquals(ref.length, wave.length, "length");
        for (int i = 0; i < ref.length; i++) {
            assertEquals(ref[i], wave[i], 2e-3, "sample " + i);
        }
        // vocoded in overlapping pieces, the same waveform
        model.vocoderChunk = 30;
        model.vocoderContext = 16;
        float[] pieces = model.synthesize(ids, new Vits.Sampling(0, 0, 1, speaker, 0));
        for (int i = 0; i < ref.length; i++) {
            assertEquals(ref[i], pieces[i], 2e-3, "pieces, sample " + i);
        }
        float[] fast = model.synthesize(ids, new Vits.Sampling(0, 0, 1.5f, speaker, 0));
        assertEquals(((Number) expected.get("fast_length")).intValue(), fast.length);
        model.close();
    }
}
