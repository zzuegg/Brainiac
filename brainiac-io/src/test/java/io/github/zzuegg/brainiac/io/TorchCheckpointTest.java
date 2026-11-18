package io.github.zzuegg.brainiac.io;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import org.junit.jupiter.api.Test;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Checkpoints written by torch.save (see torch/make_checkpoints.py). */
class TorchCheckpointTest {

    private static Path resource(String name) throws Exception {
        return Path.of(TorchCheckpointTest.class.getResource("torch/" + name).toURI());
    }

    @Test
    void readsEveryStorageTypeLayoutAndNesting() throws Exception {
        Map<String, Object> ref = Json.object(Json.read(resource("expected.json")));
        List<Object> rows = Json.array(ref.get("a"));
        float[] a = new float[12];
        for (int i = 0; i < 3; i++) {
            List<Object> r = Json.array(rows.get(i));
            for (int j = 0; j < 4; j++) {
                a[i * 4 + j] = ((Number) r.get(j)).floatValue();
            }
        }
        try (WeightSource w = WeightSource.open(resource("nested.pth"))) {
            assertTrue(w.names().containsAll(List.of("model.layer.weight", "model.layer.half", "model.steps",
                    "list.0")), w.names().toString());
            assertEquals(Shape.of(3, 4), w.shape("model.layer.weight"));
            assertArrayEquals(a, w.tensor("model.layer.weight", Device.cpu(), DType.FLOAT32).toArray());
            assertArrayEquals(a, w.tensor("model.layer.double", Device.cpu(), DType.FLOAT32).toArray(), 1e-6f);
            assertArrayEquals(a, w.tensor("model.layer.half", Device.cpu(), DType.FLOAT32).toArray(), 2e-3f);
            assertArrayEquals(a, w.tensor("model.layer.bf16", Device.cpu(), DType.FLOAT32).toArray(), 2e-2f);
            assertEquals(DType.FLOAT16, w.naturalType("model.layer.half"));
            float[] t = w.tensor("model.layer.transposed", Device.cpu(), DType.FLOAT32).toArray();
            assertEquals(Shape.of(4, 3), w.shape("model.layer.transposed"));
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 3; j++) {
                    assertEquals(a[j * 4 + i], t[i * 3 + j]);
                }
            }
            assertArrayEquals(new float[] {a[6], a[7], a[10], a[11]},
                    w.tensor("model.layer.slice", Device.cpu(), DType.FLOAT32).toArray());
            assertArrayEquals(new float[] {5, -7}, w.tensor("model.steps", Device.cpu(), DType.FLOAT32).toArray());
            assertArrayEquals(new float[] {1, 1}, w.tensor("list.0", Device.cpu(), DType.FLOAT32).toArray());
        }
        try (WeightSource w = WeightSource.open(resource("bare.pt"))) {
            assertEquals(List.of(""), w.names());
            assertArrayEquals(new float[] {a[0], a[1], a[2], a[3]}, w.tensor("", Device.cpu(), DType.FLOAT32).toArray());
        }
    }
}
