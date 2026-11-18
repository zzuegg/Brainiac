package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.onnx.OnnxModel;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.LinkedHashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** ONNX models exported from PyTorch or built by hand, against onnxruntime (make_onnx.py). */
class OnnxModelTest {

    private static Map<String, Tensor> inputs(WeightSource w, OnnxModel m) {
        Map<String, Tensor> in = new LinkedHashMap<>();
        for (String name : m.inputNames()) {
            in.put(name, w.tensor("in." + name, LlamaTest.DEVICE, DType.FLOAT32));
        }
        return in;
    }

    @ParameterizedTest
    @ValueSource(strings = {"mlp", "cnn", "block", "ops"})
    void matchesOnnxRuntime(String name) {
        OnnxModel m = OnnxModel.load(LlamaTest.resource("onnx/" + name + ".onnx"), LlamaTest.DEVICE, DType.FLOAT32);
        try (WeightSource w = WeightSource.open(LlamaTest.resource("onnx/" + name + ".safetensors"))) {
            Map<String, Tensor> out = m.run(inputs(w, m));
            for (String o : m.outputNames()) {
                float[] want = w.tensor("out." + o, Device.cpu(), DType.FLOAT32).toArray();
                Tensor got = out.get(o);
                assertNotNull(got, o);
                float[] a = got.to(DType.FLOAT32).toArray();
                assertEquals(want.length, a.length, name + " " + o + " size");
                for (int i = 0; i < a.length; i++) {
                    assertEquals(want[i], a[i], 2e-4f * Math.max(1, Math.abs(want[i])), name + " " + o + " [" + i + "]");
                }
            }
        }
    }

    @Test
    void trainsThroughTheGraph() {
        OnnxModel m = OnnxModel.load(LlamaTest.resource("onnx/mlp.onnx"), LlamaTest.DEVICE, DType.FLOAT32);
        assertTrue(m.operators().contains("Gemm"));
        assertEquals(4, m.parameters().size());
        try (WeightSource w = WeightSource.open(LlamaTest.resource("onnx/mlp.safetensors"))) {
            Var x = Var.of(w.tensor("in.x", LlamaTest.DEVICE, DType.FLOAT32));
            Var y = m.forward(Map.of("x", x)).get("y");
            y.slice(1, 0, 1).sum().backward();
            for (Var p : m.parameters().values()) {
                assertNotNull(p.grad(), "every weight gets a gradient");
            }
        }
    }
}
