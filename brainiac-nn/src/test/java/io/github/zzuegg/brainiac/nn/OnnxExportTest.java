package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.onnx.OnnxModel;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Models written as ONNX and read back give the same outputs. */
class OnnxExportTest {

    @TempDir
    Path dir;

    private void roundTrip(Model m, Tensor x, String name) throws IOException {
        SplittableRandom r = new SplittableRandom(3);
        for (Var p : m.parameters()) {
            p.assign(Tensor.uniform(p.shape(), -0.6f, 0.6f, r));
        }
        Path file = dir.resolve(name + ".onnx");
        m.exportOnnx(file);
        float[] want = m.forward(Var.of(x), false).value().toArray();
        OnnxModel onnx = OnnxModel.load(file, Device.cpu(), DType.FLOAT32);
        float[] got = onnx.run(Map.of("input", x)).get("output").toArray();
        assertEquals(want.length, got.length, name);
        for (int i = 0; i < want.length; i++) {
            assertEquals(want[i], got[i], 1e-4f * Math.max(1, Math.abs(want[i])), name + " [" + i + "]");
        }
    }

    private static Tensor random(int... dims) {
        return Tensor.uniform(Shape.of(dims), -1, 1, new SplittableRandom(7));
    }

    @Test
    void denseNetwork() throws IOException {
        Model m = Model.sequential().input(8).dense(16, Activation.GELU).dropout(0.1f).dense(12, Activation.MISH)
                .layerNorm().dense(10, Activation.elu(0.7f)).dense(6, Activation.GELU_TANH)
                .dense(4, Activation.LOG_SOFTMAX).seed(1).build();
        roundTrip(m, random(3, 8), "dense");
    }

    @Test
    void imageNetwork() throws IOException {
        Model m = Model.sequential().input(3, 12, 12).conv2d(8, 3, Activation.RELU).batchNorm()
                .maxPool(2, 2).conv2d(6, 3, Activation.leakyRelu(0.2f)).groupNorm(2).upsample(2)
                .avgPool(2, 2).globalAvgPool().dense(5, Activation.SOFTMAX).seed(2).build();
        roundTrip(m, random(2, 3, 12, 12), "image");
    }

    @Test
    void residualGraph() throws IOException {
        Node in = Node.input(6);
        Node a = in.dense(6, Activation.TANH);
        Node b = in.dense(6, Activation.SIGMOID);
        Node out = a.add(b).multiply(in).concat(a).dense(3, Activation.HARD_SWISH);
        roundTrip(Model.graph(in, out).seed(3).build(), random(4, 6), "graph");
    }

    @Test
    void embeddingsAndShapes() throws IOException {
        Model m = Model.sequential().input(5).embedding(11, 4).reshape(20).dense(3).seed(4).build();
        Tensor ids = Tensor.of(Shape.of(2, 5), 1, 4, 7, 0, 10, 3, 3, 9, 2, 5);
        roundTrip(m, ids, "embedding");
    }

    @Test
    void refusesLayersWithoutAnOnnxForm() {
        Model m = Model.sequential().input(4, 3).lstm(5, false).seed(1).build();
        assertThrows(UnsupportedOperationException.class, () -> m.exportOnnx(dir.resolve("lstm.onnx")));
    }
}
