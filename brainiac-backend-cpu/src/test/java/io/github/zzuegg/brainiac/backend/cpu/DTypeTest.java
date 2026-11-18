package io.github.zzuegg.brainiac.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.ConvGeometry;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

class DTypeTest {

    private static final Device CPU = Device.getDefault();

    private static Tensor random(long seed, int... dims) {
        return Tensor.uniform(Shape.of(dims), -1, 1, new SplittableRandom(seed));
    }

    private static double tolerance(DType type) {
        return switch (type) {
            case FLOAT64 -> 1e-5;
            case FLOAT32 -> 1e-6;
            case FLOAT16 -> 2e-2;
            case BFLOAT16 -> 8e-2;
            default -> 0;
        };
    }

    /** Runs an operation in float32 and in another type and compares, relative to the values' size. */
    private static void same(DType type, String what, UnaryOperator<Tensor> op, Tensor input) {
        Tensor expected = op.apply(input);
        Tensor actual = op.apply(input.to(type));
        assertEquals(expected.shape(), actual.shape(), what);
        float[] e = expected.toArray();
        float[] a = actual.toArray();
        double scale = 1;
        for (float v : e) {
            scale = Math.max(scale, Math.abs(v));
        }
        for (int i = 0; i < e.length; i++) {
            assertEquals(e[i], a[i], tolerance(type) * scale, what + " at " + i);
        }
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT64", "FLOAT16", "BFLOAT16"})
    void everyOperationMatchesFloat32(DType type) {
        Tensor x = random(1, 3, 4);
        Tensor y = random(2, 3, 4);
        Tensor yt = y.to(type);
        same(type, "add", t -> t.add(yt), x);
        same(type, "broadcast row", t -> t.mul(yt.slice(0, 0, 1)), x);
        same(type, "scalar", t -> t.mul(3).add(1).div(2).sub(0.5f), x);
        same(type, "pow", t -> t.abs().pow(1.5f), x);
        same(type, "max/min/greater", t -> t.maximum(yt).add(t.minimum(yt)).add(t.greater(yt)), x);
        same(type, "unary", t -> t.exp().add(t.tanh()).add(t.sigmoid()).add(t.square().add(1).log())
                .add(t.erf()).add(t.softplus()).add(t.abs().sqrt()).add(t.relu()).add(t.sign())
                .add(t.mul(4).round()), x);
        for (ActivationOp op : ActivationOp.values()) {
            same(type, op.name(), t -> t.activation(op, 0.1f), x);
            same(type, op + " backward", t -> t.activationBackward(op, 0.1f, yt), x);
        }
        same(type, "matmul", t -> t.matmul(yt.transpose()), x);
        same(type, "batched matmul", t -> t.reshape(2, 3, 2).matmul(yt.reshape(2, 2, 3)), x);
        same(type, "reductions", t -> Tensor.concat(0, t.sum(1, true), t.mean(1, true), t.max(1, true),
                t.min(1, true)), x);
        same(type, "sum all", Tensor::sum, x);
        same(type, "reduce axis 0", t -> t.sum(0), x);
        same(type, "argmax", t -> t.argmax(1), x);
        same(type, "permute and concat", t -> Tensor.concat(1, t.transpose(), t.transpose()), x);
        Tensor idx = Tensor.of(2, 0, 2);
        same(type, "gather", t -> t.gather(1, idx), x);
        same(type, "scatterAdd", t -> t.scatterAdd(1, Tensor.of(1, 1, 0, 3), 5), x);
        ConvGeometry g = ConvGeometry.sliding(1, 2, new int[] {6}, new int[] {3}, new int[] {1}, new int[] {1},
                new int[] {1}, new int[] {1}, false);
        Tensor seq = random(3, 1, 2, 6);
        same(type, "im2col", t -> t.im2col(g, 0), seq);
        same(type, "col2im", t -> t.im2col(g, 0).col2im(g), seq);
    }

    @Test
    void float64KeepsWhatFloat32Loses() {
        Tensor one = Tensor.zeros(Shape.of(1), DType.FLOAT64, CPU).add(1);
        Tensor tiny = Tensor.of(Shape.of(1), new double[] {1e-12}, DType.FLOAT64, CPU);
        assertEquals(1 + 1e-12, one.add(tiny).toDoubleArray()[0], 0);
        assertEquals(1.0, one.to(DType.FLOAT32).add(tiny.to(DType.FLOAT32)).toDoubleArray()[0], 0);
        double[] precise = {Math.PI, Math.E};
        assertEquals(Math.PI, Tensor.of(Shape.of(2), precise, DType.FLOAT64, CPU).toDoubleArray()[0], 0);
    }

    @Test
    void halfTypesRoundAndOverflowLikeTheirFormats() {
        Tensor x = Tensor.of(0.1f, 70000f, 1e30f);
        double[] half = x.to(DType.FLOAT16).toDoubleArray();
        assertEquals(Float.float16ToFloat(Float.floatToFloat16(0.1f)), half[0], 0);
        assertEquals(Double.POSITIVE_INFINITY, half[1]);
        double[] brain = x.to(DType.BFLOAT16).toDoubleArray();
        assertEquals(0.10009765625, brain[0], 0);   // 8 significant bits
        assertEquals(1e30, brain[2], 1e28);         // float32's range
        assertEquals(DType.FLOAT16, x.to(DType.FLOAT16).add(x.to(DType.FLOAT16)).dtype());
    }

    @Test
    void typesPromoteAndScalarsAdapt() {
        Tensor h = Tensor.zeros(Shape.of(3), DType.FLOAT16, CPU);
        Tensor b = Tensor.zeros(Shape.of(3), DType.BFLOAT16, CPU);
        Tensor d = Tensor.zeros(Shape.of(3), DType.FLOAT64, CPU);
        assertEquals(DType.FLOAT32, h.add(b).dtype());
        assertEquals(DType.FLOAT64, h.add(d).dtype());
        assertEquals(DType.FLOAT16, h.add(Tensor.scalar(2)).dtype());
        assertEquals(DType.FLOAT64, d.mul(2).dtype());
        assertEquals(DType.FLOAT32, d.argmax(0).dtype());
        assertEquals(DType.FLOAT64, h.reshape(1, 3).matmul(d.reshape(3, 1)).dtype());
        assertEquals(DType.FLOAT64, Tensor.concat(0, h, d).dtype());
        assertSame(h, h.to(DType.FLOAT16));
        for (DType t : List.of(DType.FLOAT64, DType.FLOAT16, DType.BFLOAT16)) {
            Tensor full = Tensor.full(Shape.of(2), 1.5f, t, CPU);
            assertEquals(t, full.dtype());
            assertEquals(1.5, full.toDoubleArray()[1], 0);
            assertTrue(full.toString() != null);
        }
    }
}
