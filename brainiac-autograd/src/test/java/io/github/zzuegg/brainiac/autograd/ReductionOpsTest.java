package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

import org.junit.jupiter.api.Test;

class ReductionOpsTest {

    static final Tensor A = EinsumTest.A;

    @Test
    void matchPyTorch() {
        // values from torch.logsumexp, Tensor.var and Tensor.std
        assertArrayEquals(new float[] {1.61299f, 2.11424f, 0.85655f, 0.51265f, 1.73290f, 1.90729f},
                A.logsumexp(-1, false).toArray(), 1e-4f);
        Tensor l = A.logsumexp(1, true);
        assertArrayEquals(new int[] {2, 1, 4}, l.shape().toArray());
        assertArrayEquals(new float[] {0.79479f, 0.95350f, 1.81104f, 1.55404f, 0.44288f, 1.69599f, 0.83801f,
            1.55616f}, l.toArray(), 1e-4f);
        assertArrayEquals(new float[] {0.19256f, 0.69581f, 0.95823f, 0.25565f, 1.36110f, 0.89179f, 0.70774f,
            0.80053f}, A.variance(1, 1, false).toArray(), 1e-4f);
        assertArrayEquals(new float[] {0.28169f, 0.69590f, 0.46334f, 0.77939f, 0.62548f, 0.74754f},
                A.std(-1, 0, false).toArray(), 1e-4f);
        assertArrayEquals(A.logsumexp(-1, false).toArray(), Var.of(A).logsumexp(-1, false).value().toArray(), 0f);
    }

    @Test
    void logsumexpStaysFiniteForLargeValues() {
        Tensor big = Tensor.of(1000f, 1000f);
        assertArrayEquals(new float[] {1000f + (float) Math.log(2)}, big.logsumexp(0, false).toArray(), 1e-3f);
    }

    @Test
    void anyAndAll() {
        Tensor t = Tensor.of(Shape.of(3, 2), 0f, 0f, 0f, -2f, 1f, 3f);
        assertArrayEquals(new float[] {0f, 1f, 1f}, t.any(1, false).toArray(), 0f);
        assertArrayEquals(new float[] {0f, 0f, 1f}, t.all(1, false).toArray(), 0f);
        assertArrayEquals(new float[] {1f, 1f}, t.any(0, false).toArray(), 0f);
        assertArrayEquals(new float[] {0f, 0f}, t.all(0, false).toArray(), 0f);
    }

    @Test
    void repeatInterleave() {
        Tensor t = Tensor.of(Shape.of(2, 2), 1f, 2f, 3f, 4f);
        assertArrayEquals(new float[] {1f, 2f, 1f, 2f, 3f, 4f, 3f, 4f}, t.repeatInterleave(2, 0).toArray(), 0f);
        Tensor c = t.repeatInterleave(3, -1);
        assertArrayEquals(new int[] {2, 6}, c.shape().toArray());
        assertArrayEquals(new float[] {1f, 1f, 1f, 2f, 2f, 2f, 3f, 3f, 3f, 4f, 4f, 4f}, c.toArray(), 0f);
    }

    @Test
    void haveGradients() {
        Tensor b = EinsumTest.B;
        GradCheck.Result r = GradCheck.check(v -> v.get(0).logsumexp(-1, false).square().sum()
                .add(v.get(1).variance(1, 1, false).sum())
                .add(v.get(1).std(-1, 0, true).sum())
                .add(v.get(0).repeatInterleave(2, 1).square().sum()), List.of(A, b));
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void erfMatchesPyTorch() {
        Tensor x = Tensor.of(-3f, -1.2f, -0.5f, 0f, 0.3f, 1f, 2.5f);
        assertArrayEquals(new float[] {-0.999978f, -0.910314f, -0.5205f, 0f, 0.328627f, 0.842701f, 0.999593f},
                x.erf().toArray(), 1e-6f);
        GradCheck.Result r = GradCheck.check(v -> v.get(0).erf().square().sum(), List.of(A));
        assertTrue(r.passed(), r.toString());
    }
}
