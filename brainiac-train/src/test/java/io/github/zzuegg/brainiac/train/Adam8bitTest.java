package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Supplier;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class Adam8bitTest {

    static final Device DEVICE = TestDevice.get();
    // a two-layer regression: w1 [32, 256] and w2 [256, 16] get 8-bit state, the biases float32
    static final Tensor X = Tensor.uniform(Shape.of(64, 32), -1, 1, new SplittableRandom(1)).to(DEVICE);
    static final Tensor Y = Tensor.uniform(Shape.of(64, 16), -1, 1, new SplittableRandom(2)).to(DEVICE);

    static Map<String, Var> model() {
        SplittableRandom r = new SplittableRandom(3);
        Map<String, Var> m = new LinkedHashMap<>();
        m.put("w1", Var.param(Tensor.uniform(Shape.of(32, 256), -0.2f, 0.2f, r).to(DEVICE)));
        m.put("b1", Var.param(Tensor.zeros(Shape.of(256)).to(DEVICE)));
        m.put("w2", Var.param(Tensor.uniform(Shape.of(256, 16), -0.1f, 0.1f, r).to(DEVICE)));
        m.put("b2", Var.param(Tensor.zeros(Shape.of(16)).to(DEVICE)));
        return m;
    }

    static Supplier<Var> loss(Map<String, Var> m) {
        return () -> Var.of(X).matmul(m.get("w1")).add(m.get("b1")).tanh().matmul(m.get("w2")).add(m.get("b2"))
                .sub(Var.of(Y)).square().mean();
    }

    static float[] train(Map<String, Var> m, TrainingLoop loop, int steps) {
        float[] losses = new float[steps];
        for (int i = 0; i < steps; i++) {
            losses[i] = loop.microBatch(loss(m));
        }
        return losses;
    }

    @Test
    void trainsLikeAdamWithAQuarterOfTheState() {
        Map<String, Var> exact = model();
        Optimizer adam = Optimizer.adamW(3e-3f, 0.9f, 0.999f, 1e-8f, 0.01f);
        float[] a = train(exact, TrainingLoop.of(exact, adam), 300);
        Map<String, Var> small = model();
        Optimizer adam8 = Optimizer.adam8bit(3e-3f, 0.9f, 0.999f, 1e-8f, 0.01f);
        float[] b = train(small, TrainingLoop.of(small, adam8), 300);
        assertTrue(a[299] < 0.2f * a[0], "adam: " + a[0] + " -> " + a[299]);
        for (int i : new int[] {10, 50, 100, 299}) {
            // the same path, to within a few percent of where it started
            assertEquals(a[i], b[i], 0.02f * a[0] + 0.1f * a[i], "step " + i + ": adam " + a[i] + ", 8-bit " + b[i]);
        }

        var full = adam.state(java.util.List.copyOf(exact.values()));
        var packed = adam8.state(java.util.List.copyOf(small.values()));
        Tensor m = packed.get("0.m");
        assertTrue(m.isPacked(), "w1's state is packed");
        assertEquals(DEVICE, m.device());
        assertTrue(m.bytes() * 3 < full.get("0.m").bytes(), m.bytes() + " bytes");
        assertFalse(packed.get("1.m").isPacked(), "a bias keeps float32 state");
    }

    @Test
    void resumesExactlyWhereItStopped(@TempDir Path dir) {
        Map<String, Var> straight = model();
        TrainingLoop a = TrainingLoop.of(straight, Optimizer.adam8bit(3e-3f));
        train(straight, a, 20);

        Map<String, Var> stopped = model();
        TrainingLoop b = TrainingLoop.of(stopped, Optimizer.adam8bit(3e-3f));
        train(stopped, b, 10);
        b.save(dir);
        Map<String, Var> resumed = model();
        TrainingLoop c = TrainingLoop.of(resumed, Optimizer.adam8bit(3e-3f));
        c.resume(dir);
        train(resumed, c, 10);
        for (String k : straight.keySet()) {
            assertArrayEquals(straight.get(k).value().toArray(), resumed.get(k).value().toArray(), 1e-6f, k);
        }
    }

    @Test
    void offloadedAdamMatchesAdamOnTheDevice(@TempDir Path dir) {
        Map<String, Var> here = model();
        float[] a = train(here, TrainingLoop.of(here, Optimizer.adamW(3e-3f, 0.01f)), 20);
        Map<String, Var> away = model();
        TrainingLoop loop = TrainingLoop.of(away, Optimizer.offloaded(Optimizer.adamW(3e-3f, 0.01f)));
        float[] b = train(away, loop, 10);
        loop.save(dir);
        Map<String, Var> resumed = model();
        TrainingLoop again = TrainingLoop.of(resumed, Optimizer.offloaded(Optimizer.adamW(3e-3f, 0.01f)));
        again.resume(dir);
        train(resumed, again, 10);
        assertEquals(a[9], b[9], 1e-5f * a[0] + 1e-6f);
        for (String k : here.keySet()) {
            assertEquals(DEVICE, resumed.get(k).value().device());
            assertArrayEquals(here.get(k).value().toArray(), resumed.get(k).value().toArray(), 2e-5f, k);
        }
    }

    @Test
    void choosesBlocksThatFit() {
        assertEquals(0, Adam8bit.blockRows(4095));
        assertEquals(256, Adam8bit.blockRows(32 * 256));
        assertEquals(64, Adam8bit.blockRows(4096 + 256));
        assertEquals(0, Adam8bit.blockRows(4099));
    }
}
