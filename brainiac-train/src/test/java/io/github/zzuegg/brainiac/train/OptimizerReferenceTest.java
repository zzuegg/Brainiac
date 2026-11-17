package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** The optimizers against PyTorch and transformers, fed the same gradients (make_optimizers.py). */
class OptimizerReferenceTest {

    static Map<String, Object> ref;

    @BeforeAll
    static void read() throws IOException {
        try (InputStream in = OptimizerReferenceTest.class.getResourceAsStream("optimizers.json")) {
            ref = Json.object(Json.parse(new String(in.readAllBytes(), StandardCharsets.UTF_8)));
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

    /** Runs five steps; returns W and B after them. */
    static List<Var> run(Optimizer opt, boolean matrixOnly) {
        Var w = Var.param(Tensor.of(Shape.of(4, 6), floats(ref.get("W0"))));
        Var b = Var.param(Tensor.of(Shape.of(6), floats(ref.get("B0"))));
        List<Var> params = matrixOnly ? List.of(w) : List.of(w, b);
        for (Object step : Json.array(ref.get("grads"))) {
            List<Object> g = Json.array(step);
            w.setGrad(Tensor.of(Shape.of(4, 6), floats(g.get(0))));
            b.setGrad(Tensor.of(Shape.of(6), floats(g.get(1))));
            opt.step(params);
        }
        return params;
    }

    static void check(String name, Supplier<Optimizer> make, boolean matrixOnly, float tolerance) {
        List<Var> params = run(make.get(), matrixOnly);
        List<Object> want = Json.array(ref.get(name));
        for (int i = 0; i < params.size(); i++) {
            assertArrayEquals(floats(want.get(i)), params.get(i).value().toArray(), tolerance, name + " param " + i);
        }
    }

    @Test
    void sgdWithMomentum() {
        check("sgd_momentum", () -> Optimizer.momentum(0.1f, 0.9f), false, 1e-5f);
    }

    @Test
    void adam() {
        check("adam", () -> Optimizer.adam(0.01f), false, 1e-5f);
    }

    @Test
    void adamW() {
        check("adamw", () -> Optimizer.adamW(0.01f, 0.1f), false, 1e-5f);
    }

    @Test
    void lion() {
        check("lion", () -> Optimizer.lion(0.01f, 0.1f), false, 1e-5f);
    }

    @Test
    void adafactor() {
        check("adafactor", () -> Optimizer.adafactor(0.01f), false, 1e-5f);
        check("adafactor_relative", () -> Optimizer.adafactor(1, true, true, 0.01f), false, 1e-5f);
    }

    @Test
    void muon() {
        check("muon", () -> Optimizer.muon(0.02f), true, 1e-5f);
        Var v = Var.param(Tensor.of(Shape.of(3), 1, 2, 3));
        v.setGrad(Tensor.of(Shape.of(3), 1, 1, 1));
        assertThrows(IllegalArgumentException.class, () -> Optimizer.muon(0.02f).step(List.of(v)));
    }

    @Test
    void muonTakesAnOrthogonalizerOfOnesOwn() {
        java.util.function.UnaryOperator<Tensor> polar = g -> {
            Tensor[] usv = io.github.zzuegg.brainiac.tensor.Linalg.svd(g);
            return usv[0].matmul(usv[2]);
        };
        Tensor g = Tensor.of(Shape.of(3, 2), 1f, 2f, -1f, 0.5f, 3f, -2f);
        Var p = Var.param(Tensor.zeros(Shape.of(3, 2)));
        p.setGrad(g);
        Optimizer.muon(0.1f, 0f, 0.9f, false, polar).step(List.of(p));
        // the direction is the exact polar factor, scaled by sqrt(max(1, rows / cols))
        assertArrayEquals(polar.apply(g).mul((float) (-0.1 * Math.sqrt(1.5))).toArray(), p.value().toArray(), 1e-5f);
        // Newton-Schulz approximates it only roughly
        float[] ns = Muon.orthogonalize(g, 5).toArray();
        float[] exact = polar.apply(g).toArray();
        for (int i = 0; i < ns.length; i++) {
            assertEquals(exact[i], ns[i], 0.35f);
        }
    }

    @Test
    void groupsSendEachParameterToItsOptimizer() {
        // Muon for the matrix, AdamW for the vector: the same as running them apart
        Optimizer groups = Optimizer.groups(Optimizer.adamW(0.01f, 0.1f), Optimizer.Group.matrices(Optimizer.muon(0.02f)));
        List<Var> params = run(groups, false);
        assertArrayEquals(floats(Json.array(ref.get("muon")).get(0)), params.get(0).value().toArray(), 1e-5f);
        assertArrayEquals(floats(Json.array(ref.get("adamw")).get(1)), params.get(1).value().toArray(), 1e-5f);
        groups.learningRate(0.005f);
        assertEquals(0.005f, groups.learningRate());
    }

    @Test
    void savedStateContinuesExactly() {
        for (Supplier<Optimizer> make : List.<Supplier<Optimizer>>of(() -> Optimizer.adamW(0.01f, 0.1f),
                () -> Optimizer.lion(0.01f, 0.1f), () -> Optimizer.adafactor(0.01f),
                () -> Optimizer.momentum(0.1f, 0.9f),
                () -> Optimizer.groups(Optimizer.adam(0.01f), Optimizer.Group.matrices(Optimizer.muon(0.02f))))) {
            String name = make.get().toString();
            List<Object> grads = Json.array(ref.get("grads"));
            // three steps, save, a fresh optimizer restores and takes the last two
            Optimizer first = make.get();
            Var w = Var.param(Tensor.of(Shape.of(4, 6), floats(ref.get("W0"))));
            Var b = Var.param(Tensor.of(Shape.of(6), floats(ref.get("B0"))));
            List<Var> params = List.of(w, b);
            Optimizer second = null;
            for (int t = 0; t < grads.size(); t++) {
                if (t == 3) {
                    Map<String, Tensor> saved = first.state(params);
                    second = make.get();
                    second.loadState(params, saved);
                }
                List<Object> g = Json.array(grads.get(t));
                w.setGrad(Tensor.of(Shape.of(4, 6), floats(g.get(0))));
                b.setGrad(Tensor.of(Shape.of(6), floats(g.get(1))));
                (t < 3 ? first : second).step(params);
            }
            List<Var> straight = run(make.get(), false);
            assertArrayEquals(straight.get(0).value().toArray(), w.value().toArray(), 1e-6f, name);
            assertArrayEquals(straight.get(1).value().toArray(), b.value().toArray(), 1e-6f, name);
        }
    }

    @Test
    void emaAveragesAndSwaps() {
        Var w = Var.param(Tensor.of(Shape.of(2), 0, 0));
        Ema ema = new Ema(List.of(w), 0.5f);
        w.assign(Tensor.of(Shape.of(2), 2, 4));
        ema.update();   // 0.5 * 0 + 0.5 * x
        w.assign(Tensor.of(Shape.of(2), 4, 8));
        ema.update();   // 0.5 * (1, 2) + 0.5 * (4, 8)
        ema.swap();
        assertArrayEquals(new float[] {2.5f, 5}, w.value().toArray());
        ema.swap();
        assertArrayEquals(new float[] {4, 8}, w.value().toArray());
        Ema warm = new Ema(List.of(w), 0.999f, true);
        assertEquals(0.1f, warm.currentDecay(), 1e-6f);
        Ema copy = new Ema(List.of(w), 0.5f);
        copy.loadState(ema.state());
        copy.swap();
        assertArrayEquals(new float[] {2.5f, 5}, w.value().toArray());
        assertEquals(2, copy.updates());
    }
}
