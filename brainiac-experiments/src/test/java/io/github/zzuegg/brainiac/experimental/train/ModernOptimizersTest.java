package io.github.zzuegg.brainiac.experimental.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Supplier;

import org.junit.jupiter.api.Test;

class ModernOptimizersTest {

    static final Tensor[] DATA = Fixtures.spiral(60, 1);

    @Test
    void everyNewOptimizerTrainsTheSpiral() {
        Map<String, Supplier<Optimizer>> zoo = new LinkedHashMap<>();
        zoo.put("c-adamw", () -> Cautious.adamW(0.02f, 0));
        zoo.put("c-lion", () -> Cautious.lion(3e-3f, 0));
        zoo.put("mars-adamw", () -> Mars.adamW(0.02f, 0));
        zoo.put("mars-lion", () -> Mars.lion(3e-3f, 0));
        zoo.put("adopt", () -> Adopt.of(0.02f, 0.9999f));
        zoo.put("adam-mini", () -> AdamMini.of(0.02f, 0));
        zoo.put("stable-adamw", () -> StableAdamW.stable(0.02f, 0));
        zoo.put("adam-atan2", () -> StableAdamW.atan2(0.02f, 0));
        zoo.put("psgd-kron", () -> PsgdKron.of(0.02f));
        zoo.put("scion", () -> new Scion(0.05f, 0.1f, false));
        zoo.put("grokfast(adam)", () -> Grokfast.ema(0.98f, 2f).wrap(Optimizer.adam(0.02f)));
        zoo.put("grokfast-ma(adam)", () -> Grokfast.movingAverage(10, 2f).wrap(Optimizer.adam(0.02f)));
        zoo.put("spike-guard(adam)", () -> SpikeGuard.of(Optimizer.adam(0.02f), 5f));
        zoo.put("muon-clip(adam)", () -> new MuonClip(Optimizer.adam(0.02f), 10));
        zoo.put("bf16-sr(adam)", () -> StochasticRounding.bf16Weights(Optimizer.adam(0.02f), true, 1));
        float initial = Fixtures.loss(SimpleMlp.tanh(3, 2, 32, 32, 3), DATA);
        for (var e : zoo.entrySet()) {
            SimpleMlp net = SimpleMlp.tanh(3, 2, 32, 32, 3);
            float loss = Fixtures.train(net, e.getValue().get(), DATA, 200);
            double acc = LearningMath.accuracy(net.predict(DATA[0]), DATA[1]);
            System.out.printf("%-20s loss %.3f -> %.3f  acc %.3f%n", e.getKey(), initial, loss, acc);
            assertTrue(loss < 0.6 * initial && acc > 0.6, e.getKey() + " trains: " + loss + ", " + acc);
        }
    }

    @Test
    void checkpointStateRestoresIdenticalTrajectories() {
        for (Supplier<SlotOptimizer> make : List.<Supplier<SlotOptimizer>>of(() -> Mars.adamW(0.02f, 0.01f),
                () -> Adopt.of(0.02f, 0.999f), () -> AdamMini.of(0.02f, 0.01f), () -> StableAdamW.stable(0.02f, 0),
                () -> new Scion(0.05f, 0.1f, true))) {
            SimpleMlp a = SimpleMlp.tanh(4, 2, 8, 3);
            SlotOptimizer oa = make.get();
            Fixtures.train(a, oa, DATA, 12);
            SimpleMlp b = SimpleMlp.tanh(99, 2, 8, 3);
            LearningMath.assign(b.parameters(), LearningMath.values(a.parameters()));
            SlotOptimizer ob = make.get();
            ob.loadState(b.parameters(), oa.state(a.parameters()));
            Fixtures.train(a, oa, DATA, 5);
            Fixtures.train(b, ob, DATA, 5);
            assertArrayEquals(LearningMath.flatten(LearningMath.values(a.parameters())),
                    LearningMath.flatten(LearningMath.values(b.parameters())), 1e-5f, oa.toString());
        }
    }

    @Test
    void atan2MatchesTheMathLibraryWithoutEpsilon() {
        float[] ys = {0, 1e-12f, -3, 0.5f, 2, -1e-3f, 7};
        float[] xs = {0, 0, 1, 1e-9f, 2, 4, 0.01f};
        float[] got = StableAdamW.atan2(Tensor.of(ys), Tensor.of(xs)).toArray();
        for (int i = 0; i < ys.length; i++) {
            assertEquals(Math.atan2(ys[i], xs[i]), got[i], 3e-5, "atan2(" + ys[i] + ", " + xs[i] + ")");
        }
    }

    @Test
    void cautiousMaskKeepsOnlyDescentComponentsAndTheMeanStep() {
        Tensor update = Tensor.of(-1, 1, -1, 1);
        Tensor grad = Tensor.of(1, 1, 1, 1);   // components 0 and 2 descend, 1 and 3 ascend
        float[] m = Cautious.mask(update, grad).toArray();
        assertEquals(0, m[1], 1e-6);
        assertEquals(0, m[3], 1e-6);
        assertEquals(m[0], m[2], 1e-6);
        assertEquals(4 / 3.0, m[0], 1e-5);   // n / (kept + 1), the paper's scaling
    }

    @Test
    void grokfastAddsTheAmplifiedSlowComponent() {
        Var w = Var.param(Tensor.of(0, 0));
        Grokfast gf = Grokfast.ema(0.5f, 2f);
        Tensor g = Tensor.of(1, -2);
        assertArrayEquals(new float[] {3, -6}, gf.apply(w, g).toArray(), 1e-6f);   // ema = g at the first step
        float[] second = gf.apply(w, Tensor.of(3, 0)).toArray();                    // ema = (2, -1)
        assertArrayEquals(new float[] {7, -2}, second, 1e-6f);
        Grokfast ma = Grokfast.movingAverage(2, 1f);
        assertArrayEquals(new float[] {1, -2}, ma.apply(w, g).toArray(), 1e-6f);   // window not yet full
        assertArrayEquals(new float[] {5, -1}, ma.apply(w, Tensor.of(3, 0)).toArray(), 1e-6f);
    }

    @Test
    void adamMiniKeepsOneSecondMomentPerOutputNeuron() {
        Var w = Var.param(Tensor.normal(Shape.of(16, 4), 0, 1, new SplittableRandom(1)));
        Var b = Var.param(Tensor.zeros(Shape.of(1, 4)));
        AdamMini mini = AdamMini.of(0.01f, 0);
        w.setGrad(Tensor.ones(Shape.of(16, 4)));
        b.setGrad(Tensor.ones(Shape.of(1, 4)));
        mini.step(List.of(w, b));
        Map<String, Tensor> state = mini.state(List.of(w, b));
        assertEquals(Shape.of(1, 4), state.get("0.v").shape());
        assertEquals(1, state.get("1.v").size());
        AdamMini heads = new AdamMini(0.01f, 0.9f, 0.999f, 1e-8f, 0, 2, p -> false);
        w.setGrad(Tensor.ones(Shape.of(16, 4)));
        heads.step(List.of(w));
        assertEquals(Shape.of(1, 2), heads.state(List.of(w)).get("0.v").shape());
        Optimizer adam = Optimizer.adam(0.01f);
        w.setGrad(Tensor.ones(Shape.of(16, 4)));
        b.setGrad(Tensor.ones(Shape.of(1, 4)));
        adam.step(List.of(w, b));
        assertTrue(AdamMini.stateBytes(mini, List.of(w, b)) < 0.6 * AdamMini.stateBytes(adam, List.of(w, b)));
    }

    @Test
    void spikeGuardSkipsGradientAndLossSpikes() {
        Var w = Var.param(Tensor.of(1, 1));
        SpikeGuard guard = new SpikeGuard(Optimizer.sgd(0.1f), 5, 3, 0, 0, 4, 0);
        for (int i = 0; i < 10; i++) {
            w.setGrad(Tensor.of(0.1f, 0.1f));
            guard.step(List.of(w));
        }
        float[] before = w.value().toArray();
        w.setGrad(Tensor.of(100, 100));
        guard.step(List.of(w));
        assertArrayEquals(before, w.value().toArray(), "the spiky batch is skipped");
        assertEquals(1, guard.skipped());
        assertEquals(1, guard.resets(), "and the inner state reset");
        for (int i = 0; i < 10; i++) {
            assertTrue(!guard.observeLoss(1f));
        }
        assertTrue(guard.observeLoss(10f), "a loss spike is flagged");
        w.setGrad(Tensor.of(0.1f, 0.1f));
        guard.step(List.of(w));
        assertEquals(2, guard.skipped());
        assertEquals(0.1f, guard.learningRate(), 1e-7f);
    }

    @Test
    void muonClipRescalesQueryAndKeyToTheThreshold() {
        SplittableRandom r = new SplittableRandom(2);
        Var wq = Var.param(Tensor.normal(Shape.of(8, 8), 0, 3, r));
        Var wk = Var.param(Tensor.normal(Shape.of(8, 8), 0, 3, r));
        Tensor x = Tensor.normal(Shape.of(2, 5, 8), 0, 1, r);
        float before = max(MuonClip.maxLogits(wq.value(), wk.value(), x, 2, 2, 4, 0.5f, true));
        assertTrue(before > 5);
        MuonClip clip = new MuonClip(Optimizer.sgd(1e-9f), 5).watch(wq, wk, 2, 2, 4, 0.5f, true);
        clip.record(wq, x);
        wq.setGrad(Tensor.zeros(wq.shape()));
        wk.setGrad(Tensor.zeros(wk.shape()));
        clip.step(List.of(wq, wk));
        float[] after = MuonClip.maxLogits(wq.value(), wk.value(), x, 2, 2, 4, 0.5f, true);
        assertEquals(5, max(after), 1e-3f);
        assertEquals(before, clip.lastMaxLogit(), 1e-3f);
        assertEquals(1, clip.clips());
    }

    private static float max(float[] v) {
        float m = Float.NEGATIVE_INFINITY;
        for (float f : v) {
            m = Math.max(m, f);
        }
        return m;
    }

    @Test
    void stochasticRoundingIsUnbiasedOnTheBf16Grid() {
        float x = 1.001f;   // between the bf16 neighbours 1 and 1.0078125
        SplittableRandom r = new SplittableRandom(5);
        double sum = 0;
        int n = 20000;
        for (int i = 0; i < n; i++) {
            float v = StochasticRounding.stochasticBf16(x, r.nextInt(1 << 16));
            assertTrue(v == 1f || v == 1.0078125f);
            sum += v;
        }
        assertEquals(x, sum / n, 2e-4);
        assertEquals(1f, StochasticRounding.nearestBf16(Tensor.of(x)).item());
        assertEquals(-1.0078125f, StochasticRounding.stochasticBf16(-1.0078125f, 65535));
    }

    @Test
    void checkpointAveragingAveragesTheLastK() {
        Var w = Var.param(Tensor.of(0f));
        CheckpointAveraging lawa = new CheckpointAveraging(List.of(w), 3, 2);
        for (int s = 1; s <= 10; s++) {
            w.assign(Tensor.of((float) s));
            lawa.onBatch(1, s, 0);
        }
        assertEquals(3, lawa.size());
        assertEquals((6 + 8 + 10) / 3f, lawa.average().getFirst().item(), 1e-6f);
        lawa.swap();
        assertEquals(8f, w.value().item(), 1e-6f);
        lawa.swap();
        assertEquals(10f, w.value().item());
    }

    @Test
    void batchSizeSchedulesPlanTheBudgetExactly() {
        int[] plan = BatchSizeSchedule.steps(32, 2, 1000, 0.5).plan(1000);
        int total = 0;
        for (int b : plan) {
            total += b;
        }
        assertEquals(1000, total);
        assertEquals(32, plan[0]);
        assertEquals(64, plan[plan.length - 2]);
        assertEquals(8, BatchSizeSchedule.linearRamp(8, 64, 100).batchSize(0));
        assertEquals(36, BatchSizeSchedule.linearRamp(8, 64, 100).batchSize(50));
        assertEquals(64, BatchSizeSchedule.linearRamp(8, 64, 100).batchSize(500));
    }

    @Test
    void psgdAndScionBuildingBlocks() {
        Tensor a = Tensor.of(new float[][] {{4, 1, 0.5f}, {1, 3, 0.2f}, {0.5f, 0.2f, 2}});
        float top = io.github.zzuegg.brainiac.tensor.Linalg.eigh(a)[0].toArray()[2];
        float bound = PsgdKron.spectralLowerBound(a);
        assertTrue(bound <= top + 1e-4f && bound > 0.95f * top, bound + " vs " + top);

        Tensor g = Tensor.normal(Shape.of(16, 8), 0, 1, new SplittableRandom(3));
        Tensor o = Scion.direction(g, Scion.Norm.SPECTRAL);   // sqrt(8/16) U V^T
        Tensor gram = o.transpose().matmul(o).div(0.5f);       // ~ identity [8, 8]
        float[] d = gram.toArray();
        for (int i = 0; i < 8; i++) {
            assertEquals(1, d[i * 8 + i], 0.35, "orthogonalised singular values near 1");
        }
        float[] rows = Scion.direction(g, Scion.Norm.ROW).square().mean(1, false).toArray();
        for (float v : rows) {
            assertEquals(1, v, 1e-4);
        }
        assertEquals(1f / 16, Scion.direction(g, Scion.Norm.SIGN).abs().max().item(), 1e-7f);
    }
}
