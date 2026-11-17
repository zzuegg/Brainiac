package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** The speech building blocks as layers: mel spectrogram, relative attention, HiFi-GAN, WaveNet, splines. */
class AudioLayersTest {

    @TempDir
    Path dir;

    @Test
    void audioModelBuildsTrainsAndRoundTrips() throws Exception {
        Model m = Model.sequential()
                .input(800)                       // 50 ms at 16 kHz
                .melSpectrogram(16_000, 128, 64, 12)   // [12, 13]
                .conv1d(8, 3, Activation.GELU)    // [8, 11]
                .hifiGanBlock(3, 1, 2)
                .waveNet(3, 2)
                .layer(new Permute(1, 0))         // [11, 8]
                .relativeAttention(2, 2)
                .flatten()
                .dense(2, Activation.SOFTMAX)
                .seed(4)
                .build();
        assertEquals(Shape.of(2), m.outputShape());
        Tensor x = Tensor.uniform(Shape.of(3, 800), -1, 1, new SplittableRandom(1));
        Tensor out = m.predict(x);
        assertEquals(Shape.of(3, 2), out.shape());
        // every parameter gets a gradient
        Var loss = m.forward(Var.of(x), true).slice(1, 0, 1).sum();
        loss.backward();
        for (Var p : m.parameters()) {
            assertTrue(p.grad() != null, "no gradient for a parameter of shape " + p.shape());
        }
        Path file = dir.resolve("audio.brainiac");
        m.save(file);
        Model loaded = Model.load(file);
        assertEquals(m.summary(), loaded.summary());
        assertArrayEquals(out.toArray(), loaded.predict(x).toArray(), 1e-5f);
    }

    @Test
    void styleConditionedLayersBuildTrainAndRoundTrip() throws Exception {
        Node in = Node.input(4, 16);
        Node style = in.globalAvgPool().dense(6, Activation.TANH);
        Node x = in.merge(StyleNorm.instance(6), style)
                .merge(AdainResBlock.of(8, 6, true), style)      // [8, 32]
                .merge(AdainSnakeBlock.of(3, 6, 1, 3), style)
                .merge(StyleNorm.layer(6), style)
                .flatten()
                .dense(2);
        Model m = Model.graph(in, x).seed(3).build();
        Tensor batch = Tensor.uniform(Shape.of(3, 4, 16), -1, 1, new SplittableRandom(5));
        Tensor out = m.predict(batch);
        assertEquals(Shape.of(3, 2), out.shape());
        m.forward(Var.of(batch), true).sum().backward();
        for (Var p : m.parameters()) {
            assertTrue(p.grad() != null, "no gradient for a parameter of shape " + p.shape());
        }
        Path file = dir.resolve("style.brainiac");
        m.save(file);
        Model loaded = Model.load(file);
        assertEquals(m.summary(), loaded.summary());
        assertArrayEquals(out.toArray(), loaded.predict(batch).toArray(), 1e-5f);
    }

    @Test
    void lstmFromWeightsMatchesTheCell() {
        // one step of a one-unit LSTM: h = o * tanh(i * g) with c0 = 0
        Recurrent.Settings s = new Recurrent.Settings(Recurrent.Cell.LSTM, 1, 1, true, false, null,
                io.github.zzuegg.brainiac.tensor.spi.ActivationOp.TANH,
                io.github.zzuegg.brainiac.tensor.spi.ActivationOp.SIGMOID, false, true, false);
        Recurrent r = Recurrent.of(s, new Var[] {Var.of(Tensor.of(Shape.of(1, 4), 1, 2, 3, 4))},
                new Var[] {Var.of(Tensor.zeros(Shape.of(1, 4)))}, new Var[] {Var.of(Tensor.zeros(Shape.of(4)))});
        float y = r.forward(Var.of(Tensor.of(Shape.of(1, 1, 1), 0.5f))).value().toArray()[0];
        double i = 1 / (1 + Math.exp(-0.5));
        double g = Math.tanh(1.5);
        double o = 1 / (1 + Math.exp(-2));
        assertEquals(o * Math.tanh(i * g), y, 1e-6);
    }

    @Test
    void whisperMelLayerShape() {
        MelSpectrogram w = MelSpectrogram.whisper(80);
        assertEquals(Shape.of(80, 100), w.outputShape(Shape.of(16_000)));
        Var y = w.forward(Var.of(Tensor.uniform(Shape.of(2, 16_000), -0.5f, 0.5f, new SplittableRandom(2))));
        assertEquals(Shape.of(2, 80, 100), y.shape());
        float[] v = y.value().toArray();
        float max = Float.NEGATIVE_INFINITY;
        float min = Float.POSITIVE_INFINITY;
        for (float f : v) {
            max = Math.max(max, f);
            min = Math.min(min, f);
        }
        assertTrue(max - min <= 2.0001f, "Whisper floors at 8 log10 units (2 after scaling) below the maximum");
    }

    @Test
    void relativeAttentionHasGradients() {
        SplittableRandom r = new SplittableRandom(3);
        RelativeAttention a = new RelativeAttention(4, 2, 1, Init.XAVIER, r);
        Tensor x = Tensor.uniform(Shape.of(1, 5, 4), -1, 1, r);
        Tensor w = Tensor.uniform(Shape.of(1, 5, 4), -1, 1, r);
        GradCheck.Result g = GradCheck.check(v -> a.forward(v.get(0)).mul(w).sum(), List.of(x));
        assertTrue(g.passed(), g.toString());
    }

    @Test
    void splineInvertsAndItsLogDetIsTheSlope() {
        SplittableRandom r = new SplittableRandom(5);
        int n = 7;
        int bins = 5;
        Tensor x = Tensor.of(Shape.of(n), new float[] {-6, -4.5f, -1, 0, 0.3f, 2.2f, 4.9f});
        Var w = Var.of(Tensor.normal(Shape.of(n, bins), 0, 1, r));
        Var h = Var.of(Tensor.normal(Shape.of(n, bins), 0, 1, r));
        Var d = Var.of(Tensor.normal(Shape.of(n, bins - 1), 0, 1, r));
        Flows.Result fwd = Flows.rationalQuadraticSpline(Var.of(x), w, h, d, 5, false);
        Flows.Result inv = Flows.rationalQuadraticSpline(fwd.output(), w, h, d, 5, true);
        assertArrayEquals(x.toArray(), inv.output().value().toArray(), 1e-4f);
        assertEquals(-6f, fwd.output().value().toArray()[0], 0f, "identity outside the tails");
        // the log-determinant is log dy/dx: compare with a finite difference
        float eps = 1e-3f;
        float[] up = Flows.rationalQuadraticSpline(Var.of(x.add(eps)), w, h, d, 5, false).output().value().toArray();
        float[] dn = Flows.rationalQuadraticSpline(Var.of(x.sub(eps)), w, h, d, 5, false).output().value().toArray();
        float[] logDet = fwd.logDet().value().toArray();
        float[] invDet = inv.logDet().value().toArray();
        for (int i = 1; i < n; i++) {
            double slope = (up[i] - dn[i]) / (2 * eps);
            assertEquals(Math.log(slope), logDet[i], 2e-2, "element " + i);
            assertEquals(-logDet[i], invDet[i], 1e-3, "inverse log-det, element " + i);
        }
    }
}
