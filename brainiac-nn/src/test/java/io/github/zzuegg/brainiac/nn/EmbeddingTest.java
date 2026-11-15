package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.nio.file.Path;
import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Embeddings, padding masks and masked pooling. */
class EmbeddingTest {

    private static final SplittableRandom R = new SplittableRandom(1);

    private static Tensor table(int v, int d) {
        float[] w = new float[v * d];
        for (int i = 0; i < w.length; i++) {
            w[i] = i + 1;
        }
        return Tensor.of(Shape.of(v, d), w);
    }

    private static Embedding create(Embedding.Spec spec, int... input) {
        return spec.create(Shape.of(input), R);
    }

    @Test
    void looksUpTheRowOfEveryId() {
        Embedding e = create(Embedding.pretrained(table(4, 3)), 2);
        assertEquals(Shape.of(2, 3), e.outputShape(Shape.of(2)));
        Var out = e.forward(Var.of(Tensor.of(Shape.of(2, 2), 3, 0, 1, 3)));
        assertEquals(Shape.of(2, 2, 3), out.shape());
        assertArrayEquals(new float[] {10, 11, 12, 1, 2, 3, 4, 5, 6, 10, 11, 12}, out.value().toArray());
    }

    @Test
    void onlyUsedRowsReceiveGradientAndRepeatsAddUp() {
        Embedding e = create(Embedding.pretrained(table(4, 2)), 3);
        e.forward(Var.of(Tensor.of(Shape.of(1, 3), 1, 3, 1))).sum().backward();
        Var w = e.parameters().get(0);
        assertArrayEquals(new float[] {0, 0, 2, 2, 0, 0, 1, 1}, w.grad().toArray());
    }

    @Test
    void paddingRowIsZeroAndNeverTrained() {
        Embedding e = create(Embedding.pretrained(table(3, 2)).paddingIndex(0), 2);
        assertArrayEquals(new float[] {0, 0}, e.weights().slice(0, 0, 1).toArray());
        Var out = e.forward(Var.of(Tensor.of(Shape.of(1, 2), 0, 2)));
        assertArrayEquals(new float[] {0, 0, 5, 6}, out.value().toArray());
        out.sum().backward();
        assertArrayEquals(new float[] {0, 0, 0, 0, 1, 1}, e.parameters().get(0).grad().toArray());
        assertThrows(IllegalArgumentException.class, () -> create(Embedding.of(3, 2).paddingIndex(3), 1));
    }

    @Test
    void frozenVectorsAreBuffersNotParameters() {
        Embedding e = create(Embedding.pretrained(table(3, 2)).frozen(), 1);
        assertTrue(e.parameters().isEmpty());
        assertEquals(1, e.buffers().size());
        assertTrue(e.toString().contains("frozen"));
    }

    @Test
    void maxNormShortensLongVectorsAndHasTheRightGradient() {
        Tensor w = Tensor.uniform(Shape.of(5, 3), -2, 2, new SplittableRandom(2));
        Tensor ids = Tensor.of(Shape.of(1, 4), 0, 4, 2, 4);
        Tensor weights = Tensor.uniform(Shape.of(1, 4, 3), 0.5f, 1.5f, new SplittableRandom(3));
        Embedding e = create(Embedding.pretrained(w).maxNorm(1), 4);
        Var out = e.forward(Var.of(ids));
        float[] o = out.value().toArray();
        for (int i = 0; i < 4; i++) {
            double n = Math.sqrt(o[3 * i] * o[3 * i] + o[3 * i + 1] * o[3 * i + 1] + o[3 * i + 2] * o[3 * i + 2]);
            assertTrue(n <= 1 + 1e-4, "norm " + n);
        }
        out.mul(weights).sum().backward();
        float[] analytic = e.parameters().get(0).grad().toArray();
        float[] base = w.toArray();
        double h = 1e-3;
        for (int i = 0; i < base.length; i++) {
            double[] f = new double[2];
            for (int k = 0; k < 2; k++) {
                float[] p = base.clone();
                p[i] += (float) (k == 0 ? h : -h);
                Embedding pe = create(Embedding.pretrained(Tensor.of(Shape.of(5, 3), p)).maxNorm(1), 4);
                f[k] = pe.forward(Var.of(ids)).mul(weights).sum().value().item();
            }
            assertEquals((f[0] - f[1]) / (2 * h), analytic[i], 2e-2, "weight " + i);
        }
    }

    @Test
    void scalingMultipliesBySqrtDimension() {
        Embedding e = create(Embedding.pretrained(table(2, 4)).scaleBySqrtDimension(true), 1);
        assertArrayEquals(new float[] {10, 12, 14, 16}, e.forward(Var.of(Tensor.of(Shape.of(1, 1), 1)))
                .value().toArray(), 1e-5f);
    }

    @Test
    void nearestFindsTheMostSimilarDirection() {
        Tensor w = Tensor.of(Shape.of(4, 2), 1, 0, 0, 1, 2, 0.1f, -1, 0);
        Embedding e = create(Embedding.pretrained(w), 1);
        assertArrayEquals(new int[] {2, 1}, e.nearest(0, 2));
    }

    @Test
    void paddingMaskAndMaskedPooling() {
        Tensor ids = Tensor.of(Shape.of(2, 3), 5, 7, 0, 0, 0, 0);
        Tensor mask = new PaddingMask(0).forward(Var.of(ids)).value();
        assertArrayEquals(new float[] {1, 1, 0, 0, 0, 0}, mask.toArray());
        Tensor x = Tensor.of(Shape.of(2, 3, 2), 1, -2, 3, 4, 100, 100, 9, 9, 9, 9, 9, 9);
        float[][] expected = {
            {2, 1, 0, 0},     // mean
            {3, 4, 0, 0},     // max
            {4, 2, 0, 0},     // sum
            {1, -2, 0, 0},    // first
            {3, 4, 0, 0},     // last
        };
        for (SequencePool.Kind kind : SequencePool.Kind.values()) {
            MaskedPool p = (MaskedPool) MaskedPool.of(kind).create(List.of(Shape.of(3, 2), Shape.of(3)), R);
            Var in = Var.param(x);
            Var out = p.forward(List.of(in, Var.of(mask)), false);
            assertArrayEquals(expected[kind.ordinal()], out.value().toArray(), 1e-5f, kind.name());
            out.sum().backward();
            float[] g = in.grad().toArray();
            for (int i = 4; i < 12; i++) {
                assertEquals(0, g[i], 1e-6, kind + ": padded steps get no gradient");
            }
        }
    }

    @Test
    void maskedTransformerIgnoresWhatIsInThePadding(@TempDir Path dir) throws Exception {
        Node ids = Node.input(5);
        Node mask = ids.layer(new PaddingMask(0));
        Node x = ids.add(Embedding.of(10, 8).paddingIndex(0).maxNorm(3)).positionalEncoding()
                .merge(TransformerBlock.of(2).keyMask(true), mask)
                .merge(MaskedPool.of(SequencePool.Kind.MEAN), mask)
                .dense(2, Activation.SOFTMAX);
        Model model = Model.graph(ids, x).seed(5).build();
        Tensor a = Tensor.of(Shape.of(1, 5), 3, 4, 0, 0, 0);
        float[] p1 = model.predict(a).toArray();
        // whatever sits in the padded steps cannot change a masked block followed by masked pooling
        TransformerBlock block = (TransformerBlock) TransformerBlock.of(2).keyMask(true)
                .create(List.of(Shape.of(5, 8), Shape.of(5)), new SplittableRandom(6));
        MaskedPool pool = (MaskedPool) MaskedPool.of(SequencePool.Kind.MEAN)
                .create(List.of(Shape.of(5, 8), Shape.of(5)), R);
        Var m = Var.of(Tensor.of(Shape.of(1, 5), 1, 1, 0, 0, 0));
        float[] clean = Tensor.uniform(Shape.of(1, 5, 8), -1, 1, new SplittableRandom(7)).toArray();
        float[] noisy = clean.clone();
        for (int i = 16; i < noisy.length; i++) {
            noisy[i] = 50 - i;
        }
        float[] y1 = pool.forward(List.of(block.forward(List.of(Var.of(Tensor.of(Shape.of(1, 5, 8), clean)), m),
                false), m), false).value().toArray();
        float[] y2 = pool.forward(List.of(block.forward(List.of(Var.of(Tensor.of(Shape.of(1, 5, 8), noisy)), m),
                false), m), false).value().toArray();
        assertArrayEquals(y1, y2, 1e-4f);
        Var out = model.forward(Var.of(Tensor.of(Shape.of(2, 5), 3, 4, 0, 0, 0, 1, 2, 3, 4, 5)), true);
        out.sum().backward();
        Model wide = Model.graph(ids, x).seed(5).build().to(io.github.zzuegg.brainiac.tensor.DType.FLOAT64);
        assertArrayEquals(p1, wide.predict(a).to(io.github.zzuegg.brainiac.tensor.DType.FLOAT32).toArray(), 1e-4f);
        Model half = Model.graph(ids, x).seed(5).build().to(io.github.zzuegg.brainiac.tensor.DType.BFLOAT16);
        assertArrayEquals(p1, half.predict(a).to(io.github.zzuegg.brainiac.tensor.DType.FLOAT32).toArray(), 0.1f);
        Path file = dir.resolve("embedding.bnm");
        model.save(file);
        Model loaded = Model.load(file);
        assertEquals(model.summary(), loaded.summary());
        assertArrayEquals(p1, loaded.predict(a).toArray(), 1e-5f);
    }

    @Test
    void sequentialModelWithAFrozenPretrainedTableSavesIt(@TempDir Path dir) throws Exception {
        Model model = Model.sequential().input(4)
                .add(Embedding.pretrained(table(6, 3)).frozen())
                .lstm(5, false)
                .dense(2, Activation.SOFTMAX)
                .seed(1)
                .build();
        Tensor x = Tensor.of(Shape.of(1, 4), 1, 2, 3, 5);
        Path file = dir.resolve("frozen.bnm");
        model.save(file);
        Model loaded = Model.load(file);
        assertArrayEquals(model.predict(x).toArray(), loaded.predict(x).toArray(), 1e-6f);
        Embedding e = (Embedding) loaded.layers().get(0);
        assertArrayEquals(table(6, 3).toArray(), e.weights().toArray());
    }
}
