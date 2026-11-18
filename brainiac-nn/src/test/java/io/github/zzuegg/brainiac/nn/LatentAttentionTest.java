package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/** DeepSeek's building blocks in core: latent attention and group-limited expert routing. */
class LatentAttentionTest {

    @TempDir
    Path dir;

    private static Tensor random(long seed, int... dims) {
        return Tensor.uniform(Shape.of(dims), -1, 1, new SplittableRandom(seed));
    }

    /** Decoding over cached latents (absorbed) gives what the whole sequence (expanded) does. */
    @ParameterizedTest
    @CsvSource({"0, true", "6, false", "0, false"})
    void cachedDecodingMatchesTheWholeSequence(int qRank, boolean interleaved) {
        LatentAttention.Settings s = new LatentAttention.Settings(3, qRank, 8, 4, 6, 5, 100f, interleaved, true, 1e-6f);
        LatentAttention a = new LatentAttention(s, 12, new SplittableRandom(1));
        Tensor x = random(2, 2, 7, 12);
        float[] whole = a.forward(Var.of(x)).value().toArray();
        LatentAttention.Cache cache = a.newCache();
        List<Tensor> parts = new ArrayList<>();
        parts.add(a.forward(Var.of(x.slice(1, 0, 4)), cache).value());   // a prompt of 4, then one step at a time
        for (int t = 4; t < 7; t++) {
            parts.add(a.forward(Var.of(x.slice(1, t, t + 1)), cache).value());
        }
        assertEquals(7, cache.length());
        assertEquals(Shape.of(2, 1, 7, 8), cache.latents().shape());
        assertEquals(Shape.of(2, 1, 7, 6), cache.rotary().shape());
        assertArrayEquals(whole, Tensor.concat(1, parts.toArray(Tensor[]::new)).toArray(), 1e-5f);
    }

    @Test
    void trainsSavesAndLoadsInAModel() throws Exception {
        Node in = Node.input(5, 10);
        Node x = in.add(LatentAttention.of(new LatentAttention.Settings(2, 4, 6, 4, 4, 3, 10_000f, true, true, 1e-6f)))
                .add(LatentAttention.of(new LatentAttention.Settings(1, 0, 4, 2, 2, 4, 10_000f, false, false, 1e-5f)))
                .flatten()
                .dense(2);
        Model m = Model.graph(in, x).seed(3).build();
        Tensor batch = random(4, 2, 5, 10);
        Tensor out = m.predict(batch);
        assertEquals(Shape.of(2, 2), out.shape());
        m.forward(Var.of(batch), true).sum().backward();
        for (Var p : m.parameters()) {
            assertTrue(p.grad() != null, "no gradient for a parameter of shape " + p.shape());
        }
        Path file = dir.resolve("mla.brainiac");
        m.save(file);
        Model loaded = Model.load(file);
        assertEquals(m.summary(), loaded.summary());
        assertArrayEquals(out.toArray(), loaded.predict(batch).toArray(), 1e-5f);
    }

    /** The backend's fused kernel (inference) and the separate operations (gradients) agree. */
    @Test
    void latentAttentionFusedAndComposedAgree() {
        Tensor qc = random(5, 2, 3, 4, 16).mul(0.3f);
        Tensor qr = random(6, 2, 3, 4, 4);
        Tensor c = random(7, 2, 1, 9, 16);
        Tensor r = random(8, 2, 1, 9, 4);
        Tensor bias = Functional.causalMask(4, 9, qc.device());
        float[] fused = Functional.latentAttention(Var.of(qc), Var.of(qr), Var.of(c), Var.of(r), bias, 0.4f).value()
                .toArray();
        Var grad = Var.param(qc);
        float[] composed = Functional.latentAttention(grad, Var.of(qr), Var.of(c), Var.of(r), bias, 0.4f).value()
                .toArray();
        assertArrayEquals(composed, fused, 1e-5f);
        // the same as ordinary attention with keys [c | r] and values c, shared by every head
        Var q = Var.of(Tensor.concat(3, qc, qr));
        Var k = Var.of(Tensor.concat(3, c, r));
        float[] plain = Functional.attention(q, k, Var.of(c), bias, 0.4f, 0, null).value().toArray();
        assertArrayEquals(plain, fused, 1e-5f);
    }

    @Test
    void absorbedUpProjectionSplitsPerHead() {
        int heads = 2;
        int nope = 3;
        int vd = 2;
        Tensor up = random(9, 4, heads * (nope + vd));
        Tensor[] w = Functional.absorbLatentUp(up, heads, nope, vd);
        assertEquals(Shape.of(1, heads, nope, 4), w[0].shape());
        assertEquals(Shape.of(1, heads, 4, vd), w[1].shape());
        float[] u = up.toArray();
        float[] uk = w[0].toArray();
        float[] uv = w[1].toArray();
        int cols = heads * (nope + vd);
        for (int h = 0; h < heads; h++) {
            for (int l = 0; l < 4; l++) {
                for (int i = 0; i < nope; i++) {
                    assertEquals(u[l * cols + h * (nope + vd) + i], uk[(h * nope + i) * 4 + l]);
                }
                for (int i = 0; i < vd; i++) {
                    assertEquals(u[l * cols + h * (nope + vd) + nope + i], uv[(h * 4 + l) * vd + i]);
                }
            }
        }
    }

    @Test
    void routingPicksWithinTheBestGroups() {
        // 8 experts in 4 groups of 2, 2 groups per token, 3 experts per token
        float[] scores = {0.9f, 0.1f, 0.5f, 0.5f, 0.8f, 0.0f, 0.2f, 0.3f};
        // DeepSeek-V2: groups ranked by their best expert: groups 0 (0.9) and 2 (0.8)
        Functional.ExpertGroups v2 = new Functional.ExpertGroups(4, 2, false);
        assertArrayEquals(new int[] {0, 4, 1}, Functional.chooseExperts(scores, null, 1, 8, 3, v2));
        // V3: ranked by the sum of the two best: groups 1 (1.0) and 0 (1.0) beat 2 (0.8)
        Functional.ExpertGroups v3 = new Functional.ExpertGroups(4, 2, true);
        assertArrayEquals(new int[] {0, 2, 3}, Functional.chooseExperts(scores, null, 1, 8, 3, v3));
        // a selection bias moves the choice
        float[] bias = {0, 0, 0, 0, 0, 0, 1, 1};
        assertArrayEquals(new int[] {7, 6, 0}, Functional.chooseExperts(scores, bias, 1, 8, 3, v3));
        // ungrouped: the plain top k
        assertArrayEquals(new int[] {0, 4, 2},
                Functional.chooseExperts(scores, null, 1, 8, 3, Functional.ExpertGroups.NONE));
    }

    @Test
    void routedExpertsSumEachTokensWeightedOutputs() {
        int n = 5;
        int k = 2;
        Tensor x = random(10, n, 4);
        List<Var> weights = new ArrayList<>();
        List<Function<Var, Var>> experts = new ArrayList<>();
        for (int e = 0; e < 3; e++) {
            Var w = Var.param(random(20 + e, 4, 3));
            weights.add(w);
            experts.add(v -> v.matmul(w));
        }
        int[] chosen = {0, 2, 1, 0, 2, 1, 0, 1, 2, 0};
        float[] gate = {0.7f, 0.3f, 0.5f, 0.5f, 0.9f, 0.1f, 0.6f, 0.4f, 0.2f, 0.8f};
        Var g = Var.param(Tensor.of(Shape.of(n * k), gate));
        Var out = Functional.routedExperts(Var.of(x), chosen, g, k, experts);
        float[] got = out.value().toArray();
        for (int t = 0; t < n; t++) {
            Tensor row = x.slice(0, t, t + 1);
            float[] expect = new float[3];
            for (int j = 0; j < k; j++) {
                float[] y = row.matmul(weights.get(chosen[t * k + j]).value()).toArray();
                for (int i = 0; i < 3; i++) {
                    expect[i] += gate[t * k + j] * y[i];
                }
            }
            assertArrayEquals(expect, java.util.Arrays.copyOfRange(got, t * 3, t * 3 + 3), 1e-5f);
        }
        out.sum().backward();
        assertTrue(g.grad() != null && weights.get(0).grad() != null && weights.get(2).grad() != null);
    }
}
