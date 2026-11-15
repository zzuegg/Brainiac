package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.GradCheck;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional.Interpolation;
import io.github.zzuegg.brainiac.nn.Functional.PoolKind;
import io.github.zzuegg.brainiac.tensor.ConvGeometry;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

import org.junit.jupiter.api.Test;

/** Convolution, pooling and resampling against direct reference loops and finite differences. */
class ConvTest {

    private static Tensor random(long seed, int... dims) {
        return Tensor.uniform(Shape.of(dims), -1, 1, new SplittableRandom(seed));
    }

    /** Direct N-d grouped convolution, straight from the definition. */
    static float[] referenceConv(Tensor x, Tensor w, int[] stride, int[] pad, int[] dilation, int groups,
            int[] outDims) {
        int[] xs = x.shape().toArray();
        int[] ws = w.shape().toArray();
        int r = xs.length - 2;
        int n = xs[0];
        int c = xs[1];
        int f = ws[0];
        int cg = c / groups;
        int fg = f / groups;
        float[] xv = x.toArray();
        float[] wv = w.toArray();
        int outSize = 1;
        for (int d : outDims) {
            outSize *= d;
        }
        int kSize = 1;
        for (int i = 2; i < ws.length; i++) {
            kSize *= ws[i];
        }
        int inSize = 1;
        for (int i = 2; i < xs.length; i++) {
            inSize *= xs[i];
        }
        float[] out = new float[n * f * outSize];
        for (int b = 0; b < n; b++) {
            for (int fo = 0; fo < f; fo++) {
                int g = fo / fg;
                for (int o = 0; o < outSize; o++) {
                    int[] oc = unravel(o, outDims);
                    double sum = 0;
                    for (int ci = 0; ci < cg; ci++) {
                        int ch = g * cg + ci;
                        for (int k = 0; k < kSize; k++) {
                            int[] kc = unravel(k, java.util.Arrays.copyOfRange(ws, 2, ws.length));
                            int idx = 0;
                            boolean inside = true;
                            for (int d = 0; d < r; d++) {
                                int pos = oc[d] * stride[d] - pad[d] + kc[d] * dilation[d];
                                if (pos < 0 || pos >= xs[d + 2]) {
                                    inside = false;
                                    break;
                                }
                                idx = idx * xs[d + 2] + pos;
                            }
                            if (inside) {
                                sum += xv[(b * c + ch) * inSize + idx] * wv[(fo * cg + ci) * kSize + k];
                            }
                        }
                    }
                    out[(b * f + fo) * outSize + o] = (float) sum;
                }
            }
        }
        return out;
    }

    private static int[] unravel(int i, int[] dims) {
        int[] out = new int[dims.length];
        for (int d = dims.length - 1; d >= 0; d--) {
            out[d] = i % dims[d];
            i /= dims[d];
        }
        return out;
    }

    private static void assertClose(float[] expected, float[] actual, float tol) {
        assertEquals(expected.length, actual.length, "length");
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], tol, "value " + i);
        }
    }

    @Test
    void conv2dMatchesDirectComputation() {
        Tensor x = random(1, 2, 4, 7, 6);
        Tensor w = random(2, 6, 2, 3, 2);   // 6 filters, 2 groups of 2 channels
        int[] stride = {2, 1};
        int[] pad = {1, 2};
        int[] dil = {1, 2};
        Var y = Functional.conv(Var.of(x), Var.of(w), null, stride, pad, pad, dil, 2);
        int[] outDims = {4, 8};
        assertEquals(Shape.of(2, 6, 4, 8), y.shape());
        assertClose(referenceConv(x, w, stride, pad, dil, 2, outDims), y.value().toArray(), 1e-4f);
    }

    @Test
    void conv1dAnd3dMatchDirectComputation() {
        Tensor x1 = random(3, 2, 3, 10);
        Tensor w1 = random(4, 5, 3, 3);
        Var y1 = Functional.conv(Var.of(x1), Var.of(w1), null, new int[] {3}, new int[] {0}, new int[] {0},
                new int[] {1}, 1);
        assertClose(referenceConv(x1, w1, new int[] {3}, new int[] {0}, new int[] {1}, 1, new int[] {3}),
                y1.value().toArray(), 1e-4f);

        Tensor x3 = random(5, 1, 2, 4, 5, 3);
        Tensor w3 = random(6, 3, 2, 2, 3, 2);
        Var y3 = Functional.conv(Var.of(x3), Var.of(w3), null, new int[] {1, 2, 1}, new int[] {1, 1, 0},
                new int[] {1, 1, 0}, new int[] {1, 1, 1}, 1);
        int[] out = y3.shape().toArray();
        assertClose(referenceConv(x3, w3, new int[] {1, 2, 1}, new int[] {1, 1, 0}, new int[] {1, 1, 1}, 1,
                new int[] {out[2], out[3], out[4]}), y3.value().toArray(), 1e-4f);
    }

    @Test
    void transposedConvIsTheAdjointOfConv() {
        // <conv(x), y> == <x, convTranspose(y)> with the same weights
        Tensor x = random(7, 2, 4, 6, 5);
        Tensor w = random(8, 6, 2, 3, 3);   // conv: 4 channels -> 6 filters, 2 groups
        int[] stride = {2, 2};
        int[] pad = {1, 1};
        int[] dil = {1, 1};
        Var cx = Functional.conv(Var.of(x), Var.of(w), null, stride, pad, pad, dil, 2);
        Tensor y = random(9, cx.shape().toArray());
        // transposed weights are [channels in = 6, filters / groups = 2, k, k]: same tensor
        int[] op = {6 - ((cx.shape().dim(2) - 1) * 2 - 2 + 3), 5 - ((cx.shape().dim(3) - 1) * 2 - 2 + 3)};
        Var ty = Functional.convTranspose(Var.of(y), Var.of(w), null, stride, pad, pad, op, dil, 2);
        assertEquals(x.shape(), ty.shape());
        float lhs = cx.value().mul(y).sum().item();
        float rhs = x.mul(ty.value()).sum().item();
        assertEquals(lhs, rhs, 1e-3f * Math.max(1, Math.abs(lhs)));
    }

    @Test
    void im2colAndCol2imAreAdjoint() {
        ConvGeometry g = ConvGeometry.sliding(2, 3, new int[] {5, 4}, new int[] {3, 2}, new int[] {2, 1},
                new int[] {1, 0}, new int[] {1, 1}, new int[] {1, 2}, true);
        Tensor x = random(10, g.imageShape().toArray());
        Tensor c = random(11, g.columnShape().toArray());
        assertEquals(x.im2col(g, 0).mul(c).sum().item(), x.mul(c.col2im(g)).sum().item(), 1e-3f);
    }

    @Test
    void batchedMatmulBroadcasts() {
        Tensor a = random(12, 2, 3, 4, 5);
        Tensor b = random(13, 3, 5, 2);
        Tensor y = a.matmul(b);
        assertEquals(Shape.of(2, 3, 4, 2), y.shape());
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 3; j++) {
                Tensor ai = a.slice(0, i, i + 1).slice(1, j, j + 1).reshape(4, 5);
                Tensor bj = b.slice(0, j, j + 1).reshape(5, 2);
                Tensor yij = y.slice(0, i, i + 1).slice(1, j, j + 1).reshape(4, 2);
                assertClose(ai.matmul(bj).toArray(), yij.toArray(), 1e-5f);
            }
        }
    }

    @Test
    void gatherAndScatterAdd() {
        Tensor table = Tensor.of(Shape.of(3, 2), 1, 2, 3, 4, 5, 6);
        Tensor ids = Tensor.of(Shape.of(2, 2), 2, 0, 2, 1);
        Tensor picked = table.gather(0, ids);
        assertEquals(Shape.of(2, 2, 2), picked.shape());
        assertArrayEquals(new float[] {5, 6, 1, 2, 5, 6, 3, 4}, picked.toArray());
        assertArrayEquals(new float[] {1, 2, 3, 4, 10, 12}, picked.scatterAdd(0, ids, 3).toArray());
    }

    @Test
    void maxPoolingSendsGradientToOneWinner() {
        Tensor x = Tensor.of(Shape.of(1, 1, 2, 4), 0, 0, 1, 3, 0, 0, 3, 2);
        Var in = Var.param(x);
        Var y = Functional.pool(in, PoolKind.MAX, new int[] {2, 2}, new int[] {2, 2}, new int[2], new int[2],
                new int[] {1, 1}, false, true, 0);
        assertArrayEquals(new float[] {0, 3}, y.value().toArray());
        y.sum().backward();
        assertArrayEquals(new float[] {1, 0, 0, 1, 0, 0, 0, 0}, in.grad().toArray());
    }

    @Test
    void averagePoolingCountsPaddingAsAsked() {
        Tensor x = Tensor.ones(Shape.of(1, 1, 3, 3));
        int[] k = {2, 2};
        int[] one = {1, 1};
        Var with = Functional.pool(Var.of(x), PoolKind.AVG, k, k, one, new int[2], one, false, true, 0);
        Var without = Functional.pool(Var.of(x), PoolKind.AVG, k, k, one, new int[2], one, false, false, 0);
        assertArrayEquals(new float[] {0.25f, 0.5f, 0.5f, 1}, with.value().toArray());
        assertArrayEquals(new float[] {1, 1, 1, 1}, without.value().toArray());
    }

    @Test
    void ceilModeAddsPartialWindows() {
        Shape out = Pool.max(2, 2).ceilMode(true).outputShape(Shape.of(3, 5, 5));
        assertEquals(Shape.of(3, 3, 3), out);
        assertEquals(Shape.of(3, 2, 2), Pool.max(2, 2).outputShape(Shape.of(3, 5, 5)));
    }

    @Test
    void padModes() {
        Var x = Var.of(Tensor.of(Shape.of(1, 1, 3), 1, 2, 3));
        int[] two = {2};
        assertArrayEquals(new float[] {3, 2, 1, 2, 3, 2, 1},
                Functional.pad(x, two, two, PadMode.REFLECT, 0).value().toArray());
        assertArrayEquals(new float[] {2, 1, 1, 2, 3, 3, 2},
                Functional.pad(x, two, two, PadMode.SYMMETRIC, 0).value().toArray());
        assertArrayEquals(new float[] {1, 1, 1, 2, 3, 3, 3},
                Functional.pad(x, two, two, PadMode.REPLICATE, 0).value().toArray());
        assertArrayEquals(new float[] {2, 3, 1, 2, 3, 1, 2},
                Functional.pad(x, two, two, PadMode.CIRCULAR, 0).value().toArray());
        assertArrayEquals(new float[] {9, 9, 1, 2, 3, 9, 9},
                Functional.pad(x, two, two, PadMode.CONSTANT, 9).value().toArray());
        assertArrayEquals(new float[] {2}, Functional.pad(x, new int[] {-1}, new int[] {-1}, PadMode.CONSTANT, 0)
                .value().toArray());
    }

    @Test
    void resizeModes() {
        Var x = Var.of(Tensor.of(Shape.of(1, 1, 2), 0, 1));
        assertArrayEquals(new float[] {0, 0, 1, 1},
                Functional.resize(x, new int[] {4}, Interpolation.NEAREST, false).value().toArray());
        assertArrayEquals(new float[] {0, 1 / 3f, 2 / 3f, 1},
                Functional.resize(x, new int[] {4}, Interpolation.LINEAR, true).value().toArray(), 1e-6f);
        assertArrayEquals(new float[] {0, 0.25f, 0.75f, 1},
                Functional.resize(x, new int[] {4}, Interpolation.LINEAR, false).value().toArray(), 1e-6f);
    }

    @Test
    void pixelShuffleRoundTrips() {
        Tensor x = random(14, 2, 8, 3, 3);
        Var up = Functional.pixelShuffle(Var.of(x), 2);
        assertEquals(Shape.of(2, 2, 6, 6), up.shape());
        assertArrayEquals(x.toArray(), Functional.pixelUnshuffle(up, 2).value().toArray());
    }

    @Test
    void adaptivePoolingMatchesRegularPoolingWhenItDivides() {
        Tensor x = random(15, 2, 3, 6, 4);
        Var a = Functional.adaptivePool(Var.of(x), PoolKind.MAX, new int[] {3, 2});
        Var b = Functional.pool(Var.of(x), PoolKind.MAX, new int[] {2, 2}, new int[] {2, 2}, new int[2], new int[2],
                new int[] {1, 1}, false, true, 0);
        assertArrayEquals(b.value().toArray(), a.value().toArray());
        Var c = Functional.adaptivePool(Var.of(x), PoolKind.AVG, new int[] {3, 2});
        Var d = Functional.pool(Var.of(x), PoolKind.AVG, new int[] {2, 2}, new int[] {2, 2}, new int[2], new int[2],
                new int[] {1, 1}, false, true, 0);
        assertArrayEquals(d.value().toArray(), c.value().toArray(), 1e-6f);
    }

    // ---------------------------------------------------------------- gradients

    private static void gradCheck(String name, Function<List<Var>, Var> f, Tensor... inputs) {
        Tensor out = f.apply(java.util.Arrays.stream(inputs).map(Var::of).toList()).value();
        Tensor weights = Tensor.uniform(out.shape(), 0.5f, 1.5f, new SplittableRandom(99));
        GradCheck.Result r = GradCheck.check(v -> f.apply(v).mul(weights).sum(), List.of(inputs));
        assertTrue(r.passed(), name + ": " + r);
    }

    @Test
    void convolutionGradients() {
        gradCheck("conv2d grouped strided dilated", v -> Functional.conv(v.get(0), v.get(1), v.get(2),
                new int[] {2, 1}, new int[] {1, 0}, new int[] {0, 1}, new int[] {1, 2}, 2),
                random(20, 2, 4, 5, 5), random(21, 4, 2, 2, 2), random(22, 4));
        gradCheck("conv1d", v -> Functional.conv(v.get(0), v.get(1), null, new int[] {1}, new int[] {1},
                new int[] {1}, new int[] {1}, 1), random(23, 2, 2, 6), random(24, 3, 2, 3));
        gradCheck("conv3d", v -> Functional.conv(v.get(0), v.get(1), null, new int[] {1, 1, 1},
                new int[] {0, 0, 0}, new int[] {0, 0, 0}, new int[] {1, 1, 1}, 1),
                random(25, 1, 2, 3, 3, 3), random(26, 2, 2, 2, 2, 2));
        gradCheck("transposed conv", v -> Functional.convTranspose(v.get(0), v.get(1), v.get(2), new int[] {2, 2},
                new int[] {1, 1}, new int[] {0, 1}, new int[] {1, 0}, new int[] {1, 1}, 2),
                random(27, 2, 4, 3, 3), random(28, 4, 1, 3, 3), random(29, 2));
    }

    @Test
    void poolingAndResamplingGradients() {
        Tensor x = random(30, 2, 2, 5, 4);
        int[] k = {2, 2};
        gradCheck("avg pool", v -> Functional.pool(v.get(0), PoolKind.AVG, k, new int[] {2, 1}, new int[] {1, 0},
                new int[] {0, 1}, new int[] {1, 1}, true, false, 0), x);
        gradCheck("lp pool", v -> Functional.pool(v.get(0), PoolKind.LP, k, k, new int[2], new int[2],
                new int[] {1, 1}, false, true, 3), x);
        gradCheck("global avg", v -> Functional.globalPool(v.get(0), PoolKind.AVG, false), x);
        gradCheck("adaptive avg", v -> Functional.adaptivePool(v.get(0), PoolKind.AVG, new int[] {3, 3}), x);
        gradCheck("reflect pad", v -> Functional.pad(v.get(0), new int[] {2, 1}, new int[] {1, 3}, PadMode.REFLECT,
                0), x);
        gradCheck("bilinear", v -> Functional.resize(v.get(0), new int[] {7, 3}, Interpolation.LINEAR, false), x);
        gradCheck("bicubic", v -> Functional.resize(v.get(0), new int[] {8, 6}, Interpolation.CUBIC, true), x);
        gradCheck("nearest", v -> Functional.resize(v.get(0), new int[] {10, 8}, Interpolation.NEAREST, false), x);
        gradCheck("group norm", v -> new GroupNorm(2, 2, 1e-5f, true).forward(v.get(0)), x);
        gradCheck("layer norm", v -> new LayerNorm(new int[] {5, 4}, 1e-5f, true, true, false).forward(v.get(0)), x);
        gradCheck("rms norm", v -> LayerNorm.rms(4).forward(v.get(0)), x);
        gradCheck("batch norm 2d", v -> new BatchNorm(2).forward(v.get(0), true), x);
    }

    // ---------------------------------------------------------------- layers and models

    @Test
    void specsComputeOutputShapes() {
        Shape img = Shape.of(3, 32, 32);
        assertEquals(Shape.of(16, 30, 30), Conv.of(16, 3, 3).outputShape(img));
        assertEquals(Shape.of(16, 16, 16), Conv.of(16, 3, 3).stride(2).padding(Padding.SAME).outputShape(img));
        assertEquals(Shape.of(16, 34, 34), Conv.of(16, 3, 3).padding(Padding.FULL).outputShape(img));
        assertEquals(Shape.of(16, 28, 28), Conv.of(16, 3, 3).dilation(2).outputShape(img));
        assertEquals(Shape.of(6, 32, 32), Conv.depthwise(3, 3).multiplier(2).padding(Padding.SAME).outputShape(img));
        assertEquals(Shape.of(8, 64, 64), Conv.transposed(8, 4, 4).stride(2).padding(1).outputShape(img));
        assertEquals(Shape.of(8, 64, 64), Conv.transposed(8, 3, 3).stride(2).padding(Padding.SAME).outputShape(img));
        assertEquals(Shape.of(8, 65, 65), Conv.transposed(8, 3, 3).stride(2).outputShape(img));
        assertEquals(Shape.of(3, 16, 16), Pool.max(2, 2).outputShape(img));
        assertEquals(Shape.of(3, 32, 32), Pool.avg(3, 3).stride(1).padding(Padding.SAME).outputShape(img));
    }

    @Test
    void layersRunAtTheirComputedShapes() {
        SplittableRandom random = new SplittableRandom(1);
        Shape img = Shape.of(4, 9, 8);
        for (LayerSpec spec : List.<LayerSpec>of(
                Conv.of(6, 3, 3).stride(2).padding(Padding.SAME).padMode(PadMode.REFLECT).activation(Activation.prelu()),
                Conv.of(4, 3, 2).groups(2).dilation(2, 1).padding(Padding.FULL).padMode(PadMode.CIRCULAR),
                Conv.depthwise(3, 3).multiplier(3).padding(1),
                Conv.transposed(5, 3, 3).stride(2).padding(Padding.SAME).groups(1),
                Conv.transposed(6, 2, 2).stride(2).groups(2).outputPadding(1),
                Pool.max(3, 3).stride(2).padding(1).ceilMode(true),
                Pool.avg(2, 3).countPadding(false).padding(1),
                Pool.lp(2, 2, 2),
                Pool.min(2, 2).dilation(2))) {
            Layer layer = spec.create(img, random);
            Var y = layer.forward(Var.of(random(3, 2, 4, 9, 8)), true);
            assertEquals(spec.outputShape(img), Shape.of(java.util.Arrays.copyOfRange(y.shape().toArray(), 1, 4)),
                    spec.toString());
            assertEquals(spec.outputShape(img), layer.outputShape(img), spec.toString());
        }
    }

    @Test
    void sequentialCnnBuildsAndTrainsAStep() {
        Model model = Model.sequential()
                .input(1, 12, 12)
                .add(Conv.of(4, 3, 3).padding(Padding.SAME).bias(false))
                .batchNorm()
                .activation(Activation.RELU)
                .maxPool(2, 2)
                .separableConv(8, Activation.RELU, 3, 3)
                .spatialDropout(0.1f)
                .groupNorm(2)
                .avgPool(2, 2)
                .flatten()
                .dense(3, Activation.SOFTMAX)
                .seed(3)
                .build();
        assertEquals(Shape.of(1, 12, 12), model.inputShape());
        assertEquals(Shape.of(3), model.outputShape());
        Tensor batch = random(40, 5, 144);
        Var out = model.forward(Var.of(batch), true);
        assertEquals(Shape.of(5, 3), out.shape());
        out.sum().backward();
        for (Var p : model.parameters()) {
            assertTrue(p.grad() != null, "no gradient for a parameter of shape " + p.shape());
        }
        float[] one = model.predict(random(41, 144).toArray());
        assertEquals(3, one.length);
    }

    @Test
    void graphWithImageBranches() {
        Node in = Node.input(3, 8, 8);
        Node a = in.conv2d(4, 3, Activation.RELU);
        Node b = in.add(Conv.of(4, 3, 3).dilation(1)).add(Pool.avg(1, 1));
        Node joined = a.concat(b).add(a.multiply(b).concat(a));
        Node out = joined.globalAvgPool().dense(2, Activation.SOFTMAX);
        assertEquals(Shape.of(8, 6, 6), joined.shape());
        Model m = Model.graph(in, out).seed(1).build();
        assertEquals(Shape.of(4, 2), m.predict(random(5, 4, 3, 8, 8)).shape());
    }

    @Test
    void imageModelsSaveAndLoad(@org.junit.jupiter.api.io.TempDir java.nio.file.Path dir) throws Exception {
        Model model = Model.sequential()
                .input(2, 10, 10)
                .add(Conv.of(4, 3, 3).stride(2).padding(Padding.SAME).padMode(PadMode.REFLECT)
                        .activation(Activation.prelu()))
                .add(Conv.depthwise(3, 3).multiplier(2).padding(1))
                .add(Conv.transposed(3, 2, 2).stride(2).groups(1).bias(false))
                .layer(new Pad(new int[] {1, 0}, new int[] {0, 1}, PadMode.REPLICATE, 0))
                .add(Pool.lp(2, 2, 2).ceilMode(true))
                .groupNorm(3)
                .layer(LayerNorm.rms(6))
                .layer(Resize.scale(Interpolation.LINEAR, 2).alignCorners(true))
                .layer(new PixelShuffle(2, true))
                .layer(AdaptivePool.max(2, 3))
                .layer(new Permute(0, 2, 1))
                .layer(new Reshape(6, -1))
                .spatialDropout(0.2f)
                .reshape(-1)
                .dense(2)
                .seed(5)
                .build();
        java.nio.file.Path file = dir.resolve("cnn.bnm");
        model.save(file);
        Model loaded = Model.load(file);
        assertEquals(model.inputShape(), loaded.inputShape());
        assertEquals(model.summary(), loaded.summary());
        Tensor x = random(50, 3, 200);
        assertArrayEquals(model.predict(x).toArray(), loaded.predict(x).toArray(), 1e-6f);
    }
}
