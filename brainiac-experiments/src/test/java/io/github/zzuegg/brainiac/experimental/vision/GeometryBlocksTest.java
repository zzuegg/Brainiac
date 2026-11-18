package io.github.zzuegg.brainiac.experimental.vision;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.List;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class GeometryBlocksTest {

    private static Tensor random(SplittableRandom r, int... dims) {
        Shape s = Shape.of(dims);
        float[] v = new float[(int) s.size()];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) r.nextGaussian();
        }
        return Tensor.of(s, v);
    }

    private static double maxDiff(float[] a, float[] b) {
        double d = 0;
        for (int i = 0; i < a.length; i++) {
            d = Math.max(d, Math.abs(a[i] - b[i]));
        }
        return d;
    }

    @Test
    void groupConvolutionIsRotationEquivariant() {
        SplittableRandom r = new SplittableRandom(3);
        GroupConv2d lift = new GroupConv2d(GroupConv2d.Kind.LIFT, 1, 3, 3, r);
        GroupConv2d group = new GroupConv2d(GroupConv2d.Kind.GROUP, 3, 2, 3, r);
        Tensor x = random(r, 2, 1, 8, 8);
        Var y = group.forward(lift.forward(Var.of(x)).relu());
        Var yr = group.forward(lift.forward(Var.of(GroupConv2d.rotateImages(x, 1))).relu());
        // rotating the input rotates the maps and shifts the rotation axis by one
        Var expected = GroupConv2d.rotate(y.reshape(2, 2, 4, 8, 8));
        expected = Var.concat(2, expected.slice(2, 3, 4), expected.slice(2, 0, 3));
        assertTrue(maxDiff(expected.value().toArray(), yr.reshape(2, 2, 4, 8, 8).value().toArray()) < 1e-4);
        Var pooled = GroupConv2d.poolRotations(y).sum(3, false).sum(2, false);
        Var pooledR = GroupConv2d.poolRotations(yr).sum(3, false).sum(2, false);
        assertTrue(maxDiff(pooled.value().toArray(), pooledR.value().toArray()) < 1e-3, "invariant after pooling");
    }

    @Test
    void pointNetIgnoresPointOrderAndEdgeConvFindsNeighbours() {
        SplittableRandom r = new SplittableRandom(4);
        PointNet net = new PointNet(3, 32, true, r);
        Tensor x = random(r, 1, 6, 3);
        float[] v = x.toArray();
        float[] shuffled = new float[v.length];
        int[] perm = {3, 1, 5, 0, 2, 4};
        for (int i = 0; i < 6; i++) {
            System.arraycopy(v, 3 * perm[i], shuffled, 3 * i, 3);
        }
        assertTrue(maxDiff(net.forward(Var.of(x)).value().toArray(),
                net.forward(Var.of(Tensor.of(x.shape(), shuffled))).value().toArray()) < 1e-5);
        float[] line = {0, 0, 0, 1, 0, 0, 5, 0, 0, 6, 0, 0};
        float[] nb = EdgeConv.neighbours(Tensor.of(Shape.of(1, 4, 3), line), 1);
        assertEquals(1f, nb[0]);
        assertEquals(3f, nb[2]);
        Var out = new EdgeConv(3, 4, 2, r).forward(Var.of(Tensor.of(Shape.of(1, 4, 3), line)));
        assertEquals(Shape.of(1, 4, 4), out.shape());
    }

    @Test
    void hashGridInterpolatesAndTrains() {
        SplittableRandom r = new SplittableRandom(5);
        HashGridEncoding enc = new HashGridEncoding(2, 3, 2, 6, 2, 16, new float[] {0, 0}, new float[] {1, 1}, r);
        Var y = enc.forward(Var.of(Tensor.of(Shape.of(2, 2), new float[] {0.1f, 0.2f, 0.9f, 0.4f})));
        assertEquals(Shape.of(2, 6), y.shape());
        y.square().sum().backward();
        assertTrue(enc.parameters().get(2).grad() != null);
        assertTrue(enc.collisionShare() > 0, "the finest level (17^2 vertices) overflows 64 rows");
    }

    @Test
    void volumeRendererCompositesAnOpaqueField() {
        // a field of huge density and red colour everywhere: the first sample is opaque
        Layer field = new Layer() {
            @Override
            public Var forward(Var input) {
                int n = input.shape().dim(0);
                float[] v = new float[n * 4];
                for (int i = 0; i < n; i++) {
                    v[4 * i] = 50;
                    v[4 * i + 1] = 10;
                    v[4 * i + 2] = -10;
                    v[4 * i + 3] = -10;
                }
                return Var.of(Tensor.of(Shape.of(n, 4), v));
            }
        };
        VolumeRenderer renderer = new VolumeRenderer(1, 3, 8, 8);
        VolumeRenderer.Rendering out = renderer.render(field, new float[3], new float[] {0, 0, 1}, null);
        float[] c = out.colour().value().toArray();
        assertEquals(1, c[0], 1e-3);
        assertEquals(0, c[1], 1e-3);
        assertEquals(24, renderer.evaluationsPerRay());
    }

    @Test
    void sdfEikonalTermUsesTheInputGradient() {
        SplittableRandom r = new SplittableRandom(6);
        SignedDistanceField f = new SignedDistanceField(SignedDistanceField.Kind.SIREN, 1, 2, 16, 2, r);
        SignedDistanceField.Terms t = f.loss(random(r, 4, 2), Tensor.zeros(Shape.of(4)),
                Tensor.of(Shape.of(2, 2), new float[] {1, 0, 0, 1}));
        t.total().backward();
        List<Var> p = f.parameters();
        assertTrue(p.get(p.size() - 2).grad() != null);
        float[] crossings = SignedDistanceField.zeroCrossings(new float[] {-1, 1, -1, 1}, 2);
        assertEquals(4, crossings.length);
    }
}
