package io.github.zzuegg.brainiac.backend.vulkan;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.zzuegg.brainiac.autograd.Decompositions;
import io.github.zzuegg.brainiac.autograd.OptimalTransport;
import io.github.zzuegg.brainiac.autograd.Transforms;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Special;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Optional;
import java.util.SplittableRandom;
import java.util.function.Function;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** The recent core additions give the CPU's results on a Vulkan device and keep their values there. */
class CoreAdditionsTest {

    private static Device gpu;
    private static final Device CPU = Device.cpu();

    @BeforeAll
    static void findDevice() {
        Optional<Device> found = Device.available().stream().filter(d -> d.name().startsWith("vulkan")).findFirst();
        assumeTrue(found.isPresent(), "no Vulkan device");
        gpu = found.get();
    }

    private static void same(Function<Device, Tensor> f, float tol) {
        Tensor want = f.apply(CPU);
        Tensor got = f.apply(gpu);
        assertEquals(gpu, got.device());
        assertEquals(want.shape(), got.shape());
        float[] w = want.toArray();
        float[] g = got.to(CPU).toArray();
        for (int i = 0; i < w.length; i++) {
            assertEquals(w[i], g[i], tol * Math.max(1, Math.abs(w[i])), "[" + i + "]");
        }
    }

    private static Tensor random(Device d, long seed, int... dims) {
        return Tensor.uniform(Shape.of(dims), -2, 2, new SplittableRandom(seed)).to(d);
    }

    @Test
    void tensorAdditions() {
        same(d -> Special.lgamma(random(d, 1, 8).abs().add(0.1f)), 1e-5f);
        same(d -> Special.digamma(random(d, 2, 8).abs().add(0.1f)), 1e-5f);
        same(d -> Special.erfc(random(d, 3, 8)), 1e-6f);
        same(d -> random(d, 4, 3, 4).maximum(0f).minimum(1f), 0);
        same(d -> random(d, 5, 2, 3, 4).reshape(-1, 4), 0);
        same(d -> Linalg.slogdet(random(d, 6, 2, 3, 3))[1], 1e-5f);
        same(d -> Linalg.squaredDistances(random(d, 14, 2, 5, 3), random(d, 15, 2, 4, 3)), 1e-5f);
        same(d -> Linalg.nearestNeighbours(random(d, 16, 6, 3), random(d, 17, 9, 3), 3)[1], 0);
        Tensor cost = random(gpu, 7, 4, 5);
        assertArrayEquals(Linalg.linearAssignment(cost.to(CPU)), Linalg.linearAssignment(cost));
    }

    @Test
    void varAdditions() {
        same(d -> Var.of(random(d, 8, 3, 4, 2)).softmax(1).value(), 1e-5f);
        same(d -> Var.of(random(d, 9, 3, 4)).normalize(0, 1e-5f).value(), 1e-4f);
        same(d -> Var.of(random(d, 10, 6)).asinh().value(), 1e-5f);
        same(d -> Var.of(random(d, 11, 2, 3)).rot90(1, 0, 1).value(), 0);
        same(d -> Transforms.vjp(xs -> xs.get(0).lgamma(), List.of(random(d, 12, 5).abs().add(0.5f)),
                Tensor.full(Shape.of(5), 1f, DType.FLOAT32, d)).gradients().get(0), 1e-5f);
        // gradient hooks fire on the device's gradient
        same(d -> {
            Var x = Var.param(random(d, 13, 4));
            Tensor[] seen = new Tensor[1];
            Var y = x.mul(x).onGradient(g -> {
                seen[0] = g;
                return g.mul(2f);
            });
            y.sum().backward();
            assertEquals(d, seen[0].device());
            return x.grad();
        }, 1e-6f);
        // integer tensors stay on the CPU and say so
        Tensor ids = Tensor.of(Shape.of(2), new long[] {1, 2}, DType.INT64, CPU);
        assertThrows(UnsupportedOperationException.class, () -> ids.to(gpu));
        assertArrayEquals(new float[] {1, 2}, ids.to(DType.FLOAT32).to(gpu).to(CPU).toArray());
    }

    @Test
    void matrixAndStatisticsOps() {
        same(d -> Var.of(random(d, 20, 2, 3, 4)).tril(0).value(), 0);
        same(d -> Var.of(random(d, 21, 3, 3)).triu(-1).value(), 0);
        same(d -> Var.of(random(d, 22, 2, 3, 3)).trace().value(), 1e-6f);
        same(d -> Var.of(random(d, 23, 4, 3)).cross(Var.of(random(d, 24, 4, 3)), 1).value(), 1e-6f);
        same(d -> Var.of(random(d, 25, 2, 3)).kron(Var.of(random(d, 26, 3, 2))).value(), 1e-6f);
        same(d -> Var.of(random(d, 27, 6)).logaddexp(Var.of(random(d, 28, 6))).value(), 1e-6f);
        same(d -> Var.of(random(d, 29, 6).abs()).xlogy(Var.of(random(d, 30, 6).abs().add(0.1f))).value(), 1e-5f);
        same(d -> Var.of(random(d, 31, 3, 7)).quantile(0.3, 1, false).value(), 1e-6f);
        same(d -> Transforms.vjp(xs -> xs.get(0).median(0, false), List.of(random(d, 32, 4, 3)),
                Tensor.full(Shape.of(3), 1f, DType.FLOAT32, d)).gradients().get(0), 1e-6f);
    }

    @Test
    void activationsLossesFft2AndSteps() {
        same(d -> Var.of(random(d, 40, 8)).silu().value(), 1e-5f);
        same(d -> Var.of(random(d, 41, 8)).gelu().value(), 1e-5f);
        same(d -> Var.of(random(d, 42, 8)).mish().value(), 1e-5f);
        same(d -> Var.of(random(d, 43, 2, 4, 3, 2)).fft2(false).value(), 1e-4f);
        same(d -> io.github.zzuegg.brainiac.nn.Loss.focalFromLogits(2).compute(Var.of(random(d, 44, 3, 4)),
                Tensor.of(Shape.of(3), new float[] {0, 3, 1}, d)).value(), 1e-5f);
        same(d -> io.github.zzuegg.brainiac.nn.Loss.generalizedCrossEntropy(0.7f).compute(Var.of(random(d, 45, 3, 4)),
                Tensor.of(Shape.of(3), new float[] {2, 1, 0}, d)).value(), 1e-5f);
        same(d -> {
            io.github.zzuegg.brainiac.nn.Recurrent lstm = (io.github.zzuegg.brainiac.nn.Recurrent)
                    io.github.zzuegg.brainiac.nn.Recurrent.lstm(4).create(Shape.of(3, 2), new SplittableRandom(46));
            lstm.parameters().forEach(p -> p.assign(p.value().to(d)));
            return lstm.step(Var.of(random(d, 47, 2, 2)), lstm.initialState(2)).h().value();
        }, 1e-5f);
    }

    @Test
    void sinkhornPlansAndGradients() {
        Function<Device, Var> points = d -> Var.param(random(d, 30, 4, 2));
        Function<Device, Var> cost = d -> OptimalTransport.squaredDistances(Var.of(random(d, 31, 4, 2)), Var.of(random(d, 32, 5, 2)));
        same(d -> OptimalTransport.sinkhorn(cost.apply(d), Tensor.of(0.25f, 0.25f, 0.25f, 0.25f).to(d),
                Tensor.of(0.2f, 0.2f, 0.2f, 0.2f, 0.2f).to(d), 0.5f, 20).value(), 1e-4f);
        same(d -> {
            Var x = points.apply(d);
            OptimalTransport.sinkhornCost(OptimalTransport.squaredDistances(x, Var.of(random(d, 33, 3, 2))),
                    Tensor.of(0.25f, 0.25f, 0.25f, 0.25f).to(d), Tensor.of(0.5f, 0.25f, 0.25f).to(d), 0.5f, 20)
                    .backward();
            return x.grad();
        }, 1e-3f);
    }

    @Test
    void decompositionGradients() {
        same(d -> {
            Var a = Var.param(random(d, 40, 4, 3));
            Var[] u = Decompositions.svd(a);
            u[0].mul(u[0]).mul(random(d, 41, 4, 3)).sum().add(u[1].sum()).backward();
            return a.grad();
        }, 1e-4f);
        same(d -> {
            Var a = Var.param(random(d, 42, 3, 3));
            Var[] e = Decompositions.eigh(a);
            e[1].mul(e[1]).mul(random(d, 43, 3, 3)).sum().add(e[0].sum()).backward();
            return a.grad();
        }, 1e-4f);
    }
}
