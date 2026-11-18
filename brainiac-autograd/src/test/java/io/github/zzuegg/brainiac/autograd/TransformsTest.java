package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

import org.junit.jupiter.api.Test;

class TransformsTest {

    private static final SplittableRandom R = new SplittableRandom(1);
    private static final Tensor W = Tensor.uniform(Shape.of(3, 4), -1, 1, R);

    /** {@code tanh(x W) * x[:, :1]}: nonlinear, with a product of inputs */
    private static Var f(Var x) {
        return x.matmul(Var.of(W)).tanh().mul(x.slice(1, 0, 1));
    }

    private static void close(float[] want, float[] got, float tol) {
        assertEquals(want.length, got.length);
        for (int i = 0; i < want.length; i++) {
            assertEquals(want[i], got[i], tol, "[" + i + "]");
        }
    }

    /** Central differences of f along v. */
    private static float[] numeric(Function<Var, Var> f, Tensor x, Tensor v) {
        float h = 1e-3f;
        float[] a = f.apply(Var.of(x.add(v.mul(h)))).value().toArray();
        float[] b = f.apply(Var.of(x.sub(v.mul(h)))).value().toArray();
        float[] d = new float[a.length];
        for (int i = 0; i < d.length; i++) {
            d[i] = (a[i] - b[i]) / (2 * h);
        }
        return d;
    }

    @Test
    void jvpMatchesFiniteDifferences() {
        Tensor x = Tensor.uniform(Shape.of(2, 3), -1, 1, R);
        Tensor v = Tensor.uniform(Shape.of(2, 3), -1, 1, R);
        Transforms.Jvp j = Transforms.jvp(TransformsTest::f, x, v);
        close(f(Var.of(x)).value().toArray(), j.output().toArray(), 1e-6f);
        close(numeric(TransformsTest::f, x, v), j.tangent().toArray(), 2e-3f);
    }

    @Test
    void jacobianAgreesWithJvpAndVjp() {
        Tensor x = Tensor.uniform(Shape.of(2, 3), -1, 1, R);
        Tensor jac = Transforms.jacobian(TransformsTest::f, x);
        assertEquals(Shape.of(2, 4, 2, 3), jac.shape());
        Tensor v = Tensor.uniform(Shape.of(2, 3), -1, 1, R);
        Tensor jv = jac.reshape(8, 6).matmul(v.reshape(6, 1)).reshape(2, 4);
        close(jv.toArray(), Transforms.jvp(TransformsTest::f, x, v).tangent().toArray(), 1e-5f);
        Tensor u = Tensor.uniform(Shape.of(2, 4), -1, 1, R);
        Tensor jtu = u.reshape(1, 8).matmul(jac.reshape(8, 6)).reshape(2, 3);
        Transforms.Vjp vjp = Transforms.vjp(xs -> f(xs.get(0)), List.of(x), u);
        close(jtu.toArray(), vjp.gradients().get(0).toArray(), 1e-5f);
    }

    @Test
    void hessianVectorProduct() {
        // f(x) = sum(x^3) / 3: H = diag(2x), so H v = 2 x v
        Tensor x = Tensor.of(Shape.of(4), 1f, -2f, 0.5f, 3f);
        Tensor v = Tensor.of(Shape.of(4), 1f, 1f, 2f, -1f);
        Tensor hv = Transforms.hvp(t -> t.mul(t).mul(t).sum().div(3f), x, v);
        close(new float[] {2f, -4f, 2f, -6f}, hv.toArray(), 1e-5f);
    }

    @Test
    void vmapAndPerExampleGradients() {
        Var batch = Var.param(Tensor.uniform(Shape.of(5, 3), -1, 1, R));
        Var mapped = Transforms.vmap(x -> x.reshape(1, 3).matmul(Var.of(W)).reshape(4).tanh(), batch, 0);
        assertEquals(Shape.of(5, 4), mapped.shape());
        close(batch.value().matmul(W).tanh().toArray(), mapped.value().toArray(), 1e-6f);
        mapped.sum().backward();
        assertArrayEquals(new int[] {5, 3}, batch.grad().shape().toArray());

        Var w = Var.param(W);
        Tensor data = Tensor.uniform(Shape.of(5, 3), -1, 1, R);
        List<Tensor> per = Transforms.perExampleGradients(x -> x.matmul(w).square().sum(), List.of(w), data);
        assertEquals(Shape.of(5, 3, 4), per.get(0).shape());
        // they add up to the gradient of the whole batch
        Tensor total = Var.gradients(Var.of(data).matmul(w).square().sum(), List.of(w), false).get(0).value();
        close(total.toArray(), per.get(0).sum(0, false).toArray(), 1e-4f);
    }

    @Test
    void privateGradientsClipEachExample() {
        Var w = Var.param(W);
        Tensor data = Tensor.uniform(Shape.of(4, 3), -1, 1, R);
        Function<Var, Var> loss = x -> x.matmul(w).square().sum();
        // a loose bound clips nothing: the batch mean
        Tensor mean = Transforms.perExampleGradients(loss, List.of(w), data).get(0).sum(0, false).div(4);
        close(mean.toArray(), Transforms.privateGradients(loss, List.of(w), data, 1e6f, 0f, R).get(0).toArray(), 1e-4f);
        // one example with a tight bound comes out at exactly that norm
        Tensor one = data.slice(0, 0, 1);
        Tensor g = Transforms.privateGradients(loss, List.of(w), one, 0.25f, 0f, R).get(0);
        assertEquals(0.25f, (float) Math.sqrt(g.square().sum().item()), 1e-5f);
        // noise changes the result but keeps the shape
        Tensor noisy = Transforms.privateGradients(loss, List.of(w), data, 0.25f, 1f, new SplittableRandom(3)).get(0);
        assertEquals(W.shape(), noisy.shape());
    }

    @Test
    void lgammaGradientsAreDigammaAndTrigamma() {
        Tensor x = Tensor.of(Shape.of(3), 0.7f, 2.5f, 6f);
        Tensor v = Tensor.of(Shape.of(3), 1f, 1f, 1f);
        // d/dx lgamma = digamma; d2/dx2 = trigamma
        close(io.github.zzuegg.brainiac.tensor.Special.digamma(x).toArray(),
                Transforms.vjp(xs -> xs.get(0).lgamma(), List.of(x), v).gradients().get(0).toArray(), 1e-5f);
        close(io.github.zzuegg.brainiac.tensor.Special.trigamma(x).toArray(),
                Transforms.hvp(t -> t.lgamma().sum(), x, v).toArray(), 1e-4f);
        Var p = Var.param(Tensor.zeros(Shape.of(2, 6)));
        assertEquals(Shape.of(3, 4), p.reshape(-1, 4).shape());
        assertEquals(p.value().device(), p.device());
    }
}
