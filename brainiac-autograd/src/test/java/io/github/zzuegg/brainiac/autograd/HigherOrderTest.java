package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

import org.junit.jupiter.api.Test;

/** Gradients of gradients: Hessian-vector products, Jacobian-vector products, meta-learning. */
class HigherOrderTest {

    private static Tensor random(SplittableRandom r, int... dims) {
        return Tensor.uniform(Shape.of(dims), -1, 1, r).to(DType.FLOAT64);
    }

    @Test
    void secondDerivativeOfACube() {
        Var x = Var.param(Tensor.of(new float[] {1, -2, 0.5f}).to(DType.FLOAT64));
        Var g = Var.gradients(x.pow(3).sum(), List.of(x), true).get(0);
        assertArrayEquals(new double[] {3, 12, 0.75}, g.value().toDoubleArray(), 1e-12);
        Var h = Var.gradients(g.sum(), List.of(x), false).get(0);   // d/dx sum(3 x^2) = 6x
        assertArrayEquals(new double[] {6, -12, 3}, h.value().toDoubleArray(), 1e-12);
        assertNull(x.grad());   // gradients() leaves grad() alone
    }

    /** Hessian-vector products of a small network's loss against finite differences of its gradient. */
    @Test
    void hessianVectorProductsMatchFiniteDifferences() {
        SplittableRandom r = new SplittableRandom(4);
        Tensor input = random(r, 5, 3);
        Tensor target = random(r, 5, 2);
        Tensor w1 = random(r, 3, 4);
        Tensor b1 = random(r, 4);
        Tensor w2 = random(r, 4, 2);
        Tensor v1 = random(r, 3, 4);
        Tensor v2 = random(r, 4, 2);
        for (ActivationOp act : List.of(ActivationOp.TANH, ActivationOp.SILU, ActivationOp.SIGMOID,
                ActivationOp.GELU, ActivationOp.SOFTPLUS)) {
            Function<Tensor[], Var[]> lossAndParams = ws -> {
                Var a = Var.param(ws[0]);
                Var b = Var.param(ws[1]);
                Var c = Var.param(ws[2]);
                Var hidden = Var.of(input).matmul(a).add(b).activation(act, 0);
                Var out = hidden.matmul(c);
                // a softmax cross-entropy-like term and a squared error, to cover exp/log/div/max
                Var shifted = out.sub(out.max(1, true));
                Var logSoftmax = shifted.sub(shifted.exp().sum(1, true).log());
                Var loss = logSoftmax.mul(Var.of(target)).mean().neg().add(out.sub(Var.of(target)).square().mean());
                return new Var[] {loss, a, b, c};
            };
            Var[] f = lossAndParams.apply(new Tensor[] {w1, b1, w2});
            List<Var> grads = Var.gradients(f[0], List.of(f[1], f[2], f[3]), true);
            Var dot = grads.get(0).mul(Var.of(v1)).sum().add(grads.get(2).mul(Var.of(v2)).sum());
            List<Var> hv = Var.gradients(dot, List.of(f[1], f[2], f[3]), false);
            float eps = 1e-5f;
            Var[] plus = lossAndParams.apply(new Tensor[] {w1.add(v1.mul(eps)), b1, w2.add(v2.mul(eps))});
            Var[] minus = lossAndParams.apply(new Tensor[] {w1.sub(v1.mul(eps)), b1, w2.sub(v2.mul(eps))});
            List<Var> gp = Var.gradients(plus[0], List.of(plus[1], plus[2], plus[3]), false);
            List<Var> gm = Var.gradients(minus[0], List.of(minus[1], minus[2], minus[3]), false);
            for (int i = 0; i < 3; i++) {
                double[] fd = gp.get(i).value().sub(gm.get(i).value()).div(2 * eps).toDoubleArray();
                assertArrayEquals(fd, hv.get(i).value().toDoubleArray(), 1e-6, act + " parameter " + i);
            }
        }
    }

    /** The double-backward trick: J v is the gradient of (u -> u . J v) ... of u -> vjp(u) . v. */
    @Test
    void jacobianVectorProductThroughTwoBackwardPasses() {
        SplittableRandom r = new SplittableRandom(7);
        Tensor x0 = random(r, 4);
        Tensor m = random(r, 4, 3);
        Tensor v = random(r, 4);
        Var x = Var.param(x0);
        Var y = x.reshape(1, 4).matmul(Var.of(m)).sin().reshape(3);   // f: R^4 -> R^3
        Var u = Var.param(Tensor.zeros(Shape.of(3), DType.FLOAT64, x0.device()));
        Var vjp = Var.gradients(y, u, List.of(x), true).get(0);       // J^T u, linear in u
        Var jvp = Var.gradients(vjp.mul(Var.of(v)).sum(), List.of(u), false).get(0);   // J v
        // J v = cos(x m) * (v m)
        double[] xm = x0.reshape(1, 4).matmul(m).cos().mul(v.reshape(1, 4).matmul(m)).toDoubleArray();
        assertArrayEquals(xm, jvp.value().toDoubleArray(), 1e-12);
    }

    /** One inner gradient step, then the outer loss differentiated through it (MAML). */
    @Test
    void differentiatesThroughAnInnerUpdate() {
        Var w = Var.param(Tensor.of(new float[] {2}).to(DType.FLOAT64));
        Var inner = w.square().sum();                                   // inner loss w^2
        Var g = Var.gradients(inner, List.of(w), true).get(0);          // 2w
        Var adapted = w.sub(g.mul(0.25f));                              // w - 0.25 * 2w = w / 2
        Var outer = adapted.pow(3).sum();                               // (w/2)^3
        outer.backward();
        assertEquals(3.0 / 8 * 4, w.grad().toDoubleArray()[0], 1e-12);  // 3 w^2 / 8
    }

    @Test
    void customOpsSayTheyCannotBeDifferentiatedTwice() {
        Op twice = new Op() {
            public Tensor forward(Tensor... in) {
                return in[0].mul(2);
            }

            public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
                return new Tensor[] {grad.mul(2)};
            }
        };
        Var x = Var.param(Tensor.of(new float[] {1}));
        Var y = Var.apply(twice, x).sum();
        assertEquals(2, Var.gradients(y, List.of(x), false).get(0).value().toArray()[0]);
        assertThrows(IllegalStateException.class, () -> Var.gradients(y, List.of(x), true));
    }
}
