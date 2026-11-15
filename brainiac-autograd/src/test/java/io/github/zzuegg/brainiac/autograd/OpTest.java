package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class OpTest {

    private static final Tensor X = Tensor.uniform(Shape.of(3, 4), -2, 2, new SplittableRandom(1));
    private static final Tensor Y = Tensor.uniform(Shape.of(3, 4), -2, 2, new SplittableRandom(2));

    /** {@code x^3}, with the right gradient. */
    static final Op CUBE = new Op() {
        @Override
        public Tensor forward(Tensor... in) {
            return in[0].mul(in[0]).mul(in[0]);
        }

        @Override
        public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
            return new Tensor[] {grad.mul(in[0].square()).mul(3)};
        }

        @Override
        public String name() {
            return "cube";
        }
    };

    /** {@code a * b}, a two-input op. */
    static final Op PRODUCT = new Op() {
        @Override
        public Tensor forward(Tensor... in) {
            return in[0].mul(in[1]);
        }

        @Override
        public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
            return new Tensor[] {grad.mul(in[1]), grad.mul(in[0])};
        }
    };

    @Test
    void correctGradientsPass() {
        GradCheck.Result r = GradCheck.check(CUBE, X);
        assertTrue(r.passed(), r.toString());
        r = GradCheck.check(PRODUCT, X, Y);
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void wrongGradientsAreCaught() {
        Op wrong = new Op() {
            @Override
            public Tensor forward(Tensor... in) {
                return in[0].mul(in[0]).mul(in[0]);
            }

            @Override
            public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
                return new Tensor[] {grad.mul(in[0].square()).mul(2)};
            }
        };
        assertFalse(GradCheck.check(wrong, X).passed());
    }

    @Test
    void composesWithOrdinaryOperations() {
        GradCheck.Result r = GradCheck.check(v -> Var.apply(CUBE, v.get(0).tanh()).add(v.get(1)).square().sum(),
                java.util.List.of(X, Y));
        assertTrue(r.passed(), r.toString());
    }

    @Test
    void straightThroughEstimator() {
        // forward rounds to -1/0/1, backward pretends it was the identity
        Op sign = new Op() {
            @Override
            public Tensor forward(Tensor... in) {
                return in[0].sign();
            }

            @Override
            public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
                return new Tensor[] {grad};
            }
        };
        Var x = Var.param(Tensor.of(-0.3f, 0.7f));
        Var y = Var.apply(sign, x);
        y.mul(Var.of(Tensor.of(2f, 5f))).sum().backward();
        assertArrayEquals(new float[] {-1f, 1f}, y.value().toArray());
        assertArrayEquals(new float[] {2f, 5f}, x.grad().toArray());
    }

    @Test
    void nullMeansNoGradient() {
        Op firstOnly = new Op() {
            @Override
            public Tensor forward(Tensor... in) {
                return in[0].add(in[1]);
            }

            @Override
            public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
                return new Tensor[] {grad, null};
            }
        };
        Var a = Var.param(Tensor.of(1f, 2f));
        Var b = Var.param(Tensor.of(3f, 4f));
        Var.apply(firstOnly, a, b).sum().backward();
        assertArrayEquals(new float[] {1f, 1f}, a.grad().toArray());
        assertArrayEquals(new float[] {0f, 0f}, b.grad().toArray());
    }

    @Test
    void badGradientsGiveClearErrors() {
        Op badShape = new Op() {
            @Override
            public Tensor forward(Tensor... in) {
                return in[0];
            }

            @Override
            public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
                return new Tensor[] {grad.sum()};
            }

            @Override
            public String name() {
                return "bad-shape";
            }
        };
        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> Var.apply(badShape, Var.param(X)).sum().backward());
        assertTrue(e.getMessage().contains("bad-shape"), e.getMessage());

        Op tooFew = new Op() {
            @Override
            public Tensor forward(Tensor... in) {
                return in[0].add(in[1]);
            }

            @Override
            public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
                return new Tensor[] {grad};
            }
        };
        assertThrows(IllegalStateException.class,
                () -> Var.apply(tooFew, Var.param(X), Var.param(Y)).sum().backward());
    }

    @Test
    void constantInputsNeedNoBackward() {
        Var y = Var.apply(CUBE, Var.of(Tensor.of(2f)));
        assertFalse(y.requiresGrad());
        assertArrayEquals(new float[] {8f}, y.value().toArray());
    }
}
