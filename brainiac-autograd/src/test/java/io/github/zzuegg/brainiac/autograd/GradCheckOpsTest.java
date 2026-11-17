package io.github.zzuegg.brainiac.autograd;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.stream.Stream;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/** Every differentiable operation, checked against finite differences. */
class GradCheckOpsTest {

    private static Tensor rand(long seed, int... dims) {
        return Tensor.uniform(Shape.of(dims), -2, 2, new SplittableRandom(seed));
    }

    /** Values kept away from 0 so log, sqrt, div and kinks at 0 are smooth. */
    private static Tensor positive(long seed, int... dims) {
        return Tensor.uniform(Shape.of(dims), 0.5f, 2, new SplittableRandom(seed));
    }

    private static Arguments op(String name, Function<List<Var>, Var> f, Tensor... inputs) {
        return Arguments.of(name, f, List.of(inputs));
    }

    static Stream<Arguments> ops() {
        Tensor a = rand(1, 2, 3);
        Tensor b = rand(2, 2, 3);
        Tensor row = rand(3, 3);
        Tensor col = rand(4, 2, 1);
        Tensor p = positive(5, 2, 3);
        return Stream.of(
                op("add", v -> v.get(0).add(v.get(1)).square().sum(), a, b),
                op("add broadcast row", v -> v.get(0).add(v.get(1)).square().sum(), a, row),
                op("add broadcast column", v -> v.get(1).add(v.get(0)).square().sum(), a, col),
                op("add constant", v -> v.get(0).add(3).square().sum(), a),
                op("sub", v -> v.get(0).sub(v.get(1)).square().sum(), a, row),
                op("sub constant", v -> v.get(0).sub(1).square().sum(), a),
                op("mul", v -> v.get(0).mul(v.get(1)).sum(), a, b),
                op("mul broadcast", v -> v.get(0).mul(v.get(1)).sum(), a, col),
                op("mul constant", v -> v.get(0).mul(-1.5f).square().sum(), a),
                op("div", v -> v.get(0).div(v.get(1)).sum(), a, p),
                op("div broadcast", v -> v.get(0).div(v.get(1)).sum(), a, positive(6, 3)),
                op("div constant", v -> v.get(0).div(4).square().sum(), a),
                op("pow", v -> v.get(0).pow(3).sum(), p),
                op("neg", v -> v.get(0).neg().square().sum(), a),
                op("square", v -> v.get(0).square().sum(), a),
                op("sqrt", v -> v.get(0).sqrt().sum(), p),
                op("exp", v -> v.get(0).exp().sum(), a),
                op("log", v -> v.get(0).log().sum(), p),
                op("abs", v -> v.get(0).abs().sum(), p.mul(-1)),
                op("tanh", v -> v.get(0).tanh().sum(), a),
                op("sigmoid", v -> v.get(0).sigmoid().sum(), a),
                op("relu", v -> v.get(0).relu().sum(), p.sub(1.25f).mul(2).add(0.3f)),
                op("matmul", v -> v.get(0).matmul(v.get(1)).square().sum(), a, rand(7, 3, 4)),
                op("transpose", v -> v.get(0).transpose().matmul(v.get(1)).sum(), a, rand(8, 2, 2)),
                op("reshape", v -> v.get(0).reshape(3, 2).matmul(v.get(1)).sum(), a, rand(9, 2, 2)),
                op("sum", v -> v.get(0).sum().square(), a),
                op("sum axis 0", v -> v.get(0).sum(0, false).square().sum(), a),
                op("sum axis 1 keep", v -> v.get(0).sum(1, true).square().sum(), a),
                op("mean", v -> v.get(0).mean().square(), a),
                op("mean axis", v -> v.get(0).mean(-1, false).square().sum(), a),
                op("max axis", v -> v.get(0).max(1, false).square().sum(), a),
                op("max axis keep", v -> v.get(0).max(0, true).square().sum(), a),
                op("erf", v -> v.get(0).erf().sum(), a),
                op("softplus", v -> v.get(0).softplus().sum(), a),
                op("sin", v -> v.get(0).mul(2).sin().sum(), a),
                op("cumsum", v -> v.get(0).cumsum(1).square().sum(), a),
                op("cumsum axis 0", v -> v.get(0).cumsum(0).square().sum(), a),
                op("top 2", v -> v.get(0).topk(2).square().sum(), a),
                op("scatterAdd", v -> v.get(0).scatterAdd(1, Tensor.of(2, 0, 2), 4).square().sum(), a),
                op("cos", v -> v.get(0).mul(2).cos().square().sum(), a),
                op("maximum", v -> v.get(0).maximum(v.get(1)).square().sum(), a, b),
                op("maximum broadcast", v -> v.get(0).maximum(v.get(1)).square().sum(), a, row),
                op("minimum", v -> v.get(0).minimum(v.get(1)).square().sum(), a, b),
                op("minimum broadcast", v -> v.get(1).minimum(v.get(0)).square().sum(), a, col),
                op("maximum constant", v -> v.get(0).maximum(0.1f).square().sum(), a),
                op("minimum constant", v -> v.get(0).minimum(0.1f).square().sum(), a),
                op("clamp", v -> v.get(0).clamp(-1.05f, 0.95f).square().sum(), a),
                op("where", v -> Var.where(a.greater(b), v.get(0).square(), v.get(1).exp()).sum(), a, b),
                op("where broadcast", v -> Var.where(a.greater(Tensor.scalar(0)), v.get(0), v.get(1)).square().sum(),
                        a, row),
                op("reused value", v -> v.get(0).mul(v.get(0)).add(v.get(0)).sum(), a),
                op("fft", v -> v.get(0).fft(false).mul(v.get(1)).sum(), rand(11, 2, 6, 2), rand(12, 2, 6, 2)),
                op("inverse fft", v -> v.get(0).fft(true).mul(v.get(1)).sum(), rand(13, 5, 2), rand(14, 5, 2)),
                op("rfft", v -> v.get(0).rfft().mul(v.get(1)).sum(), rand(15, 2, 8), rand(16, 2, 5, 2)),
                op("irfft", v -> v.get(0).irfft(7).mul(v.get(1)).sum(), rand(17, 4, 2), rand(18, 7)),
                op("dense layer", v -> v.get(0).matmul(v.get(1)).add(v.get(2)).tanh().sub(v.get(3)).square().mean(),
                        rand(10, 4, 2), rand(11, 2, 8), rand(12, 8), rand(13, 4, 8)));
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("ops")
    void gradientMatchesFiniteDifferences(String name, Function<List<Var>, Var> f, List<Tensor> inputs) {
        GradCheck.Result result = GradCheck.check(f, inputs);
        assertTrue(result.passed(), name + ": " + result);
    }
}
