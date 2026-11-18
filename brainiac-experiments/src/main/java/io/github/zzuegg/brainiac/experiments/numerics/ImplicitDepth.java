package io.github.zzuegg.brainiac.experiments.numerics;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.DeepEquilibriumLayer;
import io.github.zzuegg.brainiac.experimental.train.AndersonSolver;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Implicit against explicit depth (wave 7, item 7.3.8): a {@link DeepEquilibriumLayer} on
 * three-arm spirals against the same weight-tied cell unrolled for a fixed number of steps
 * with backpropagation through every step (the looped / universal-transformer baseline) and a
 * plain stack of untied layers. Reports test accuracy, the forward and backward solver
 * iterations with Anderson acceleration against plain fixed-point iteration, and the activations
 * each model keeps for the backward pass.
 */
public final class ImplicitDepth {

    private ImplicitDepth() {
    }

    /**
     * One model's outcome.
     *
     * @param testAccuracy accuracy on held-out spirals
     * @param forwardIterations mean solver iterations per forward pass (the unroll depth for
     *     explicit models)
     * @param backwardIterations mean solver iterations per backward pass (0 for explicit models)
     * @param storedPerExample activation values kept for the backward pass per example
     * @param millis the training wall time
     * @param andersonAtEnd Anderson iterations to solve the trained layer on the test inputs (DEQ only)
     * @param plainAtEnd plain fixed-point iterations for the same solve (DEQ only)
     */
    public record Row(double testAccuracy, double forwardIterations, double backwardIterations, int storedPerExample,
            long millis, int andersonAtEnd, int plainAtEnd) {
    }

    private static final int MAX_ITERATIONS = 30;

    /**
     * Trains every model.
     *
     * @param hidden the state width
     * @param unroll the explicit models' depth
     * @param steps full-batch Adam steps
     * @param seed the seed
     * @return one row per model
     */
    public static Map<String, Row> run(int hidden, int unroll, int steps, long seed) {
        ToyTasks.Data train = ToyTasks.spiral(100, 3, 0.1, seed);
        ToyTasks.Data test = ToyTasks.spiral(50, 3, 0.1, seed + 1000);
        Map<String, Row> out = new LinkedHashMap<>();
        out.put("DEQ (Anderson m=5)", deq(train, test, hidden, steps, 0f, seed));
        out.put("DEQ (Anderson, Jacobian reg.)", deq(train, test, hidden, steps, 0.1f, seed));
        out.put("tied, unrolled x" + unroll, explicit(train, test, hidden, unroll, true, steps, seed));
        out.put("untied stack x" + unroll, explicit(train, test, hidden, unroll, false, steps, seed));
        return out;
    }

    private static Row deq(ToyTasks.Data train, ToyTasks.Data test, int hidden, int steps, float jacobian,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        DeepEquilibriumLayer layer = new DeepEquilibriumLayer(2, hidden, new AndersonSolver(5, MAX_ITERATIONS, 1e-4, 1.0),
                jacobian, r);
        Var head = Var.param(Tensor.normal(Shape.of(hidden, 3), 0, (float) Math.sqrt(1.0 / hidden), r));
        Var headBias = Var.param(Tensor.zeros(Shape.of(1, 3)));
        List<Var> params = new ArrayList<>(layer.parameters());
        params.add(head);
        params.add(headBias);
        Optimizer opt = Optimizer.adam(0.02f);
        double fwd = 0;
        double bwd = 0;
        long t0 = System.nanoTime();
        for (int t = 0; t < steps; t++) {
            Var logits = layer.forward(Var.of(train.inputs())).matmul(head).add(headBias);
            Var loss = Functional.crossEntropy(logits, train.ids());
            if (layer.auxiliaryLoss() != null) {
                loss = loss.add(layer.auxiliaryLoss());
            }
            loss.backward();
            fwd += layer.forwardIterations();
            bwd += layer.backwardIterations();
            opt.step(params);
        }
        long ms = (System.nanoTime() - t0) / 1_000_000;
        Tensor scores = Var.noGrad(() -> layer.forward(Var.of(test.inputs())).matmul(head).add(headBias)).value();
        // the same trained layer solved by plain fixed-point iteration, for the solver comparison
        Tensor inject = Var.noGrad(() -> Var.of(test.inputs()).matmul(layer.parameters().get(0))
                .add(layer.parameters().get(1))).value();
        Tensor w = layer.parameters().get(2).value();
        int plain = new AndersonSolver(1, 5 * MAX_ITERATIONS, 1e-4, 1.0).solve(z -> z.matmul(w).add(inject).tanh(),
                Tensor.zeros(inject.shape())).iterations();
        int anderson = new AndersonSolver(5, 5 * MAX_ITERATIONS, 1e-4, 1.0).solve(z -> z.matmul(w).add(inject).tanh(),
                Tensor.zeros(inject.shape())).iterations();
        return new Row(ToyTasks.accuracy(scores, test.y()), fwd / steps, bwd / steps, 2 * hidden, ms, anderson, plain);
    }

    private static Row explicit(ToyTasks.Data train, ToyTasks.Data test, int hidden, int depth, boolean tied,
            int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Var u = Var.param(Tensor.normal(Shape.of(2, hidden), 0, (float) Math.sqrt(0.5), r));
        Var b = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        List<Var> ws = new ArrayList<>();
        for (int k = 0; k < (tied ? 1 : depth); k++) {
            ws.add(Var.param(Tensor.normal(Shape.of(hidden, hidden), 0, (float) (tied ? 0.5 / Math.sqrt(hidden)
                    : Math.sqrt(1.0 / hidden)), r)));
        }
        Var head = Var.param(Tensor.normal(Shape.of(hidden, 3), 0, (float) Math.sqrt(1.0 / hidden), r));
        Var headBias = Var.param(Tensor.zeros(Shape.of(1, 3)));
        List<Var> params = new ArrayList<>(List.of(u, b, head, headBias));
        params.addAll(ws);
        Optimizer opt = Optimizer.adam(0.02f);
        java.util.function.Function<Tensor, Var> model = x -> {
            Var inject = Var.of(x).matmul(u).add(b);
            Var z = inject.tanh();
            for (int k = 0; k < depth; k++) {
                Var wk = ws.get(tied ? 0 : k);
                z = tied ? z.matmul(wk).add(inject).tanh() : z.matmul(wk).add(b).tanh();
            }
            return z.matmul(head).add(headBias);
        };
        long t0 = System.nanoTime();
        for (int t = 0; t < steps; t++) {
            Functional.crossEntropy(model.apply(train.inputs()), train.ids()).backward();
            opt.step(params);
        }
        long ms = (System.nanoTime() - t0) / 1_000_000;
        Tensor scores = Var.noGrad(() -> model.apply(test.inputs())).value();
        return new Row(ToyTasks.accuracy(scores, test.y()), depth, 0, (depth + 1) * hidden, ms, 0, 0);
    }
}
