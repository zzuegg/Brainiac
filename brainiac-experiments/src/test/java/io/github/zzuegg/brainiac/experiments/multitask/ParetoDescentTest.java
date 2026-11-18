package io.github.zzuegg.brainiac.experiments.multitask;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.multitask.GradientBalancing.Method;
import io.github.zzuegg.brainiac.experiments.multitask.GradientBalancing.Problem;
import io.github.zzuegg.brainiac.experiments.multitask.GradientBalancing.Result;
import io.github.zzuegg.brainiac.experiments.multitask.GradientBalancing.ToyResult;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

/** Item 10.3.15: MGDA, MGDA-UB, CAGrad and Nash-MTL on the digit tasks and the toy surface. */
class ParetoDescentTest {

    @Test
    void paretoMethodsOnDigits() {
        double[] single = GradientBalancing.singleTask(Problem.DIGITS, 250, 1);
        System.out.println("DIGITS single-task " + Arrays.toString(single));
        for (Method m : new Method[] {Method.MGDA, Method.MGDA_UB, Method.CAGRAD, Method.NASH_MTL}) {
            Result r = GradientBalancing.run(Problem.DIGITS, m, 250, 1, single);
            System.out.println(r);
            assertTrue(Double.isFinite(r.deltaM()), r.toString());
        }
    }

    @Test
    void toySurface() {
        StringBuilder line = new StringBuilder("toy surface (f1, f2, starts of 5 reaching the Pareto set):");
        for (Method m : new Method[] {Method.MGDA, Method.CAGRAD, Method.NASH_MTL}) {
            ToyResult r = GradientBalancing.toySurface(m, 1500, 0.02);
            line.append(String.format(" %s %.2f %.2f %d |", r.method(), r.f1(), r.f2(), r.reached()));
            assertTrue(Double.isFinite(r.f1()) && Double.isFinite(r.f2()));
        }
        System.out.println(line);
    }
}
