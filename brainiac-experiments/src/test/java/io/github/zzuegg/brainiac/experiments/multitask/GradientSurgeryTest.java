package io.github.zzuegg.brainiac.experiments.multitask;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.multitask.GradientBalancing.Method;
import io.github.zzuegg.brainiac.experiments.multitask.GradientBalancing.Problem;
import io.github.zzuegg.brainiac.experiments.multitask.GradientBalancing.Result;
import io.github.zzuegg.brainiac.experiments.multitask.GradientBalancing.ToyResult;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

/** Item 10.3.14: PCGrad, GradVac and GradOPS on the digit tasks and on the toy surface. */
class GradientSurgeryTest {

    @Test
    void surgeryOnDigits() {
        double[] single = GradientBalancing.singleTask(Problem.DIGITS, 250, 1);
        System.out.println("DIGITS single-task " + Arrays.toString(single));
        for (Method m : new Method[] {Method.EQUAL, Method.PCGRAD, Method.GRADVAC, Method.GRADOPS}) {
            Result r = GradientBalancing.run(Problem.DIGITS, m, 250, 1, single);
            System.out.println(r);
            assertTrue(r.metrics()[0] > 0.3, "class accuracy beats chance: " + r);
        }
    }

    @Test
    void toySurface() {
        StringBuilder line = new StringBuilder("toy surface (f1, f2, starts of 5 reaching the Pareto set):");
        ToyResult equal = null;
        ToyResult pc = null;
        for (Method m : new Method[] {Method.EQUAL, Method.PCGRAD, Method.GRADOPS}) {
            ToyResult r = GradientBalancing.toySurface(m, 1500, 0.02);
            line.append(String.format(" %s %.2f %.2f %d |", r.method(), r.f1(), r.f2(), r.reached()));
            equal = m == Method.EQUAL ? r : equal;
            pc = m == Method.PCGRAD ? r : pc;
        }
        System.out.println(line);
        assertTrue(Double.isFinite(pc.f1()) && Double.isFinite(equal.f1()), "the runs finish");
    }
}
