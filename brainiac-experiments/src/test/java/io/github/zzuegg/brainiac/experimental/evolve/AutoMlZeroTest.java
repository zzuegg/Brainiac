package io.github.zzuegg.brainiac.experimental.evolve;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.evolve.AutoMlZero.Instruction;
import io.github.zzuegg.brainiac.experimental.evolve.AutoMlZero.Op;
import io.github.zzuegg.brainiac.experimental.evolve.AutoMlZero.Program;

import java.util.List;

import org.junit.jupiter.api.Test;

class AutoMlZeroTest {

    @Test
    void handWrittenSgdLearnsLinearRegression() {
        Program sgd = new Program(
                List.of(new Instruction(Op.S_CONST, 2, 0, 0, 0.05)),
                List.of(new Instruction(Op.V_DOT, 1, 1, 0, 0)),
                List.of(new Instruction(Op.S_SUB, 3, 0, 1, 0),
                        new Instruction(Op.S_MUL, 3, 3, 2, 0),
                        new Instruction(Op.SV_MUL, 2, 3, 0, 0),
                        new Instruction(Op.V_ADD, 1, 1, 2, 0)));
        List<AutoMlZero.Task> tasks = AutoMlZero.linearTasks(5, 4, 100, 20, 1);
        AutoMlZero search = AutoMlZero.of(4);
        assertTrue(search.fitness(sgd, tasks, 1) > -0.05, "sgd fitness " + search.fitness(sgd, tasks, 1));
        assertEquals(-1, search.fitness(Program.empty(), tasks, 1), 1e-9);
    }

    @Test
    void evolutionDiscoversALearningAlgorithmFromScratch() {
        AutoMlZero search = AutoMlZero.of(4).population(200, 10).seed(3)
                .searchSpace(4, 3, Op.S_ADD, Op.S_SUB, Op.S_MUL, Op.S_CONST, Op.V_ADD, Op.SV_MUL, Op.V_DOT);
        Program best = search.evolve(i -> AutoMlZero.linearClassificationTasks(4, 4, 30, 50, i), 40_000);
        double heldOut = search.fitness(best, AutoMlZero.linearClassificationTasks(20, 4, 40, 40, 99_999), 3);
        System.out.println("evolved (held-out accuracy " + (1 + heldOut) + "):\n" + best);
        assertTrue(heldOut > -0.15, "held-out error rate " + -heldOut);
    }
}
