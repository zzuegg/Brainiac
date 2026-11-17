package io.github.zzuegg.brainiac.experiments.optim;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.optim.ClassIncrementalDigits.Method;
import io.github.zzuegg.brainiac.experiments.optim.ClassIncrementalDigits.Result;
import io.github.zzuegg.brainiac.experiments.optim.ClassIncrementalDigits.Task;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class ClassIncrementalDigitsTest {

    private static final List<Task> TASKS = ClassIncrementalDigits.split(60, 1);
    private static final int EPOCHS = 15;
    private static final int MEMORY = 100;

    private static Map<Method, Result> run(Method... methods) {
        Map<Method, Result> out = new EnumMap<>(Method.class);
        for (Method m : methods) {
            out.put(m, ClassIncrementalDigits.run(m, TASKS, EPOCHS, MEMORY, 1));
        }
        return out;
    }

    private static double cil(Map<Method, Result> r, Method m) {
        return r.get(m).classIncremental().averageAccuracy();
    }

    @Test
    void distillationAndReplayLimitClassIncrementalForgetting() {
        Map<Method, Result> r = run(Method.FINETUNE, Method.EWC, Method.SI, Method.REPLAY, Method.LWF, Method.ICARL,
                Method.DER_PLUS_PLUS);
        System.out.printf("split digits class-IL avg acc: LwF %.3f, iCaRL %.3f, DER++ %.3f | finetune %.3f, EWC %.3f, "
                + "SI %.3f, reservoir replay %.3f (forgetting: iCaRL %.3f, DER++ %.3f, finetune %.3f)%n",
                cil(r, Method.LWF), cil(r, Method.ICARL), cil(r, Method.DER_PLUS_PLUS), cil(r, Method.FINETUNE),
                cil(r, Method.EWC), cil(r, Method.SI), cil(r, Method.REPLAY),
                r.get(Method.ICARL).classIncremental().averageForgetting(),
                r.get(Method.DER_PLUS_PLUS).classIncremental().averageForgetting(),
                r.get(Method.FINETUNE).classIncremental().averageForgetting());
        assertTrue(cil(r, Method.ICARL) > cil(r, Method.FINETUNE) + 0.15, "iCaRL beats fine-tuning");
        assertTrue(cil(r, Method.DER_PLUS_PLUS) > cil(r, Method.FINETUNE) + 0.15, "DER++ beats fine-tuning");
        assertTrue(cil(r, Method.DER_PLUS_PLUS) > cil(r, Method.EWC), "DER++ beats EWC in class-IL");
    }

    @Test
    void gradientEpisodicMemoryProjectsAwayInterference() {
        Map<Method, Result> r = run(Method.FINETUNE, Method.EWC, Method.REPLAY, Method.AGEM, Method.GEM);
        System.out.printf("split digits GEM %.3f (BWT %.3f, %.2f ms/step), A-GEM %.3f (BWT %.3f, %.2f ms/step) | "
                + "replay %.3f (BWT %.3f, %.2f ms/step), EWC %.3f (BWT %.3f), finetune %.3f (BWT %.3f)%n",
                cil(r, Method.GEM), r.get(Method.GEM).classIncremental().backwardTransfer(),
                r.get(Method.GEM).millisPerStep(), cil(r, Method.AGEM),
                r.get(Method.AGEM).classIncremental().backwardTransfer(), r.get(Method.AGEM).millisPerStep(),
                cil(r, Method.REPLAY), r.get(Method.REPLAY).classIncremental().backwardTransfer(),
                r.get(Method.REPLAY).millisPerStep(), cil(r, Method.EWC),
                r.get(Method.EWC).classIncremental().backwardTransfer(), cil(r, Method.FINETUNE),
                r.get(Method.FINETUNE).classIncremental().backwardTransfer());
        double bwt = r.get(Method.FINETUNE).classIncremental().backwardTransfer();
        assertTrue(r.get(Method.GEM).classIncremental().backwardTransfer() > bwt, "GEM forgets less");
        assertTrue(r.get(Method.AGEM).classIncremental().backwardTransfer() > bwt, "A-GEM forgets less");
        assertTrue(cil(r, Method.GEM) > cil(r, Method.FINETUNE), "GEM beats fine-tuning");
    }

    @Test
    void parameterIsolationDoesNotForget() {
        Map<Method, Result> r = run(Method.FINETUNE, Method.EWC, Method.PACKNET, Method.HAT, Method.GROW_FROZEN);
        System.out.printf("split digits task-IL avg acc / forgetting / capacity per task: PackNet %.3f/%.3f/%.2f, "
                + "HAT %.3f/%.3f/%.2f | progressive %.3f/%.3f/%.2f, EWC %.3f/%.3f, finetune %.3f/%.3f%n",
                til(r, Method.PACKNET), forget(r, Method.PACKNET), r.get(Method.PACKNET).capacityPerTask(),
                til(r, Method.HAT), forget(r, Method.HAT), r.get(Method.HAT).capacityPerTask(),
                til(r, Method.GROW_FROZEN), forget(r, Method.GROW_FROZEN), r.get(Method.GROW_FROZEN).capacityPerTask(),
                til(r, Method.EWC), forget(r, Method.EWC), til(r, Method.FINETUNE), forget(r, Method.FINETUNE));
        assertTrue(forget(r, Method.PACKNET) < 0.02, "PackNet freezes old tasks: " + forget(r, Method.PACKNET));
        assertTrue(forget(r, Method.HAT) < forget(r, Method.FINETUNE), "HAT forgets less than fine-tuning");
        assertTrue(til(r, Method.PACKNET) > 0.8, "PackNet still learns every task");
    }

    private static double til(Map<Method, Result> r, Method m) {
        return r.get(m).taskIncremental().averageAccuracy();
    }

    private static double forget(Map<Method, Result> r, Method m) {
        return r.get(m).taskIncremental().averageForgetting();
    }
}
