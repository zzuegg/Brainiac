package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.OnPolicyDistiller;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experimental.nn.FeatureDistillationLoss;
import io.github.zzuegg.brainiac.experiments.reasoning.ArithmeticRl;
import io.github.zzuegg.brainiac.experiments.reasoning.ArithmeticTask;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.EnumMap;
import java.util.Map;
import org.junit.jupiter.api.Test;

class DistillationShootoutTest {

    @Test
    void divergencesVanishOnIdenticalDistributionsAndAreOrdered() {
        Tensor t = Tensor.of(Shape.of(2, 3), 1f, 2f, 3f, 0f, 0f, 4f);
        Var same = Var.param(t);
        for (OnPolicyDistiller.Divergence d : OnPolicyDistiller.Divergence.values()) {
            assertEquals(0, OnPolicyDistiller.divergence(same, t, d, 0.5f).value().item(), 1e-5, d.name());
        }
        Var other = Var.param(Tensor.of(Shape.of(2, 3), 3f, 2f, 1f, 4f, 0f, 0f));
        float fwd = OnPolicyDistiller.divergence(other, t, OnPolicyDistiller.Divergence.FORWARD_KL, 0.5f).value().item();
        float jsd = OnPolicyDistiller.divergence(other, t, OnPolicyDistiller.Divergence.JSD, 0.5f).value().item();
        assertTrue(jsd > 0 && jsd < Math.log(2) && jsd < fwd, jsd + " vs " + fwd);
        // a perfect projection makes the feature loss zero
        FeatureDistillationLoss f = new FeatureDistillationLoss(2, 2, 1);
        f.projection().assign(Tensor.of(Shape.of(2, 2), 1f, 0f, 0f, 1f));
        assertEquals(0, f.compute(Var.of(t.slice(1, 0, 2)), t.slice(1, 0, 2)).value().item(), 1e-7);
    }

    @Test
    void distillationMethodsAgainstOfflineLogitKd() {
        ArithmeticTask task = ArithmeticTask.addition(9);
        WindowLm teacher = DistillationShootout.teacher(task, 128, 250, 1);
        double teacherAcc = ArithmeticRl.accuracy(teacher, task);
        Map<DistillationShootout.Method, DistillationShootout.Result> res =
                new EnumMap<>(DistillationShootout.Method.class);
        for (DistillationShootout.Method m : DistillationShootout.Method.values()) {
            DistillationShootout.Result r = DistillationShootout.run(teacher, task, m, 24, 300, 2);
            res.put(m, r);
            System.out.printf("distil %-10s student accuracy %.2f, perplexity %.3f, teacher tokens %d (teacher %.2f)%n",
                    m, r.accuracy(), r.perplexity(), r.teacherTokens(), teacherAcc);
        }
        assertTrue(teacherAcc > 0.9, "teacher " + teacherAcc);
        DistillationShootout.Result kd = res.get(DistillationShootout.Method.LOGIT_KD);
        assertTrue(kd.accuracy() > 0.3, kd.toString());
        for (DistillationShootout.Result r : res.values()) {
            assertTrue(r.accuracy() > 0.1, r.toString());
        }
    }
}
