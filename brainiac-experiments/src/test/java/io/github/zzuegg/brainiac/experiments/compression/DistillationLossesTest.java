package io.github.zzuegg.brainiac.experiments.compression;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.compress.DecoupledKdLoss;
import io.github.zzuegg.brainiac.experimental.compress.RelationalKdLoss;
import io.github.zzuegg.brainiac.experimental.nn.DistillationLoss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class DistillationLossesTest {

    @Test
    void dkdDecomposesClassicKd() {
        SplittableRandom r = new SplittableRandom(3);
        Tensor s = Tensor.normal(Shape.of(5, 4), 0, 2, r);
        Tensor t = Tensor.normal(Shape.of(5, 4), 0, 2, r);
        Tensor labels = Tensor.of(0, 1, 2, 3, 0);
        // with alpha = 1 and per-row beta = 1 - p_target, DKD equals KD; with beta = 1 it bounds it
        double kd = new DistillationLoss(1).compute(Var.of(s), t).value().item();
        double tckd = new DecoupledKdLoss(1, 1, 0).compute(Var.of(s), t, labels).value().item();
        double nckd = new DecoupledKdLoss(1, 0, 1).compute(Var.of(s), t, labels).value().item();
        assertTrue(kd >= tckd - 1e-6 && kd <= tckd + nckd + 1e-6, "KD = TCKD + (1 - p_t) NCKD: " + kd);
        assertEquals(0, new DecoupledKdLoss(2, 1, 8).compute(Var.of(t), t, labels).value().item(), 1e-5);
        assertEquals(0, new RelationalKdLoss(1, 2).compute(Var.of(t.mul(3f)), t).value().item(), 1e-5,
                "relations are scale-invariant");
    }

    @Test
    void decoupledAndRelationalDistillation() {
        List<CompactRow> rows = CompactModels.distillation(120, 120, 1);
        StringBuilder line = new StringBuilder("8.5.8 DKD/RKD vs KD/FitNet, digits acc (retrieval recall@1):");
        for (CompactRow r : rows) {
            line.append(String.format(" %s %.3f;", r.method(), r.metric()));
        }
        System.out.println(line);
        double alone = rows.get(1).metric();
        assertTrue(rows.get(3).metric() > alone - 0.05, "DKD student is not worse than the student alone");
        assertTrue(rows.get(9).metric() > rows.get(7).metric(), "RKD embedding beats an untrained one");
    }
}
