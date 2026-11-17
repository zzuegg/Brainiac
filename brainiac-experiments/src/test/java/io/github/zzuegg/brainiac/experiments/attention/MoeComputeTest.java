package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class MoeComputeTest {

    @Test
    void reluRoutingAndZeroComputeExpertsAdaptComputePerToken() {
        List<MoeRefinements.ComputeRow> rows = MoeRefinements.computePerToken(400, 1);
        System.out.print(MoeRefinements.computeTable(rows));
        double zero = MoeRefinements.zeroPredictorMse();
        MoeRefinements.ComputeRow topK = rows.get(0);
        MoeRefinements.ComputeRow ec = rows.get(1);
        MoeRefinements.ComputeRow remoe = rows.get(2);
        MoeRefinements.ComputeRow plus = rows.get(3);
        System.out.printf("val MSE (zero predictor %.4f) / FFN experts per token: ReMoE %.4f / %.2f (var %.2f), MoE++ "
                + "%.4f / %.2f vs top-2 %.4f / 2, expert choice %.4f / %.2f (var %.2f)%n", zero, remoe.validationMse(),
                remoe.ffnPerToken(), remoe.ffnVariance(), plus.validationMse(), plus.ffnPerToken(),
                topK.validationMse(), ec.validationMse(), ec.ffnPerToken(), ec.ffnVariance());
        assertTrue(remoe.validationMse() < 0.6 * zero);
        assertTrue(remoe.ffnVariance() > 0, "ReLU routing gives tokens different expert counts");
        assertTrue(remoe.ffnPerToken() < 4);
        assertTrue(plus.ffnPerToken() < 2);
        assertTrue(plus.validationMse() < topK.validationMse() * 1.3);
    }
}
