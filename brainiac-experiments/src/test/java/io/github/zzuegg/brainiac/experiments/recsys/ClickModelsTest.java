package io.github.zzuegg.brainiac.experiments.recsys;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class ClickModelsTest {

    @Test
    void wideDeepAndDeepFmLearnTheFieldInteraction() {
        Map<String, BinaryBench.Result> r = ClickModels.run(500, 1);
        StringBuilder line = new StringBuilder("clicks acc/AUC/logloss:");
        r.forEach((k, v) -> line.append(String.format(" %s %.3f/%.3f/%.3f", k, v.accuracy(), v.auc(), v.logLoss())));
        System.out.println(line);
        double logistic = r.get("logistic").auc();
        assertTrue(r.get("DeepFM").auc() > logistic + 0.03, line.toString());
        assertTrue(r.get("Wide&Deep").auc() > logistic + 0.03, line.toString());
    }

    @Test
    void neuMfRanksHeldOutItems() {
        CollaborativeFiltering cf = new CollaborativeFiltering(300, 200, 1);
        CollaborativeFiltering.Score neuMf = cf.neuMf(400);
        CollaborativeFiltering.Score bpr = cf.bprMf(300);
        CollaborativeFiltering.Score pop = cf.popularity();
        System.out.printf("NeuMF recall@10 %.3f NDCG@10 %.3f | BPR-MF %.3f/%.3f | popularity %.3f/%.3f%n",
                neuMf.recall(), neuMf.ndcg(), bpr.recall(), bpr.ndcg(), pop.recall(), pop.ndcg());
        assertTrue(neuMf.recall() > 2 * pop.recall(), neuMf + " vs " + pop);
    }
}
