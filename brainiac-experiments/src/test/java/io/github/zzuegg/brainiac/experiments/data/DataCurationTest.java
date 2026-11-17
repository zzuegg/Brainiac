package io.github.zzuegg.brainiac.experiments.data;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.data.DataCuration.FactCorpus;
import io.github.zzuegg.brainiac.experiments.data.DataCuration.FactRun;
import io.github.zzuegg.brainiac.experiments.data.DataCuration.Filter;
import io.github.zzuegg.brainiac.experiments.data.DataCuration.MixRun;
import io.github.zzuegg.brainiac.experiments.data.DataCuration.Mixing;
import io.github.zzuegg.brainiac.experiments.data.DataCuration.QualityRun;
import io.github.zzuegg.brainiac.experiments.data.DataCuration.TokenSelection;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

class DataCurationTest {

    @Test
    void rho1TrainsOnTheTokensWorthLearning() {
        float full = DataCuration.selectiveTokens(TokenSelection.FULL, 0.3, 400, 1);
        float rho1 = DataCuration.selectiveTokens(TokenSelection.RHO1, 0.3, 400, 1);
        float pruned = DataCuration.selectiveTokens(TokenSelection.EL2N_PRUNED, 0.3, 400, 1);
        System.out.printf("Rho-1 on counting with 30%% noise tokens: clean loss %.3f vs full-token CE %.3f, "
                + "EL2N-pruned %.3f%n", rho1, full, pruned);
        assertTrue(rho1 < full, "selecting by excess loss beats training on every token");
    }

    @Test
    void aQualityClassifierPicksTheCleanDocuments() {
        QualityRun none = DataCuration.qualityFilter(Filter.NONE, 400, 1);
        QualityRun ppl = DataCuration.qualityFilter(Filter.PERPLEXITY, 400, 1);
        QualityRun clf = DataCuration.qualityFilter(Filter.CLASSIFIER, 400, 1);
        System.out.printf("Quality filtering (keep 50%% of a 40%%-clean pool): clean loss %.3f (%.0f%% clean kept) vs "
                + "no filter %.3f (%.0f%%), perplexity filter %.3f (%.0f%%)%n", clf.loss(), 100 * clf.cleanShare(),
                none.loss(), 100 * none.cleanShare(), ppl.loss(), 100 * ppl.cleanShare());
        assertTrue(clf.loss() < none.loss() && clf.cleanShare() > 0.5,
                "the classifier enriches clean documents, which train a better model than a random half");
    }

    @Test
    void regMixFindsTheTargetMixture() {
        MixRun uniform = DataCuration.mixture(Mixing.UNIFORM, 400, 1);
        MixRun doremi = DataCuration.mixture(Mixing.DOREMI, 400, 1);
        MixRun regmix = DataCuration.mixture(Mixing.REGMIX, 400, 1);
        System.out.printf("RegMix: target loss %.3f with %s vs uniform %.3f, DoReMi %.3f with %s%n",
                regmix.targetLoss(), round(regmix.weights()), uniform.targetLoss(), doremi.targetLoss(),
                round(doremi.weights()));
        assertTrue(regmix.targetLoss() < uniform.targetLoss() && regmix.targetLoss() < doremi.targetLoss(),
                "the regression's mixture fits the target best");
    }

    static String round(double[] w) {
        return Arrays.toString(Arrays.stream(w).map(x -> Math.round(x * 100) / 100.0).toArray());
    }

    @Test
    void entiGraphMakesFactsAnswerableInNewForms() {
        FactRun raw = DataCuration.factLearning(FactCorpus.RAW, 400, 1);
        FactRun para = DataCuration.factLearning(FactCorpus.PARAPHRASE, 400, 1);
        FactRun graph = DataCuration.factLearning(FactCorpus.ENTIGRAPH, 400, 1);
        System.out.printf("EntiGraph fact QA (forward/reversed/two-hop/overall): %.2f/%.2f/%.2f/%.2f vs raw "
                + "%.2f/%.2f/%.2f/%.2f, paraphrases %.2f/%.2f/%.2f/%.2f%n", graph.forward(), graph.reversed(),
                graph.twoHop(), graph.overall(), raw.forward(), raw.reversed(), raw.twoHop(), raw.overall(),
                para.forward(), para.reversed(), para.twoHop(), para.overall());
        assertTrue(graph.overall() > raw.overall() + 0.3 && graph.overall() > para.overall() + 0.3,
                "the entity graph's rewrites teach the facts in every form");
    }
}
