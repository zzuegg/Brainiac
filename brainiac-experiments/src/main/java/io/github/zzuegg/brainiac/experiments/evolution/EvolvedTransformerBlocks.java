package io.github.zzuegg.brainiac.experiments.evolution;

import io.github.zzuegg.brainiac.experimental.evolve.TransformerBlockGenome;
import io.github.zzuegg.brainiac.experimental.evolve.TransformerBlockSearch;
import io.github.zzuegg.brainiac.experimental.nn.GrowableTransformerBlock.NormPlacement;
import io.github.zzuegg.brainiac.nn.Activation;

import java.util.Collections;
import java.util.List;

/**
 * X18, evolved transformer blocks: searches heads, key/value heads, gating, norm placement,
 * depth and per-neuron FFN activations on the fold-and-add task of
 * {@link GrowingTransformers}, training each generation's candidates together as one
 * population. The search is seeded with a conventional design (pre-RMS-norm, two heads,
 * GELU FFN), so the result shows what evolution changes about it.
 */
public final class EvolvedTransformerBlocks {

    /**
     * The outcome.
     *
     * @param best the best design found, described
     * @param bestLoss its validation loss
     * @param baselineLoss the conventional design's validation loss
     * @param firstMedianLoss the median loss of the first generation
     */
    public record Outcome(String best, double bestLoss, double baselineLoss, double firstMedianLoss) {
    }

    private EvolvedTransformerBlocks() {
    }

    /**
     * Runs the search.
     *
     * @param generations generations
     * @param population candidates per generation
     * @param steps training steps per candidate
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome run(int generations, int population, int steps, long seed) {
        TransformerBlockGenome baseline = new TransformerBlockGenome(2, 2, 4, false, NormPlacement.PRE, true, 1,
                Collections.nCopies(16, Activation.GELU));
        TransformerBlockSearch search = TransformerBlockSearch.of(10, 11, 8, r -> GrowingTransformers.batch(r, 32))
                .population(population, 2).steps(steps).seed(seed);
        List<TransformerBlockSearch.Candidate> last = search.run(generations, List.of(baseline));
        TransformerBlockSearch.Candidate best = last.getFirst();
        double baselineLoss = search.trainTogether(List.of(baseline), seed).getFirst().validationLoss();
        return new Outcome(best.genome().toString(), best.validationLoss(), baselineLoss,
                search.history().getFirst().medianLoss());
    }
}
