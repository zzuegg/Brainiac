package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.DecodingPipeline;
import io.github.zzuegg.brainiac.experimental.llm.EarlyExit;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.llm.JacobiDecoding;
import io.github.zzuegg.brainiac.experimental.llm.MedusaHeads;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Lossless decoding speed-ups side by side on one model: plain greedy decoding against
 * Jacobi, lookahead and prompt-lookup decoding, Medusa heads and self-speculative early
 * exit. All of them must produce exactly greedy decoding's tokens; what differs is the
 * number of full-model passes.
 */
public final class DecodingShowdown {

    /**
     * @param tokens the generated tokens
     * @param passes the full-model passes
     */
    public record Outcome(int[] tokens, int passes) {
    }

    private DecodingShowdown() {
    }

    /**
     * Runs every method.
     *
     * @param model a trained decoder with at least two blocks
     * @param prompt the prompt
     * @param maxNew tokens to generate
     * @param medusaSteps training steps for the Medusa heads
     * @param seed the seed
     * @return the outcome of each method, greedy first
     */
    public static Map<String, Outcome> run(HookedDecoder model, int[] prompt, int maxNew, int medusaSteps, long seed) {
        Map<String, Outcome> out = new LinkedHashMap<>();
        out.put("greedy", new Outcome(DecodingPipeline.of(model).generate(prompt, maxNew, new SplittableRandom(seed)),
                maxNew));
        for (JacobiDecoding.Guess g : JacobiDecoding.Guess.values()) {
            JacobiDecoding.Result r = JacobiDecoding.generate(model, prompt, maxNew, 4, g);
            out.put(g.name().toLowerCase(java.util.Locale.ROOT), new Outcome(r.tokens(), r.modelCalls()));
        }
        MedusaHeads medusa = new MedusaHeads(model, 3, seed);
        Optimizer opt = Optimizer.adam(0.01f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < medusaSteps; s++) {
            Var loss = medusa.loss(CountingTasks.sequences(r, 16, 12, model.vocabSize(), 1, 2));
            loss.backward();
            opt.step(medusa.parameters());
            model.zeroGrad();
            for (Var p : medusa.parameters()) {
                p.zeroGrad();
            }
        }
        MedusaHeads.Result m = medusa.generate(prompt, maxNew);
        out.put("medusa", new Outcome(m.tokens(), m.forwardPasses()));
        EarlyExit.Speculation spec = EarlyExit.selfSpeculative(model, prompt, maxNew, 1, 3);
        out.put("self-speculative", new Outcome(spec.tokens(), spec.fullPasses()));
        return out;
    }
}
