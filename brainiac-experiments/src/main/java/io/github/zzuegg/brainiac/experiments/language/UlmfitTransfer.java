package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.nn.LoraAdapterDense;
import io.github.zzuegg.brainiac.experimental.train.DiscriminativeFineTuning;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * Wave 6 item 2.18: the ULMFiT fine-tuning recipe ({@link DiscriminativeFineTuning}) on the
 * transfer task of {@link PeftShootout}: a decoder pre-trained on counting with strides 1 and 2
 * ({@link CountingTasks#pretrained}) is fine-tuned to stride 5. Layer groups, bottom to top:
 * the embedding, the projections of each block, and the head with the norms. Compared:
 * full fine-tuning at one constant rate, ULMFiT (discriminative rates with decay 2.6, slanted
 * triangular schedule, gradual unfreezing), the same without unfreezing, and LoRA (rank 4).
 * Reported: the target loss after fine-tuning and source forgetting (the rise of the loss on
 * strides 1 and 2).
 */
public final class UlmfitTransfer {

    /** The fine-tuning method. */
    public enum Method {
        /** every parameter at one constant rate */
        FULL,
        /** discriminative rates, slanted triangular schedule, gradual unfreezing */
        ULMFIT,
        /** discriminative rates and slanted triangular schedule, no unfreezing */
        ULMFIT_NO_UNFREEZE,
        /** rank-4 LoRA on every projection */
        LORA
    }

    /**
     * Fine-tuning results.
     *
     * @param targetLoss the stride-5 loss after fine-tuning
     * @param sourceBefore the stride-1/2 loss before
     * @param sourceAfter the stride-1/2 loss after
     */
    public record Result(float targetLoss, float sourceBefore, float sourceAfter) {

        /** @return the rise of the source loss */
        public float forgetting() {
            return sourceAfter - sourceBefore;
        }
    }

    private UlmfitTransfer() {
    }

    /**
     * The layer groups of a decoder, bottom to top.
     *
     * @param m the decoder
     * @return embedding, each block's projections, then head and norms
     */
    static List<List<Var>> groups(HookedDecoder m) {
        List<List<Var>> g = new ArrayList<>();
        g.add(List.of(m.embedding()));
        Set<Var> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        seen.add(m.embedding());
        for (int i = 0; i < m.layers(); i++) {
            List<Var> block = new ArrayList<>();
            for (Map.Entry<String, Layer> e : m.linears().entrySet()) {
                if (e.getKey().startsWith("block." + i + ".")) {
                    block.addAll(e.getValue().parameters());
                }
            }
            seen.addAll(block);
            g.add(block);
        }
        List<Var> top = new ArrayList<>();
        for (Var p : m.baseParameters()) {
            if (!seen.contains(p)) {
                top.add(p);
            }
        }
        g.add(top);
        return g;
    }

    /**
     * Fine-tunes a copy of the base decoder to stride 5.
     *
     * @param base the pre-trained decoder (not changed)
     * @param method the method
     * @param steps fine-tuning steps of 16 sequences
     * @param seed the seed
     * @return the result
     */
    public static Result run(HookedDecoder base, Method method, int steps, long seed) {
        HookedDecoder m = base.copy();
        float sourceBefore = CountingTasks.evaluate(m, 12, seed + 200, 1, 2);
        List<Var> params = new ArrayList<>();
        Optimizer opt;
        float lr = 0.01f;
        switch (method) {
            case FULL -> {
                params.addAll(m.parameters());
                opt = Optimizer.adam(lr);
            }
            case ULMFIT, ULMFIT_NO_UNFREEZE -> {
                params.addAll(m.parameters());
                List<List<Var>> groups = groups(m);
                int every = method == Method.ULMFIT ? Math.max(1, steps / (2 * groups.size())) : 0;
                opt = new DiscriminativeFineTuning(groups, Optimizer::adam, 2 * lr, 2.6f, steps, 0.1f, 32f, every);
            }
            case LORA -> {
                SplittableRandom r = new SplittableRandom(seed);
                m.replaceLinears((name, l) -> {
                    LoraAdapterDense a = LoraAdapterDense.lora((Dense) l, 4, 8, r);
                    params.addAll(a.parameters());
                    return a;
                });
                opt = Optimizer.adam(lr);
            }
            default -> throw new IllegalArgumentException(method.name());
        }
        CountingTasks.train(m, params, opt, steps, 12, seed, PeftShootout.NEW_STRIDE);
        return new Result(CountingTasks.evaluate(m, 12, seed + 100, PeftShootout.NEW_STRIDE), sourceBefore,
                CountingTasks.evaluate(m, 12, seed + 200, 1, 2));
    }
}
