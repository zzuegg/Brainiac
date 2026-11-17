package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.llm.SoftPrompts;
import io.github.zzuegg.brainiac.experimental.nn.HoulsbyAdapter;
import io.github.zzuegg.brainiac.experimental.nn.Ia3Scaling;
import io.github.zzuegg.brainiac.experimental.nn.LoraAdapterDense;
import io.github.zzuegg.brainiac.experimental.nn.VeraDense;
import io.github.zzuegg.brainiac.experimental.train.BitFit;
import io.github.zzuegg.brainiac.experimental.train.LoraPlus;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Parameter-efficient fine-tuning side by side: a decoder pretrained to count by 1 and 2 is
 * adapted to count by 5 with each method, training only that method's values, and the
 * new-task loss before and after is compared with the number of values trained.
 */
public final class PeftShootout {

    /** The fine-tuning methods compared. */
    public enum Method {
        /** every weight */
        FULL,
        /** LoRA on the attention and feed-forward projections */
        LORA,
        /** LoRA with a 16x learning rate on B */
        LORA_PLUS,
        /** LoRA started from the principal singular directions */
        PISSA,
        /** shared frozen random low-rank matrices, trained scaling vectors */
        VERA,
        /** learned rescaling of keys, values and the feed-forward activation */
        IA3,
        /** bottleneck adapters after attention and feed-forward */
        HOULSBY,
        /** only the biases (and norm parameters) */
        BITFIT,
        /** learned input embeddings */
        PROMPT_TUNING,
        /** learned input embeddings from an MLP encoder */
        P_TUNING,
        /** learned keys and values at every layer */
        PREFIX_TUNING
    }

    /**
     * @param method the method
     * @param trainable the values trained
     * @param before the new-task loss before fine-tuning
     * @param after the new-task loss after
     */
    public record Result(Method method, long trainable, float before, float after) {
    }

    private PeftShootout() {
    }

    /** The stride of the new task. */
    public static final int NEW_STRIDE = 5;

    /**
     * Adapts a copy of {@code base} with one method.
     *
     * @param base the pretrained decoder (not changed)
     * @param method the method
     * @param steps fine-tuning steps
     * @param seed the seed
     * @return the result
     */
    public static Result run(HookedDecoder base, Method method, int steps, long seed) {
        HookedDecoder m = base.copy();
        float before = CountingTasks.evaluate(m, 12, seed + 100, NEW_STRIDE);
        SplittableRandom r = new SplittableRandom(seed);
        float lr = 0.01f;
        List<Var> params = new ArrayList<>();
        Optimizer opt = null;
        switch (method) {
            case FULL -> params.addAll(m.parameters());
            case LORA, LORA_PLUS -> {
                List<Var> bs = new ArrayList<>();
                m.replaceLinears((name, l) -> {
                    LoraAdapterDense a = LoraAdapterDense.lora((Dense) l, 4, 8, r);
                    params.addAll(a.parameters());
                    bs.add(a.b());
                    return a;
                });
                if (method == Method.LORA_PLUS) {
                    opt = LoraPlus.optimizer(bs, lr, 16, Optimizer::adam);
                }
            }
            case PISSA -> m.replaceLinears((name, l) -> {
                LoraAdapterDense a = LoraAdapterDense.pissa((Dense) l, 4);
                params.addAll(a.parameters());
                return a;
            });
            case VERA -> {
                int ffn = m.settings().dim() * m.settings().ffnMultiplier();
                VeraDense.Shared shared = new VeraDense.Shared(ffn, ffn, 32, seed);
                m.replaceLinears((name, l) -> {
                    VeraDense v = shared.adapt((Dense) l);
                    params.addAll(v.parameters());
                    return v;
                });
                lr = 0.03f;
            }
            case IA3 -> {
                m.replaceLinears((name, l) -> {
                    if (name.endsWith("attn.k") || name.endsWith("attn.v") || name.endsWith("ffn.up")) {
                        Ia3Scaling s = new Ia3Scaling(l, ((Dense) l).weight().shape().dim(1));
                        params.addAll(s.parameters());
                        return s;
                    }
                    return l;
                });
                lr = 0.03f;
            }
            case HOULSBY -> {
                for (int i = 0; i < m.layers(); i++) {
                    Layer a = new HoulsbyAdapter(m.settings().dim(), 4, Activation.GELU, r);
                    Layer f = new HoulsbyAdapter(m.settings().dim(), 4, Activation.GELU, r);
                    m.adapters(i, a, f);
                    params.addAll(a.parameters());
                    params.addAll(f.parameters());
                }
            }
            case BITFIT -> {
                params.addAll(BitFit.biases(m.parameters()));
                lr = 0.03f;
            }
            case PROMPT_TUNING -> {
                params.addAll(SoftPrompts.promptTuning(m, 4, seed));
                lr = 0.03f;
            }
            case P_TUNING -> params.addAll(SoftPrompts.pTuning(m, 4, 32, seed));
            case PREFIX_TUNING -> {
                params.addAll(SoftPrompts.prefixTuning(m, 4, seed));
                lr = 0.05f;
            }
            default -> throw new IllegalArgumentException(method.name());
        }
        if (opt == null) {
            opt = Optimizer.adam(lr);
        }
        CountingTasks.train(m, params, opt, steps, 12, seed, NEW_STRIDE);
        float after = CountingTasks.evaluate(m, 12, seed + 100, NEW_STRIDE);
        long count = 0;
        for (Var p : params) {
            count += p.shape().size();
        }
        return new Result(method, count, before, after);
    }
}
