package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.nn.LowRankIntervention;
import io.github.zzuegg.brainiac.experimental.train.LayerSampler;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Two fine-tuning methods next to {@link PeftShootout}'s, on its task (a decoder pretrained to
 * count by 1 and 2 adapted to count by 5):
 * <ul>
 * <li>{@link #reft}: {@link LowRankIntervention} (LoReFT) on each block's feed-forward output
 *     representation, weights frozen; compared with LoRA, IA3 and BitFit by loss against
 *     trainable values;</li>
 * <li>{@link #lisa}: {@link LayerSampler} (LISA) on a four-block decoder, one or two random
 *     blocks plus embedding and head unfrozen per period; compared with full fine-tuning and
 *     LoRA by loss against peak optimizer-state values.</li>
 * </ul>
 */
public final class RepresentationTuning {

    /**
     * One run.
     *
     * @param before the new-task loss before
     * @param after the new-task loss after
     * @param trainable the values that can change over the run
     * @param peakState the most optimizer-state values held at once (Adam: two per trained value)
     */
    public record Result(float before, float after, long trainable, long peakState) {
    }

    private RepresentationTuning() {
    }

    private static long count(List<Var> vars) {
        long n = 0;
        for (Var v : vars) {
            n += v.shape().size();
        }
        return n;
    }

    /**
     * LoReFT on every block of a copy of {@code base}.
     *
     * @param base the pretrained decoder (not changed)
     * @param steps fine-tuning steps
     * @param seed the seed
     * @param rank the intervention rank
     * @return the result
     */
    public static Result reft(HookedDecoder base, int steps, long seed, int rank) {
        HookedDecoder m = base.copy();
        float before = CountingTasks.evaluate(m, 12, seed + 100, PeftShootout.NEW_STRIDE);
        List<LowRankIntervention> edits = new ArrayList<>();
        List<Var> params = new ArrayList<>();
        for (int i = 0; i < m.layers(); i++) {
            LowRankIntervention e = new LowRankIntervention(m.settings().dim(), rank, seed + i);
            m.adapters(i, null, e);
            edits.add(e);
            params.addAll(e.parameters());
        }
        Optimizer opt = Optimizer.adam(0.01f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            Var loss = m.loss(CountingTasks.sequences(r, 16, 12, m.vocabSize(), PeftShootout.NEW_STRIDE), true);
            loss.backward();
            opt.step(params);
            m.zeroGrad();
            for (Var p : params) {
                p.zeroGrad();
            }
            edits.forEach(LowRankIntervention::orthonormalize);
        }
        float after = CountingTasks.evaluate(m, 12, seed + 100, PeftShootout.NEW_STRIDE);
        return new Result(before, after, count(params), 2 * count(params));
    }

    /**
     * A four-block decoder pretrained on strides 1 and 2.
     *
     * @param steps pretraining steps
     * @param seed the seed
     * @return the decoder
     */
    public static HookedDecoder deepBase(int steps, long seed) {
        HookedDecoder m = HookedDecoder.create(HookedDecoder.Settings.tiny(12).withLayers(4), seed);
        CountingTasks.train(m, m.parameters(), Optimizer.adam(0.01f), steps, 12, seed + 1, 1, 2);
        return m;
    }

    /**
     * LISA, full fine-tuning or LoRA on a copy of {@code base}.
     *
     * @param base the pretrained decoder (not changed)
     * @param method "lisa", "full" or "lora"
     * @param active blocks unfrozen per period (LISA)
     * @param period steps per period (LISA)
     * @param steps fine-tuning steps
     * @param seed the seed
     * @return the result
     */
    public static Result lisa(HookedDecoder base, String method, int active, int period, int steps, long seed) {
        if (method.equals("lora")) {
            PeftShootout.Result p = PeftShootout.run(base, PeftShootout.Method.LORA, steps, seed);
            return new Result(p.before(), p.after(), p.trainable(), 2 * p.trainable());
        }
        HookedDecoder m = base.copy();
        float before = CountingTasks.evaluate(m, 12, seed + 100, PeftShootout.NEW_STRIDE);
        List<Var> all = m.parameters();
        Optimizer opt;
        long peak;
        if (method.equals("full")) {
            opt = Optimizer.adam(0.01f);
            peak = 2 * count(all);
        } else {
            List<List<Var>> layers = new ArrayList<>();
            for (int i = 0; i < m.layers(); i++) {
                layers.add(new ArrayList<>());
            }
            for (Map.Entry<String, Layer> e : m.linears().entrySet()) {
                int block = Integer.parseInt(e.getKey().split("\\.")[1]);
                layers.get(block).addAll(e.getValue().parameters());
            }
            List<Var> always = new ArrayList<>(List.of(m.embedding()));
            always.addAll(m.head().parameters());
            LayerSampler sampler = new LayerSampler(() -> Optimizer.adam(0.01f), layers, always, active, period, 2,
                    seed);
            opt = sampler;
            CountingTasks.train(m, all, opt, steps, 12, seed, PeftShootout.NEW_STRIDE);
            float after = CountingTasks.evaluate(m, 12, seed + 100, PeftShootout.NEW_STRIDE);
            return new Result(before, after, count(all), sampler.peakStateValues());
        }
        CountingTasks.train(m, all, opt, steps, 12, seed, PeftShootout.NEW_STRIDE);
        float after = CountingTasks.evaluate(m, 12, seed + 100, PeftShootout.NEW_STRIDE);
        return new Result(before, after, count(all), peak);
    }
}
