package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.AttributionPatching;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Attribution patching against exact activation patching on a toy recall task. A two-block
 * decoder reads {@code x0 r1 r2 r3 r4 Q} and must output {@code x0} after the query token,
 * which needs attention from the last position back to the first. The corrupted prompts
 * change {@code x0}; the metric is the clean answer's logit minus the corrupted answer's.
 * Every node (block x attention/feed-forward x position) is scored three ways: exact
 * patching (one forward per node), AtP (two forwards and one backward) and AtP* with
 * GradDrop. The outcome gives the Spearman rank correlation of the estimated and the exact
 * effect magnitudes, whether the estimate's top three nodes are the exact top three, and the
 * wall-clock time of each.
 *
 * <p>Kramár et al. 2024, "AtP*", <a href="https://arxiv.org/abs/2403.00745">arXiv:2403.00745</a>.
 */
public final class PatchingAttribution {

    /**
     * The outcome.
     *
     * @param accuracy the decoder's recall accuracy
     * @param nodes nodes scored
     * @param atpSpearman rank correlation of |AtP| with |exact|
     * @param atpStarSpearman rank correlation of AtP* with |exact|
     * @param atpTop3 exact top-3 nodes found in AtP's top 3
     * @param atpStarTop3 the same for AtP*
     * @param exactMillis time of exact patching
     * @param atpMillis time of AtP
     * @param atpStarMillis time of AtP*
     */
    public record Outcome(double accuracy, int nodes, double atpSpearman, double atpStarSpearman, int atpTop3,
            int atpStarTop3, double exactMillis, double atpMillis, double atpStarMillis) {
    }

    private static final int VOCAB = 12;
    private static final int QUERY = 11;
    private static final int STEPS = 6;

    private PatchingAttribution() {
    }

    private static int[][] prompts(SplittableRandom r, int n) {
        int[][] p = new int[n][STEPS];
        for (int[] row : p) {
            for (int t = 0; t < STEPS - 1; t++) {
                row[t] = r.nextInt(10);
            }
            row[STEPS - 1] = QUERY;
        }
        return p;
    }

    /**
     * Runs the experiment.
     *
     * @param trainSteps decoder training steps
     * @param seed the seed
     * @return the outcome
     */
    public static Outcome run(int trainSteps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        HookedDecoder model = HookedDecoder.create(HookedDecoder.Settings.tiny(VOCAB), seed);
        Optimizer opt = Optimizer.adam(0.01f);
        for (int s = 0; s < trainSteps; s++) {
            int[][] p = prompts(r, 32);
            int[][] target = new int[32][STEPS];
            for (int b = 0; b < 32; b++) {
                Arrays.fill(target[b], -1);
                target[b][STEPS - 1] = p[b][0];
            }
            Var loss = HookedDecoder.loss(model.forward(p, null, null, true).logits(), target);
            loss.backward();
            opt.step(model.parameters());
            model.zeroGrad();
        }
        int[][] test = prompts(r, 64);
        float[] logits = model.forward(test).logits().value().toArray();
        int ok = 0;
        for (int b = 0; b < test.length; b++) {
            int base = (b * STEPS + STEPS - 1) * VOCAB;
            int best = 0;
            for (int v = 1; v < VOCAB; v++) {
                best = logits[base + v] > logits[base + best] ? v : best;
            }
            ok += best == test[b][0] ? 1 : 0;
        }
        int batch = 8;
        int[][] clean = prompts(r, batch);
        int[][] corrupt = new int[batch][];
        float[] diff = new float[batch * STEPS * VOCAB];
        for (int b = 0; b < batch; b++) {
            corrupt[b] = clean[b].clone();
            corrupt[b][0] = (clean[b][0] + 1 + r.nextInt(9)) % 10;
            int at = (b * STEPS + STEPS - 1) * VOCAB;
            diff[at + clean[b][0]] += 1f / batch;
            diff[at + corrupt[b][0]] -= 1f / batch;
        }
        Tensor weights = Tensor.of(Shape.of(batch, STEPS, VOCAB), diff);
        AttributionPatching.Metric metric = l -> l.mul(weights).sum();
        long t0 = System.nanoTime();
        double[] exact = AttributionPatching.exact(model, clean, corrupt, metric);
        long t1 = System.nanoTime();
        double[] atp = AttributionPatching.atp(model, clean, corrupt, metric);
        long t2 = System.nanoTime();
        double[] star = AttributionPatching.atpStar(model, clean, corrupt, metric);
        long t3 = System.nanoTime();
        double[] absExact = abs(exact);
        double[] absAtp = abs(atp);
        return new Outcome(ok / (double) test.length, exact.length, AttributionPatching.spearman(absAtp, absExact),
                AttributionPatching.spearman(star, absExact), topOverlap(absAtp, absExact, 3),
                topOverlap(star, absExact, 3), (t1 - t0) / 1e6, (t2 - t1) / 1e6, (t3 - t2) / 1e6);
    }

    private static double[] abs(double[] x) {
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            out[i] = Math.abs(x[i]);
        }
        return out;
    }

    private static int topOverlap(double[] a, double[] b, int k) {
        int[] ta = top(a, k);
        int[] tb = top(b, k);
        int n = 0;
        for (int i : ta) {
            for (int j : tb) {
                n += i == j ? 1 : 0;
            }
        }
        return n;
    }

    private static int[] top(double[] x, int k) {
        Integer[] idx = new Integer[x.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (p, q) -> Double.compare(x[q], x[p]));
        int[] out = new int[k];
        for (int i = 0; i < k; i++) {
            out[i] = idx[i];
        }
        return out;
    }
}
