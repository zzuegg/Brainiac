package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.nn.LayerChain;
import io.github.zzuegg.brainiac.experimental.nn.LoopedTransformer;
import io.github.zzuegg.brainiac.experimental.nn.NestedFeedForward;
import io.github.zzuegg.brainiac.experimental.nn.RecurrentDepthModel;
import io.github.zzuegg.brainiac.experimental.nn.RefinedAttention;
import io.github.zzuegg.brainiac.experimental.nn.ResidualMixerBlock;
import io.github.zzuegg.brainiac.experimental.nn.UniversalTransformer;
import io.github.zzuegg.brainiac.experiments.attention.ArchitectureBench;
import io.github.zzuegg.brainiac.experiments.sequence.MixerShootout;
import io.github.zzuegg.brainiac.nn.LayerSpec;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Test-time compute by recurrent depth ({@link RecurrentDepthModel}, Huginn) on multi-hop
 * pointer chasing: the context lists a random permutation {@code f} of {@code n} symbols as
 * pairs {@code a f(a)}, then a query {@code q}; the answer is {@code f^h(q)}. One attention
 * pass can follow one pointer, so the task needs {@code h} sequential steps of retrieval,
 * the kind of reasoning that more iterations should buy.
 *
 * <p>The recurrent-depth model (token-local prelude and coda around one attention block as
 * the recurrent core, so only the core can follow pointers) is trained with a random
 * iteration count per batch and then evaluated with {@code 1, 2, 4, 8} iterations; it is
 * compared with a {@link LoopedTransformer} that halts by ACT, a {@link UniversalTransformer}
 * and a fixed stack of three attention blocks, all at width 16.
 *
 * <p>Paper: Geiping et al., "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent
 * Depth Approach", <a href="https://arxiv.org/abs/2502.05171">arXiv:2502.05171</a>.
 */
public final class RecurrentDepthScaling {

    private RecurrentDepthScaling() {
    }

    /**
     * The pointer-chasing task.
     *
     * @param symbols the permutation size {@code n}
     * @param hops the pointers to follow {@code h}
     * @return the task ({@code 2n + 1} tokens, {@code n} classes)
     */
    public static MixerShootout.Task pointerChasing(int symbols, int hops) {
        return new MixerShootout.Task("hops(" + symbols + "," + hops + ")", symbols, 2 * symbols + 1,
                (random, in) -> {
                    int[] target = new int[in.length];
                    for (int b = 0; b < in.length; b++) {
                        int[] f = permutation(random, symbols);
                        int[] order = permutation(random, symbols);
                        for (int i = 0; i < symbols; i++) {
                            in[b][2 * i] = order[i];
                            in[b][2 * i + 1] = f[order[i]];
                        }
                        int q = random.nextInt(symbols);
                        in[b][2 * symbols] = q;
                        int x = q;
                        for (int h = 0; h < hops; h++) {
                            x = f[x];
                        }
                        target[b] = x;
                    }
                    return target;
                });
    }

    private static int[] permutation(SplittableRandom random, int n) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    private static LayerSpec block() {
        return ResidualMixerBlock.of(RefinedAttention.of(2).causal(true).rotary(10000f));
    }

    /** A token-local block (no attention): the prelude and coda cannot follow pointers. */
    private static LayerSpec local() {
        return ResidualMixerBlock.of(NestedFeedForward.spec(2));
    }

    /**
     * One model's result.
     *
     * @param name the model
     * @param iterations the test-time iterations (or ticks) of each accuracy
     * @param accuracy the accuracy at each
     */
    public record Row(String name, int[] iterations, double[] accuracy) {
    }

    /**
     * Trains the recurrent-depth model and evaluates it at several iteration counts.
     *
     * @param task the task
     * @param meanIterations the mean training iterations
     * @param testIterations the iterations to evaluate
     * @param steps training steps
     * @param seed the seed
     * @return the row
     */
    public static Row recurrentDepth(MixerShootout.Task task, double meanIterations, int[] testIterations, int steps,
            long seed) {
        ArchitectureBench.TinyModel m = ArchitectureBench.trainRetrieval(RecurrentDepthModel.of(local(), block(),
                local(), meanIterations, 4), 16, task, steps, 0.01f, seed, true);
        RecurrentDepthModel model = (RecurrentDepthModel) m.body();
        double[] acc = new double[testIterations.length];
        for (int k = 0; k < acc.length; k++) {
            model.iterations(testIterations[k]);
            acc[k] = ArchitectureBench.accuracy(m, task.batch(new SplittableRandom(seed + 1000), 256));
        }
        return new Row("recurrent depth (Huginn)", testIterations.clone(), acc);
    }

    /**
     * Trains the baselines: ACT-looped block, Universal Transformer, fixed 3-block stack.
     *
     * @param task the task
     * @param maxTicks the loops' most ticks
     * @param steps training steps
     * @param seed the seed
     * @return one row each (the mean ticks used as the iteration count)
     */
    public static List<Row> baselines(MixerShootout.Task task, int maxTicks, int steps, long seed) {
        List<Row> rows = new ArrayList<>();
        ArchitectureBench.TinyModel looped = ArchitectureBench.trainRetrieval(LoopedTransformer.of(block(), maxTicks,
                LoopedTransformer.Halting.ACT), 16, task, steps, 0.01f, seed, true);
        double acc = ArchitectureBench.accuracy(looped, task.batch(new SplittableRandom(seed + 1000), 256));
        rows.add(new Row("looped (ACT)", new int[] {Math.round(((LoopedTransformer) looped.body()).meanTicks())},
                new double[] {acc}));
        rows.add(new Row("Universal Transformer", new int[] {maxTicks}, ArchitectureBench.retrieval(
                UniversalTransformer.of(2, true, maxTicks), 16, task, List.of(task), steps, 0.01f, seed, true)));
        rows.add(fixed(task, steps, seed));
        return rows;
    }

    /**
     * Trains the fixed baseline: three attention blocks.
     *
     * @param task the task
     * @param steps training steps
     * @param seed the seed
     * @return the row
     */
    public static Row fixed(MixerShootout.Task task, int steps, long seed) {
        return new Row("fixed 3 blocks", new int[] {3}, ArchitectureBench.retrieval(LayerChain.repeat(block(), 3), 16,
                task, List.of(task), steps, 0.01f, seed, true));
    }

    /**
     * Formats rows.
     *
     * @param rows the rows
     * @return a text table
     */
    public static String table(List<Row> rows) {
        StringBuilder sb = new StringBuilder();
        for (Row r : rows) {
            sb.append(String.format("%-26s", r.name()));
            for (int i = 0; i < r.accuracy().length; i++) {
                sb.append(String.format("  r=%-2d %.3f", r.iterations()[i], r.accuracy()[i]));
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    /**
     * Runs the comparison.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        MixerShootout.Task task = pointerChasing(4, 2);
        List<Row> rows = new ArrayList<>();
        rows.add(recurrentDepth(task, 4, new int[] {1, 2, 4, 8, 16}, 1200, 1));
        rows.addAll(baselines(task, 6, 1200, 1));
        System.out.print(table(rows));
    }
}
