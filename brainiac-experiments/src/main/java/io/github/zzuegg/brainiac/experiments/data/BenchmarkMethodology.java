package io.github.zzuegg.brainiac.experiments.data;

import io.github.zzuegg.brainiac.experimental.llm.ContaminationDetector;
import io.github.zzuegg.brainiac.experimental.llm.IrtEvaluator;
import io.github.zzuegg.brainiac.experimental.train.MachineUnlearning;
import io.github.zzuegg.brainiac.experiments.data.DataCuration.Grammar;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * How toy language models are measured, on checkpoints of {@link ContextLm} runs over a
 * Markov {@link Grammar}:
 *
 * <ul>
 *   <li>{@link #fluid}: Fluid Benchmarking ({@link IrtEvaluator}): a 2PL model fitted on
 *       the answers of many training checkpoints to 200 next-token questions (the most likely
 *       next token of distinct contexts, 15% of them mislabelled), then held-out
 *       checkpoints scored by IRT ability on 10% of the items chosen adaptively, against
 *       accuracy on all items and on a random 10%. Each score is compared with the
 *       checkpoint's true quality, its expected accuracy under the grammar.</li>
 *   <li>{@link #contamination}: half of a toy test set planted into the training data;
 *       AUROC of telling the planted half apart with Min-K%++ ({@link ContaminationDetector}),
 *       Min-K% and the loss threshold ({@link MachineUnlearning#auc}).</li>
 * </ul>
 */
@SuppressWarnings("exports")   // experimental.* gets exported when the lead integrates it
public final class BenchmarkMethodology {

    private BenchmarkMethodology() {
    }

    /** A next-token question: two context tokens and the answer the grammar produced. */
    private record Item(int a, int b, int answer) {
    }

    /**
     * The outcome of {@link #fluid}.
     *
     * @param fullTau Kendall's tau between full-set accuracy and true quality over held-out checkpoints
     * @param randomTau the same for accuracy on a random 10% of the items (mean over draws)
     * @param irtTau the same for the adaptive IRT ability on 10% of the items
     * @param irtRandomTau the same for the IRT ability on the random 10% (mean over draws)
     * @param irtFullTau the same for the IRT ability on every item
     * @param randomReversals the share of consecutive checkpoints of a run whose random-subset score drops
     * @param irtReversals the same for the adaptive IRT ability
     * @param trueReversals the same for true quality
     */
    public record FluidResult(double fullTau, double randomTau, double irtTau, double irtRandomTau,
            double irtFullTau, double randomReversals, double irtReversals, double trueReversals) {
    }

    private static int[] argmaxAll(ContextLm lm, int vocab) {
        float[] x = new float[vocab * vocab * 2 * vocab];
        for (int a = 0; a < vocab; a++) {
            for (int b = 0; b < vocab; b++) {
                int row = a * vocab + b;
                x[row * 2 * vocab + a] = 1;
                x[row * 2 * vocab + vocab + b] = 1;
            }
        }
        float[] z = lm.logits(Tensor.of(Shape.of(vocab * vocab, 2 * vocab), x)).toArray();
        int[] best = new int[vocab * vocab];
        for (int row = 0; row < best.length; row++) {
            for (int k = 1; k < vocab; k++) {
                best[row] = z[row * vocab + k] > z[row * vocab + best[row]] ? k : best[row];
            }
        }
        return best;
    }

    /** A checkpoint's argmax predictions for every context. */
    private record Checkpoint(int run, int[] predictions, double trueQuality) {
    }

    private static List<Checkpoint> checkpoints(Grammar g, int run, int hidden, float lr, int count, int every,
            long seed) {
        List<int[]> docs = g.documents(150, 16, new SplittableRandom(seed));
        ContextLm lm = new ContextLm(g.vocab(), hidden, seed + 1);
        List<Checkpoint> out = new ArrayList<>();
        for (int c = 0; c < count; c++) {
            lm.train(docs, every, 32, lr, seed + 2 + c);
            int[] pred = argmaxAll(lm, g.vocab());
            double quality = 0;
            for (int ctx = 0; ctx < pred.length; ctx++) {
                quality += pred[ctx] == mode(g, ctx) ? 1.0 / pred.length : 0;
            }
            out.add(new Checkpoint(run, pred, quality));
        }
        return out;
    }

    private static double kendall(double[] x, double[] y) {
        double s = 0;
        int pairs = 0;
        for (int i = 0; i < x.length; i++) {
            for (int j = i + 1; j < x.length; j++) {
                s += Math.signum(x[i] - x[j]) * Math.signum(y[i] - y[j]);
                pairs++;
            }
        }
        return s / pairs;
    }

    private static double reversals(List<Checkpoint> cps, double[] score) {
        int drops = 0;
        int pairs = 0;
        for (int i = 1; i < cps.size(); i++) {
            if (cps.get(i).run() == cps.get(i - 1).run()) {
                drops += score[i] < score[i - 1] ? 1 : 0;
                pairs++;
            }
        }
        return drops / (double) pairs;
    }

    /**
     * Fits IRT on six training runs (different widths and learning rates, 12 checkpoints
     * each) and evaluates two held-out runs.
     *
     * @param items the questions, at most 256 (one per distinct context)
     * @param budget the items each held-out checkpoint is asked (10% of {@code items})
     * @param seed the seed
     * @return the comparison
     */
    public static FluidResult fluid(int items, int budget, long seed) {
        Grammar g = Grammar.random(16, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        List<Item> bank = new ArrayList<>();
        int vocab = g.vocab();
        int[] contexts = r.ints(0, vocab * vocab).distinct().limit(items).toArray();
        for (int ctx : contexts) {   // the grammar's most likely next token, 15% of labels wrong
            int answer = r.nextDouble() < 0.15 ? r.nextInt(vocab) : mode(g, ctx);
            bank.add(new Item(ctx / vocab, ctx % vocab, answer));
        }
        int[] hidden = {4, 8, 16, 32, 48, 64};
        float[] rates = {3e-3f, 1e-2f, 3e-3f, 1e-2f, 3e-3f, 1e-2f};
        List<Checkpoint> fit = new ArrayList<>();
        for (int k = 0; k < hidden.length; k++) {
            fit.addAll(checkpoints(g, k, hidden[k], rates[k], 12, 20, seed + 100 * k));
        }
        List<Checkpoint> held = new ArrayList<>();
        held.addAll(checkpoints(g, 10, 24, 5e-3f, 12, 20, seed + 1000));
        held.addAll(checkpoints(g, 11, 40, 3e-3f, 12, 20, seed + 2000));
        boolean[][] responses = new boolean[fit.size()][items];
        for (int m = 0; m < fit.size(); m++) {
            for (int j = 0; j < items; j++) {
                responses[m][j] = correct(fit.get(m), bank.get(j), g.vocab());
            }
        }
        IrtEvaluator irt = IrtEvaluator.fit(responses, 3000);
        int n = held.size();
        double[] truth = new double[n];
        double[] full = new double[n];
        double[] irtScore = new double[n];
        double[] irtAll = new double[n];
        double[] firstRandom = new double[n];
        double randomTau = 0;
        double irtRandomTau = 0;
        int draws = 20;
        int[] every = new int[items];
        for (int j = 0; j < items; j++) {
            every[j] = j;
        }
        for (int i = 0; i < n; i++) {
            Checkpoint c = held.get(i);
            truth[i] = c.trueQuality();
            boolean[] answers = new boolean[items];
            int right = 0;
            for (int j = 0; j < items; j++) {
                answers[j] = correct(c, bank.get(j), g.vocab());
                right += answers[j] ? 1 : 0;
            }
            full[i] = right / (double) items;
            irtAll[i] = irt.ability(every, answers);
            irtScore[i] = irt.adaptiveAbility(j -> answers[j], budget);
        }
        SplittableRandom pick = new SplittableRandom(seed + 3);
        for (int d = 0; d < draws; d++) {
            int[] subset = pick.ints(0, items).distinct().limit(budget).toArray();
            double[] score = new double[n];
            double[] ability = new double[n];
            for (int i = 0; i < n; i++) {
                boolean[] answers = new boolean[budget];
                int right = 0;
                for (int k = 0; k < budget; k++) {
                    answers[k] = correct(held.get(i), bank.get(subset[k]), g.vocab());
                    right += answers[k] ? 1 : 0;
                }
                score[i] = right / (double) budget;
                ability[i] = irt.ability(subset, answers);
            }
            randomTau += kendall(score, truth) / draws;
            irtRandomTau += kendall(ability, truth) / draws;
            if (d == 0) {
                firstRandom = score;
            }
        }
        return new FluidResult(kendall(full, truth), randomTau, kendall(irtScore, truth), irtRandomTau,
                kendall(irtAll, truth), reversals(held, firstRandom), reversals(held, irtScore),
                reversals(held, truth));
    }

    private static Grammar flatten(Grammar g, double uniform) {
        double[][] t = new double[g.table().length][];
        for (int i = 0; i < t.length; i++) {
            t[i] = new double[g.vocab()];
            for (int k = 0; k < g.vocab(); k++) {
                t[i][k] = (1 - uniform) * g.table()[i][k] + uniform / g.vocab();
            }
        }
        return new Grammar(g.vocab(), t);
    }

    private static int mode(Grammar g, int context) {
        double[] row = g.table()[context];
        int best = 0;
        for (int k = 1; k < row.length; k++) {
            best = row[k] > row[best] ? k : best;
        }
        return best;
    }

    private static boolean correct(Checkpoint c, Item item, int vocab) {
        return c.predictions()[item.a() * vocab + item.b()] == item.answer();
    }

    /**
     * The outcome of {@link #contamination}.
     *
     * @param minKPlusPlus AUROC of Min-K%++ (k = 20%)
     * @param minK AUROC of Min-K% (k = 20%)
     * @param lossThreshold AUROC of the loss threshold (perplexity)
     */
    public record ContaminationResult(double minKPlusPlus, double minK, double lossThreshold) {
    }

    /**
     * Trains on 300 documents of two domains, one predictable and one close to random (as a
     * benchmark mixes easy and hard text), plus half of a 100-document test set from both
     * (each planted document twice), then scores both halves of the test set. A raw loss
     * mostly tells the domains apart; the planted documents are what is to be found.
     *
     * @param seed the seed
     * @return the AUROCs of detecting the planted half
     */
    public static ContaminationResult contamination(long seed) {
        Grammar sharp = Grammar.random(16, seed);
        Grammar flat = flatten(Grammar.random(16, seed + 9), 0.85);
        SplittableRandom r = new SplittableRandom(seed + 1);
        List<int[]> train = new ArrayList<>(sharp.documents(150, 24, r));
        train.addAll(flat.documents(150, 24, r));
        List<int[]> planted = new ArrayList<>(sharp.documents(25, 24, r));
        planted.addAll(flat.documents(25, 24, r));
        List<int[]> clean = new ArrayList<>(sharp.documents(25, 24, r));
        clean.addAll(flat.documents(25, 24, r));
        train.addAll(planted);
        train.addAll(planted);
        train.addAll(planted);
        ContextLm lm = new ContextLm(16, 96, seed + 2).train(train, 2500, 32, 1e-2f, seed + 3);
        double[][] plus = new double[2][50];
        double[][] mink = new double[2][50];
        double[][] loss = new double[2][50];
        for (int half = 0; half < 2; half++) {
            List<int[]> docs = half == 0 ? planted : clean;
            for (int i = 0; i < 50; i++) {
                int[] d = docs.get(i);
                Tensor logits = lm.logits(d);
                int[] targets = java.util.Arrays.copyOfRange(d, 2, d.length);
                plus[half][i] = ContaminationDetector.minKPlusPlus(logits, targets, 0.2);
                mink[half][i] = ContaminationDetector.minK(logits, targets, 0.2);
                loss[half][i] = -ContaminationDetector.logLikelihood(logits, targets);
            }
        }
        return new ContaminationResult(ContaminationDetector.auroc(plus[0], plus[1]),
                ContaminationDetector.auroc(mink[0], mink[1]), MachineUnlearning.auc(loss[0], loss[1]));
    }
}
