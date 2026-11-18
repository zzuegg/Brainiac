package io.github.zzuegg.brainiac.experiments.alignment;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.experimental.align.PreferenceEmbeddingModel;
import io.github.zzuegg.brainiac.experimental.align.SelfPlayPreference;
import io.github.zzuegg.brainiac.experimental.llm.DpoTrainer;
import io.github.zzuegg.brainiac.experimental.llm.PpoLmTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.experiments.reasoning.ArithmeticTask;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Wave 9, 9.2.3 and 9.2.4: preferences that no scalar reward can represent. A response is a
 * style token (rock, paper or scissors: {@code 0}, {@code 1}, {@code 2}), a content token
 * ({@code 3} to {@code 6}) and the end token. Three annotator types each hold a transitive
 * order of the styles, {@code 0 > 1 > 2}, {@code 1 > 2 > 0} and {@code 2 > 0 > 1}, and break
 * style ties by the larger content token; a preference label comes from a random annotator, so
 * the majority preference is cyclic ({@code 0 > 1}, {@code 1 > 2} and {@code 2 > 0}, each by
 * two to one). In the transitive control every annotator is of the first type.
 *
 * <p>{@link #models} (9.2.4) fits a {@link PreferenceEmbeddingModel} (general preference
 * model), a Bradley-Terry model and a pairwise classifier with the same network on preference
 * pairs and measures held-out accuracy against the majority and on the cyclic triples.
 * {@link #policies} (9.2.3) tunes the same start with SPPO and Nash-MD
 * ({@link SelfPlayPreference}) on a learned preference model, and with offline DPO, iterative
 * (online) DPO and Bradley-Terry reward plus PPO, and scores each policy's win rate against a
 * fixed panel (always rock, always paper, always scissors, uniform) under the true preference.
 */
public final class GeneralPreferences {

    private static final CharacterTokenizer TOK = new CharacterTokenizer(ArithmeticTask.CHARACTERS);
    private static final int EOS = TOK.encode("#")[0];
    private static final int[] PROMPT = TOK.encode("?=");
    private static final int STYLES = 3;
    private static final int CONTENTS = 4;
    private static final int FEATURES = STYLES + CONTENTS;

    /**
     * 9.2.4's accuracies.
     *
     * @param heldOut accuracy on held-out pairs against the majority preference
     * @param cyclicTriples mean accuracy over the three pairs of each cyclic style triple
     */
    public record ModelRow(double heldOut, double cyclicTriples) {
    }

    /**
     * 9.2.3's scores of one tuned policy.
     *
     * @param worstCase the lowest win rate against the panel
     * @param styles the policy's probability of each style
     * @param panel win rate against always 0, always 1, always 2 and uniform
     */
    public record PolicyRow(double worstCase, double[] styles, double[] panel) {
    }

    private GeneralPreferences() {
    }

    /**
     * The true preference between two responses: the share of annotator types preferring the
     * first.
     *
     * @param a a response
     * @param b another
     * @param cyclic whether the annotators are cyclic (otherwise all are of the first type)
     * @return {@code P(a > b)}
     */
    public static double truePreference(int[] a, int[] b, boolean cyclic) {
        int sa = style(a);
        int sb = style(b);
        if (sa < 0 || sb < 0) {
            return sa < 0 && sb < 0 ? 0.5 : sa < 0 ? 0 : 1;
        }
        if (sa == sb) {
            int ca = content(a);
            int cb = content(b);
            return ca == cb ? 0.5 : ca > cb ? 1 : 0;
        }
        int types = cyclic ? 3 : 1;
        double p = 0;
        for (int t = 0; t < types; t++) {
            p += rank(t, sa) < rank(t, sb) ? 1 : 0;
        }
        return p / types;
    }

    /** Position of style {@code s} in annotator type {@code t}'s order (0 is best). */
    private static int rank(int t, int s) {
        return Math.floorMod(s - t, STYLES);
    }

    private static double styleWin(int sa, int sb, boolean cyclic) {
        if (sa == sb) {
            return 0.5;
        }
        int types = cyclic ? 3 : 1;
        double p = 0;
        for (int t = 0; t < types; t++) {
            p += rank(t, sa) < rank(t, sb) ? 1 : 0;
        }
        return p / types;
    }

    static int style(int[] r) {
        return r.length >= 1 && r[0] < STYLES ? r[0] : -1;
    }

    static int content(int[] r) {
        return r.length >= 2 && r[1] >= STYLES && r[1] < STYLES + CONTENTS ? r[1] - STYLES : -1;
    }

    /**
     * @param r a response
     * @return one-hot style and content (zeros where malformed)
     */
    public static float[] features(int[] r) {
        float[] f = new float[FEATURES];
        int s = style(r);
        int c = content(r);
        if (s >= 0) {
            f[s] = 1;
        }
        if (c >= 0) {
            f[STYLES + c] = 1;
        }
        return f;
    }

    private static int[] randomResponse(SplittableRandom r) {
        return new int[] {r.nextInt(STYLES), STYLES + r.nextInt(CONTENTS), EOS};
    }

    /** A labelled pair: one random annotator's choice between two random responses. */
    private static int[][] labelledPair(SplittableRandom r, boolean cyclic) {
        int[] a = randomResponse(r);
        int[] b = randomResponse(r);
        while (a[0] == b[0] && a[1] == b[1]) {
            b = randomResponse(r);
        }
        int type = cyclic ? r.nextInt(3) : 0;
        boolean aWins = a[0] == b[0] ? a[1] > b[1] : rank(type, a[0]) < rank(type, b[0]);
        return aWins ? new int[][] {a, b} : new int[][] {b, a};
    }

    private static PreferenceEmbeddingModel fit(PreferenceEmbeddingModel m, int pairs, int steps, boolean cyclic,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] w = new float[pairs][];
        float[][] l = new float[pairs][];
        for (int i = 0; i < pairs; i++) {
            int[][] p = labelledPair(r, cyclic);
            w[i] = features(p[0]);
            l[i] = features(p[1]);
        }
        for (int s = 0; s < steps; s++) {
            int b = Math.min(128, pairs);
            float[][] bw = new float[b][];
            float[][] bl = new float[b][];
            for (int i = 0; i < b; i++) {
                int j = r.nextInt(pairs);
                bw[i] = w[j];
                bl[i] = l[j];
            }
            m.step(bw, bl);
        }
        return m;
    }

    /**
     * Fits the three preference models (9.2.4).
     *
     * @param cyclic cyclic or transitive annotators
     * @param pairs training pairs
     * @param steps optimizer steps
     * @param seed the seed
     * @return model name to its accuracies
     */
    public static Map<String, ModelRow> models(boolean cyclic, int pairs, int steps, long seed) {
        Map<String, PreferenceEmbeddingModel> ms = new LinkedHashMap<>();
        ms.put("GPM (preference embeddings)", PreferenceEmbeddingModel.embedding(FEATURES, 32, 2, seed));
        ms.put("Bradley-Terry", PreferenceEmbeddingModel.bradleyTerry(FEATURES, 32, seed));
        ms.put("pairwise classifier", PreferenceEmbeddingModel.pairwise(FEATURES, 32, seed));
        Map<String, ModelRow> out = new LinkedHashMap<>();
        for (Map.Entry<String, PreferenceEmbeddingModel> e : ms.entrySet()) {
            fit(e.getValue(), pairs, steps, cyclic, seed + 1);
            out.put(e.getKey(), evaluate(e.getValue(), cyclic, seed + 2));
        }
        return out;
    }

    private static ModelRow evaluate(PreferenceEmbeddingModel m, boolean cyclic, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> w = new ArrayList<>();
        List<float[]> l = new ArrayList<>();
        while (w.size() < 512) {
            int[] a = randomResponse(r);
            int[] b = randomResponse(r);
            double p = truePreference(a, b, cyclic);
            if (p != 0.5) {
                w.add(features(p > 0.5 ? a : b));
                l.add(features(p > 0.5 ? b : a));
            }
        }
        double held = m.accuracy(w.toArray(float[][]::new), l.toArray(float[][]::new));
        // the cyclic triples: one response of each style with equal content
        double triples = 0;
        for (int c = 0; c < CONTENTS; c++) {
            float[][] x = new float[STYLES][];
            int[][] resp = new int[STYLES][];
            for (int s = 0; s < STYLES; s++) {
                resp[s] = new int[] {s, STYLES + c, EOS};
                x[s] = features(resp[s]);
            }
            double[][] p = m.matrix(x);
            double ok = 0;
            for (int s = 0; s < STYLES; s++) {
                int t = (s + 1) % STYLES;
                boolean truth = truePreference(resp[s], resp[t], cyclic) > 0.5;
                ok += (p[s][t] > 0.5) == truth ? 1 : 0;
            }
            triples += ok / STYLES / CONTENTS;
        }
        return new ModelRow(held, triples);
    }

    /** The start: a policy that writes uniformly random well-formed responses. */
    static WindowLm start(long seed) {
        WindowLm lm = WindowLm.of(TOK.size(), 4, 32, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        Optimizer opt = Optimizer.adam(1e-2f);
        for (int s = 0; s < 60; s++) {
            List<int[]> ps = new ArrayList<>();
            List<int[]> cs = new ArrayList<>();
            for (int i = 0; i < 64; i++) {
                ps.add(PROMPT);
                cs.add(randomResponse(r));
            }
            lm.supervisedStep(ps, cs, opt);
        }
        return lm;
    }

    /**
     * @param lm a policy
     * @return its probability of starting with each style, renormalized over the styles
     */
    public static double[] styles(WindowLm lm) {
        float[] logits = lm.logits(PROMPT);
        double[] p = new double[STYLES];
        double max = Double.NEGATIVE_INFINITY;
        for (int s = 0; s < STYLES; s++) {
            max = Math.max(max, logits[s]);
        }
        double sum = 0;
        for (int s = 0; s < STYLES; s++) {
            p[s] = Math.exp(logits[s] - max);
            sum += p[s];
        }
        for (int s = 0; s < STYLES; s++) {
            p[s] /= sum;
        }
        return p;
    }

    /**
     * @param a one style distribution
     * @param b another
     * @param cyclic the annotators
     * @return the probability a sample of {@code a} beats one of {@code b} on style (ties half)
     */
    public static double winRate(double[] a, double[] b, boolean cyclic) {
        double w = 0;
        for (int i = 0; i < STYLES; i++) {
            for (int j = 0; j < STYLES; j++) {
                w += a[i] * b[j] * styleWin(i, j, cyclic);
            }
        }
        return w;
    }

    private static PolicyRow score(WindowLm lm, boolean cyclic) {
        double[] s = styles(lm);
        double[][] panel = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1.0 / 3, 1.0 / 3, 1.0 / 3}};
        double[] w = new double[panel.length];
        double worst = 1;
        for (int i = 0; i < panel.length; i++) {
            w[i] = winRate(s, panel[i], cyclic);
            worst = Math.min(worst, w[i]);
        }
        return new PolicyRow(worst, s, w);
    }

    private static SelfPlayPreference.Preference preference(PreferenceEmbeddingModel m) {
        return (prompt, candidates) -> {
            float[][] x = new float[candidates.size()][];
            for (int i = 0; i < x.length; i++) {
                x[i] = features(candidates.get(i));
            }
            return m.matrix(x);
        };
    }

    /**
     * Tunes the shared start with each method (9.2.3).
     *
     * @param cyclic cyclic or transitive annotators
     * @param pairs preference pairs (offline data and preference-model training)
     * @param steps training steps per method
     * @param seed the seed
     * @return method name to its scores; {@code "start"} first
     */
    public static Map<String, PolicyRow> policies(boolean cyclic, int pairs, int steps, long seed) {
        WindowLm start = start(seed);
        PreferenceEmbeddingModel gpm = fit(PreferenceEmbeddingModel.embedding(FEATURES, 32, 2, seed), pairs, 150,
                cyclic, seed + 1);
        PreferenceEmbeddingModel bt = fit(PreferenceEmbeddingModel.bradleyTerry(FEATURES, 32, seed), pairs, 150,
                cyclic, seed + 1);
        Map<String, PolicyRow> out = new LinkedHashMap<>();
        out.put("start", score(start, cyclic));
        SplittableRandom data = new SplittableRandom(seed + 3);
        List<DpoTrainer.Preference> offline = new ArrayList<>();
        for (int i = 0; i < pairs; i++) {
            int[][] p = labelledPair(data, cyclic);
            offline.add(new DpoTrainer.Preference(PROMPT, p[0], p[1]));
        }
        // offline DPO
        WindowLm dpoLm = start.copy();
        DpoTrainer dpo = DpoTrainer.of(dpoLm).beta(0.5f).optimizer(Optimizer.adam(3e-3f));
        SplittableRandom r = new SplittableRandom(seed + 4);
        for (int s = 0; s < steps; s++) {
            List<DpoTrainer.Preference> batch = new ArrayList<>();
            for (int i = 0; i < 32; i++) {
                batch.add(offline.get(r.nextInt(offline.size())));
            }
            dpo.step(batch);
        }
        out.put("DPO", score(dpoLm, cyclic));
        // iterative (online) DPO: the policy's own pairs, labelled by a random annotator
        WindowLm iter = start.copy();
        DpoTrainer online = DpoTrainer.of(iter).beta(0.5f).optimizer(Optimizer.adam(3e-3f));
        for (int s = 0; s < steps; s++) {
            if (s % 10 == 0) {
                online.reference(iter.copy());
            }
            int[][] rep = new int[64][];
            java.util.Arrays.fill(rep, PROMPT);
            int[][] o = iter.sample(rep, 3, EOS, 1.0, r);
            List<DpoTrainer.Preference> batch = new ArrayList<>();
            for (int i = 0; i + 1 < o.length; i += 2) {
                int type = cyclic ? r.nextInt(3) : 0;
                int sa = style(o[i]);
                int sb = style(o[i + 1]);
                if (sa < 0 || sb < 0 || sa == sb) {
                    continue;
                }
                boolean aWins = rank(type, sa) < rank(type, sb);
                batch.add(new DpoTrainer.Preference(PROMPT, aWins ? o[i] : o[i + 1], aWins ? o[i + 1] : o[i]));
            }
            if (!batch.isEmpty()) {
                online.step(batch);
            }
        }
        out.put("iterative DPO", score(iter, cyclic));
        // Bradley-Terry reward + PPO
        WindowLm ppoLm = start.copy();
        PpoLmTrainer ppo = PpoLmTrainer.of(ppoLm).optimizers(Optimizer.adam(3e-3f), Optimizer.adam(3e-3f)).batch(64)
                .klCoefficient(0.02f).completions(3, EOS);
        for (int s = 0; s < steps; s++) {
            ppo.step(x -> PROMPT, (p, c) -> {
                float[][] f = {features(c)};
                return Var.noGrad(() -> bt.embed(f)).value().toArray()[0];
            }, r);
        }
        out.put("BT reward + PPO", score(ppoLm, cyclic));
        // SPPO with the general preference model, with the BT model, and Nash-MD
        Map<String, SelfPlayPreference.Preference> sources = new LinkedHashMap<>();
        sources.put("SPPO (GPM)", preference(gpm));
        sources.put("SPPO (BT preference)", preference(bt));
        sources.put("SPPO (true preference)", (prompt, cs) -> {
            double[][] p = new double[cs.size()][cs.size()];
            for (int i = 0; i < p.length; i++) {
                for (int j = 0; j < p.length; j++) {
                    p[i][j] = truePreference(cs.get(i), cs.get(j), cyclic);
                }
            }
            return p;
        });
        for (Map.Entry<String, SelfPlayPreference.Preference> e : sources.entrySet()) {
            WindowLm lm = start.copy();
            SelfPlayPreference sppo = SelfPlayPreference.of(lm).samples(8).completions(3, EOS).eta(2)
                    .optimizer(Optimizer.adam(3e-3f));
            for (int s = 0; s < steps / 4; s++) {
                sppo.sppoIteration(x -> PROMPT, 8, e.getValue(), 4, r);
            }
            out.put(e.getKey(), score(lm, cyclic));
        }
        WindowLm nash = start.copy();
        SelfPlayPreference md = SelfPlayPreference.of(nash).samples(16).completions(3, EOS).nashMd(0.1f, 0.02f)
                .optimizer(Optimizer.adam(3e-3f));
        for (int s = 0; s < steps; s++) {
            md.nashMdStep(x -> PROMPT, 4, preference(gpm), r);
        }
        out.put("Nash-MD (GPM)", score(nash, cyclic));
        return out;
    }
}
