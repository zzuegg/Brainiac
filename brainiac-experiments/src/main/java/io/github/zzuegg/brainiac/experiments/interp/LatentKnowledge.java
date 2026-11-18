package io.github.zzuegg.brainiac.experiments.interp;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.interp.ContrastConsistentSearch;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.nn.LinearProbe;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 9 item 9.4.5: unsupervised latent knowledge with contrast-consistent search.
 *
 * <p>A {@link HookedDecoder} learns to judge arithmetic statements {@code a + b = c ?}
 * ({@code a, b} in 0-4) with a YES/NO answer token. Half the training statements are
 * prefixed by a user claim {@code USER u}, and there the model is taught to agree with the
 * user (answer YES exactly when {@code c = u}): it is sycophantic. On the
 * <em>sycophantic split</em> the user always claims a wrong sum, so the model's outputs are
 * wrong for every statement, although the truth may still be computed inside. Plain
 * statements completed with a random answer are also followed by a RIGHT/WRONG verdict
 * token (scored at that position only), standing in for the pretraining that makes a large
 * model track whether a completed statement is true.
 *
 * <p>For each statement both completions ({@code ... YES} and {@code ... NO}) are run and the
 * residual stream at the last position after block 1 is read. {@link ContrastConsistentSearch}
 * is fitted on the unlabelled pairs of a mixed training set and scored (sign-free) on held-out
 * plain statements and on the sycophantic split. Baselines: a supervised {@link LinearProbe}
 * on the normalised pair differences (the upper bound), the model's own zero-shot answer, the
 * contrastive-PCA direction and a random direction. The <em>distractor</em> split appends
 * one of two unseen tokens to the "yes" completion and the other to the "no" completion, at
 * random per statement: a feature that is perfectly consistent and confident but has
 * nothing to do with truth, which CCS may find instead.
 */
public final class LatentKnowledge {

    private static final int PLUS = 10;
    private static final int EQ = 11;
    private static final int YES = 12;
    private static final int NO = 13;
    private static final int USER = 14;
    private static final int D1 = 15;
    private static final int D2 = 16;
    private static final int BOS = 17;
    private static final int RIGHT = 18;
    private static final int WRONG = 19;
    private static final int VOCAB = 20;
    private static final int LAYER = 1;

    private LatentKnowledge() {
    }

    /** A statement and its truth. */
    private record Statement(int[] prefix, boolean truth, boolean sycophantAnswer) {
    }

    private static Statement statement(SplittableRandom r, int user) {
        int a = r.nextInt(5);
        int b = r.nextInt(5);
        int c = r.nextBoolean() ? a + b : (a + b + 1 + r.nextInt(8)) % 9;
        boolean truth = c == a + b;
        if (user < 0) {
            return new Statement(new int[] {BOS, a, PLUS, b, EQ, c}, truth, truth);
        }
        return new Statement(new int[] {BOS, USER, user, a, PLUS, b, EQ, c}, truth, c == user);
    }

    private static Statement sycophantic(SplittableRandom r) {
        int a = r.nextInt(5);
        int b = r.nextInt(5);
        int wrong = (a + b + 1 + r.nextInt(8)) % 9;
        int c = r.nextBoolean() ? wrong : a + b;
        return new Statement(new int[] {BOS, USER, wrong, a, PLUS, b, EQ, c}, c == a + b, c == wrong);
    }

    private static int[] append(int[] p, int... more) {
        int[] s = java.util.Arrays.copyOf(p, p.length + more.length);
        System.arraycopy(more, 0, s, p.length, more.length);
        return s;
    }

    /**
     * Trains the judge.
     *
     * @param steps Adam steps of 32 statements (half with a user claim)
     * @param seed the seed
     * @return the model
     */
    public static HookedDecoder train(int steps, long seed) {
        HookedDecoder m = HookedDecoder.create(HookedDecoder.Settings.tiny(VOCAB), seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            int[][] plain = new int[16][];
            int[][] user = new int[16][];
            for (int i = 0; i < 16; i++) {
                Statement p = statement(r, -1);
                plain[i] = append(p.prefix(), p.sycophantAnswer() ? YES : NO);
                Statement u = statement(r, r.nextInt(9));
                user[i] = append(u.prefix(), u.sycophantAnswer() ? YES : NO);
            }
            Var loss = m.loss(plain, true).add(m.loss(user, true)).add(verdictLoss(m, r));
            loss.backward();
            opt.step(m.parameters());
        }
        return m;
    }

    /**
     * The loss of a verdict after a completed plain statement: given {@code a + b = c ANSWER},
     * predict RIGHT when the answer is the truth and WRONG otherwise. Only the last position is
     * scored, so the answer head is not taught to answer at random. This gives the answer
     * position a reason to carry the truth of the completed statement, as next-token
     * pretraining does in large models.
     */
    private static Var verdictLoss(HookedDecoder m, SplittableRandom r) {
        int[][] in = new int[16][];
        float[] verdict = new float[16];
        for (int i = 0; i < 16; i++) {
            Statement p = statement(r, -1);
            boolean yes = r.nextBoolean();
            in[i] = append(p.prefix(), yes ? YES : NO);
            verdict[i] = yes == p.truth() ? RIGHT : WRONG;
        }
        int len = in[0].length;
        Var last = m.forward(in, null, null, true).logits().slice(1, len - 1, len).reshape(16, VOCAB);
        return Functional.crossEntropy(last, Tensor.of(Shape.of(16), verdict));
    }

    /** Hidden states of the last position after block {@link #LAYER}. */
    private static float[][] states(HookedDecoder m, List<int[]> seqs) {
        int n = seqs.size();
        int len = seqs.get(0).length;
        int[][] in = seqs.toArray(new int[0][]);
        float[] h = Var.noGrad(() -> m.forward(in).hidden().get(LAYER)).value().toArray();
        int d = m.settings().dim();
        float[][] out = new float[n][d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(h, ((i + 1) * len - 1) * d, out[i], 0, d);
        }
        return out;
    }

    /** Contrast pairs of one split. */
    private record Pairs(float[][] pos, float[][] neg, boolean[] truth, double zeroShot, boolean[] distractorOnYes) {
    }

    private static Pairs pairs(HookedDecoder m, List<Statement> st, boolean distractor, SplittableRandom r) {
        List<int[]> pos = new ArrayList<>();
        List<int[]> neg = new ArrayList<>();
        List<int[]> prompts = new ArrayList<>();
        boolean[] truth = new boolean[st.size()];
        boolean[] onYes = new boolean[st.size()];
        for (int i = 0; i < st.size(); i++) {
            Statement s = st.get(i);
            truth[i] = s.truth();
            onYes[i] = r.nextBoolean();
            if (distractor) {
                pos.add(append(s.prefix(), YES, onYes[i] ? D1 : D2));
                neg.add(append(s.prefix(), NO, onYes[i] ? D2 : D1));
            } else {
                pos.add(append(s.prefix(), YES));
                neg.add(append(s.prefix(), NO));
            }
            prompts.add(s.prefix());
        }
        float[] logits = Var.noGrad(() -> m.forward(prompts.toArray(new int[0][])).logits()).value().toArray();
        int len = prompts.get(0).length;
        int hit = 0;
        for (int i = 0; i < st.size(); i++) {
            int off = ((i + 1) * len - 1) * VOCAB;
            boolean saysYes = logits[off + YES] > logits[off + NO];
            hit += saysYes == truth[i] ? 1 : 0;
        }
        return new Pairs(states(m, pos), states(m, neg), truth, (double) hit / st.size(), onYes);
    }

    /**
     * The scores.
     *
     * @param ccsPlain CCS sign-free accuracy on held-out plain statements
     * @param ccsSycophantic on the sycophantic split
     * @param supervisedPlain the supervised probe on plain statements
     * @param supervisedSycophantic the supervised probe on the sycophantic split
     * @param zeroShotPlain the model's own answers on plain statements
     * @param zeroShotSycophantic the model's own answers on the sycophantic split
     * @param contrastivePca the contrastive-PCA baseline on plain statements
     * @param random a random direction on plain statements
     * @param distractorTruth per seed CCS's truth accuracy with the distractor present
     * @param distractorFeature per seed how well CCS's credence tracks the distractor instead
     */
    public record Scores(double ccsPlain, double ccsSycophantic, double supervisedPlain, double supervisedSycophantic,
            double zeroShotPlain, double zeroShotSycophantic, double contrastivePca, double random,
            double[] distractorTruth, double[] distractorFeature) {
    }

    private static float[][] differences(Pairs p) {
        float[][] d = new float[p.pos().length][];
        for (int i = 0; i < d.length; i++) {
            d[i] = new float[p.pos()[i].length];
            for (int j = 0; j < d[i].length; j++) {
                d[i][j] = p.pos()[i][j] - p.neg()[i][j];
            }
        }
        return d;
    }

    private static int[] ints(boolean[] b) {
        int[] v = new int[b.length];
        for (int i = 0; i < b.length; i++) {
            v[i] = b[i] ? 1 : 0;
        }
        return v;
    }

    private static Pairs concat(Pairs a, Pairs b) {
        int n = a.truth().length + b.truth().length;
        float[][] pos = new float[n][];
        float[][] neg = new float[n][];
        boolean[] truth = new boolean[n];
        boolean[] onYes = new boolean[n];
        for (int i = 0; i < n; i++) {
            Pairs src = i < a.truth().length ? a : b;
            int k = i < a.truth().length ? i : i - a.truth().length;
            pos[i] = src.pos()[k];
            neg[i] = src.neg()[k];
            truth[i] = src.truth()[k];
            onYes[i] = src.distractorOnYes()[k];
        }
        return new Pairs(pos, neg, truth, 0, onYes);
    }

    /**
     * Runs the study.
     *
     * @param trainSteps judge training steps
     * @param distractorSeeds CCS seeds on the distractor split
     * @param seed the seed
     * @return the scores
     */
    public static Scores run(int trainSteps, int distractorSeeds, long seed) {
        HookedDecoder m = train(trainSteps, seed);
        SplittableRandom r = new SplittableRandom(seed + 2);
        List<Statement> fitPlain = new ArrayList<>();
        List<Statement> fitSyc = new ArrayList<>();
        List<Statement> testPlain = new ArrayList<>();
        List<Statement> testSyc = new ArrayList<>();
        for (int i = 0; i < 200; i++) {
            fitPlain.add(statement(r, -1));
            fitSyc.add(sycophantic(r));
            testPlain.add(statement(r, -1));
            testSyc.add(sycophantic(r));
        }
        Pairs fit = concat(pairs(m, fitPlain, false, r), pairs(m, fitSyc, false, r));
        Pairs plain = pairs(m, testPlain, false, r);
        Pairs syc = pairs(m, testSyc, false, r);
        ContrastConsistentSearch ccs = ContrastConsistentSearch.fit(fit.pos(), fit.neg(), 5, 400, seed + 3);
        LinearProbe sup = LinearProbe.fit(differences(fit), ints(fit.truth()), 2, 300, 0.05, 1e-4);
        double supPlain = sup.accuracy(differences(plain), ints(plain.truth()));
        double supSyc = sup.accuracy(differences(syc), ints(syc.truth()));
        double pca = ContrastConsistentSearch.contrastivePca(plain.pos(), plain.neg(), plain.truth());
        double rnd = ContrastConsistentSearch.randomDirection(plain.pos(), plain.neg(), plain.truth(), seed + 4);
        double[] dTruth = new double[distractorSeeds];
        double[] dFeature = new double[distractorSeeds];
        for (int s = 0; s < distractorSeeds; s++) {
            SplittableRandom rs = new SplittableRandom(seed + 50 + s);
            Pairs dFit = pairs(m, fitPlain, true, rs);
            Pairs dTest = pairs(m, testPlain, true, rs);
            ContrastConsistentSearch d = ContrastConsistentSearch.fit(dFit.pos(), dFit.neg(), 3, 300, seed + 60 + s);
            dTruth[s] = d.accuracy(dTest.pos(), dTest.neg(), dTest.truth());
            dFeature[s] = d.accuracy(dTest.pos(), dTest.neg(), dTest.distractorOnYes());
        }
        return new Scores(ccs.accuracy(plain.pos(), plain.neg(), plain.truth()),
                ccs.accuracy(syc.pos(), syc.neg(), syc.truth()), supPlain, supSyc, plain.zeroShot(), syc.zeroShot(),
                pca, rnd, dTruth, dFeature);
    }
}
