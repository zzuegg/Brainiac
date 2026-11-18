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
        int vocab = m.settings().vocab();
        int len = prompts.get(0).length;
        int hit = 0;
        for (int i = 0; i < st.size(); i++) {
            int off = ((i + 1) * len - 1) * vocab;
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

    // ------------------------------------------------------------------ wave 10 follow-up (10.3.17)

    private static final int MARK = 20;
    private static final int TEMPLATES = 8;

    /**
     * Renders a statement prefix in prompt template {@code k} (0-7): the operands in the given
     * or swapped order ({@code k % 2}) and a template marker token at one of four positions
     * ({@code k / 2}); truth and length are unchanged.
     */
    private static int[] render(int[] prefix, int k) {
        int head = prefix.length - 5;
        int a = prefix[head];
        int b = prefix[head + 2];
        int c = prefix[head + 4];
        int[] body = k % 2 == 0 ? new int[] {a, PLUS, b, EQ, c} : new int[] {b, PLUS, a, EQ, c};
        int pos = switch (k / 2) {
            case 0 -> 0;
            case 1 -> 1;
            case 2 -> 3;
            default -> 2;
        };
        int[] out = new int[prefix.length + 1];
        System.arraycopy(prefix, 0, out, 0, head);
        int o = head;
        for (int i = 0; i < 5; i++) {
            if (i == pos) {
                out[o++] = MARK + k;
            }
            out[o++] = body[i];
        }
        return out;
    }

    private static Statement templated(Statement s, int k) {
        return new Statement(render(s.prefix(), k), s.truth(), s.sycophantAnswer());
    }

    /**
     * Trains the judge on statements rendered in random prompt templates (the follow-up's
     * model; wave 9's {@link #train} is unchanged).
     *
     * @param steps Adam steps of 32 statements plus 16 verdicts
     * @param templates how many templates the training statements use, 1 to 8
     * @param seed the seed
     * @return the model
     */
    public static HookedDecoder trainTemplated(int steps, int templates, long seed) {
        HookedDecoder m = HookedDecoder.create(HookedDecoder.Settings.tiny(MARK + TEMPLATES), seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            int[][] plain = new int[16][];
            int[][] user = new int[16][];
            int[][] verdictIn = new int[16][];
            float[] verdict = new float[16];
            for (int i = 0; i < 16; i++) {
                Statement p = templated(statement(r, -1), r.nextInt(templates));
                plain[i] = append(p.prefix(), p.sycophantAnswer() ? YES : NO);
                Statement u = templated(statement(r, r.nextInt(9)), r.nextInt(templates));
                user[i] = append(u.prefix(), u.sycophantAnswer() ? YES : NO);
                Statement v = templated(statement(r, -1), r.nextInt(templates));
                boolean yes = r.nextBoolean();
                verdictIn[i] = append(v.prefix(), yes ? YES : NO);
                verdict[i] = yes == v.truth() ? RIGHT : WRONG;
            }
            int len = verdictIn[0].length;
            Var last = m.forward(verdictIn, null, null, true).logits().slice(1, len - 1, len)
                    .reshape(16, MARK + TEMPLATES);
            Var loss = m.loss(plain, true).add(m.loss(user, true))
                    .add(Functional.crossEntropy(last, Tensor.of(Shape.of(16), verdict)));
            loss.backward();
            opt.step(m.parameters());
        }
        return m;
    }

    /**
     * One row of the follow-up, for one number of prompt templates. Accuracies are sign-free;
     * per-template numbers are averaged over the templates.
     *
     * @param templates the number of templates pooled
     * @param ccsPooled wave 9's CCS (one normalisation for the pooled set), plain test statements
     * @param ccsPerTemplate CCS with per-template normalisation
     * @param ccsEnsemble the same with credences averaged over the templates
     * @param crcEnsemble CRC with per-template normalisation, averaged over templates
     * @param supervised the supervised probe on pair differences (upper bound)
     * @param zeroShot the model's own answers
     * @param random a random direction
     * @param ccsSycophantic per-template CCS ensemble on the sycophantic split
     * @param distractorTruth per-template CCS fitted with the distractor: truth accuracy
     * @param distractorFeature the same CCS: agreement with the distractor feature
     * @param crcDistractorTruth CRC with the distractor: truth accuracy
     * @param crcDistractorFeature CRC with the distractor: agreement with the distractor
     */
    public record FollowUpRow(int templates, double ccsPooled, double ccsPerTemplate, double ccsEnsemble,
            double crcEnsemble, double supervised, double zeroShot, double random, double ccsSycophantic,
            double distractorTruth, double distractorFeature, double crcDistractorTruth,
            double crcDistractorFeature) {

        @Override
        public String toString() {
            return String.format("T=%d CCS pooled %.3f per-template %.3f ensemble %.3f | CRC %.3f | supervised %.3f"
                    + " zero-shot %.3f random %.3f | sycophantic CCS %.3f | distractor CCS truth %.3f feature %.3f,"
                    + " CRC truth %.3f feature %.3f", templates, ccsPooled, ccsPerTemplate, ccsEnsemble, crcEnsemble,
                    supervised, zeroShot, random, ccsSycophantic, distractorTruth, distractorFeature,
                    crcDistractorTruth, crcDistractorFeature);
        }
    }

    /** Pairs of the same statements under templates 0..t-1, stacked, with their template ids. */
    private record Templated(Pairs[] perTemplate, Pairs pooled, int[] template) {
    }

    private static Templated templatedPairs(HookedDecoder m, List<Statement> st, int t, boolean distractor,
            SplittableRandom r) {
        Pairs[] per = new Pairs[t];
        Pairs pooled = null;
        int[] template = new int[st.size() * t];
        for (int k = 0; k < t; k++) {
            List<Statement> rendered = new ArrayList<>();
            for (Statement s : st) {
                rendered.add(templated(s, k));
            }
            per[k] = pairs(m, rendered, distractor, r);
            pooled = pooled == null ? per[k] : concat(pooled, per[k]);
            java.util.Arrays.fill(template, k * st.size(), (k + 1) * st.size(), k);
        }
        return new Templated(per, pooled, template);
    }

    private static int[] constant(int n, int k) {
        int[] t = new int[n];
        java.util.Arrays.fill(t, k);
        return t;
    }

    /**
     * Wave 10 item 10.3.17: CCS with per-template normalisation, prompt ensembles and CRC,
     * against wave 9's pooled CCS, the supervised probe, zero-shot answers and a random
     * direction, with sycophantic and distractor splits.
     *
     * @param trainSteps judge training steps (wave 9 used 450; the follow-up's Run uses 4x)
     * @param templateCounts the template counts to compare, each at most 8
     * @param seed the seed
     * @return one row per template count
     */
    public static List<FollowUpRow> followUp(int trainSteps, int[] templateCounts, long seed) {
        return followUp(trainSteps, templateCounts, 150, seed);
    }

    /**
     * The follow-up with a chosen number of statements per split.
     *
     * @param trainSteps judge training steps
     * @param templateCounts the template counts to compare, each at most 8
     * @param statements statements per split (fit and test, plain and sycophantic)
     * @param seed the seed
     * @return one row per template count
     */
    public static List<FollowUpRow> followUp(int trainSteps, int[] templateCounts, int statements, long seed) {
        int maxT = 1;
        for (int t : templateCounts) {
            maxT = Math.max(maxT, t);
        }
        HookedDecoder m = trainTemplated(trainSteps, maxT, seed);
        List<FollowUpRow> rows = new ArrayList<>();
        for (int t : templateCounts) {
            SplittableRandom r = new SplittableRandom(seed + 2);
            List<Statement> fitPlain = new ArrayList<>();
            List<Statement> fitSyc = new ArrayList<>();
            List<Statement> testPlain = new ArrayList<>();
            List<Statement> testSyc = new ArrayList<>();
            for (int i = 0; i < statements; i++) {
                fitPlain.add(statement(r, -1));
                fitSyc.add(sycophantic(r));
                testPlain.add(statement(r, -1));
                testSyc.add(sycophantic(r));
            }
            Templated fp = templatedPairs(m, fitPlain, t, false, r);
            Templated fs = templatedPairs(m, fitSyc, t, false, r);
            Pairs fit = concat(fp.pooled(), fs.pooled());
            int[] fitTemplate = new int[fp.template().length + fs.template().length];
            System.arraycopy(fp.template(), 0, fitTemplate, 0, fp.template().length);
            System.arraycopy(fs.template(), 0, fitTemplate, fp.template().length, fs.template().length);
            Templated plain = templatedPairs(m, testPlain, t, false, r);
            Templated syc = templatedPairs(m, testSyc, t, false, r);
            ContrastConsistentSearch pooledCcs = ContrastConsistentSearch.fit(fit.pos(), fit.neg(), 3, 250, seed + 3);
            ContrastConsistentSearch ccs = ContrastConsistentSearch.fit(fit.pos(), fit.neg(), fitTemplate, 3, 250,
                    seed + 3);
            ContrastConsistentSearch.Crc crc = ContrastConsistentSearch.crc(fit.pos(), fit.neg(), fitTemplate);
            LinearProbe sup = LinearProbe.fit(differences(fit), ints(fit.truth()), 2, 300, 0.05, 1e-4);
            double pooledAcc = 0;
            double perAcc = 0;
            double supAcc = 0;
            double zero = 0;
            double[][] ccsCred = new double[t][];
            double[][] crcCred = new double[t][];
            double[][] sycCred = new double[t][];
            for (int k = 0; k < t; k++) {
                Pairs p = plain.perTemplate()[k];
                int[] tk = constant(p.truth().length, k);
                pooledAcc += pooledCcs.accuracy(p.pos(), p.neg(), p.truth()) / t;
                ccsCred[k] = ccs.credences(p.pos(), p.neg(), tk);
                perAcc += ContrastConsistentSearch.signFreeAccuracy(ccsCred[k], p.truth()) / t;
                crcCred[k] = crc.credences(p.pos(), p.neg(), tk);
                supAcc += sup.accuracy(differences(p), ints(p.truth())) / t;
                zero += p.zeroShot() / t;
                Pairs q = syc.perTemplate()[k];
                sycCred[k] = ccs.credences(q.pos(), q.neg(), constant(q.truth().length, k));
            }
            Pairs p0 = plain.perTemplate()[0];
            double rnd = ContrastConsistentSearch.randomDirection(p0.pos(), p0.neg(), p0.truth(), seed + 4);
            // distractor split: fit per-template CCS and CRC on distractor pairs of the plain statements
            SplittableRandom rd = new SplittableRandom(seed + 50);
            Templated dFit = templatedPairs(m, fitPlain, t, true, rd);
            Templated dTest = templatedPairs(m, testPlain, t, true, rd);
            ContrastConsistentSearch dCcs = ContrastConsistentSearch.fit(dFit.pooled().pos(), dFit.pooled().neg(),
                    dFit.template(), 3, 300, seed + 60);
            ContrastConsistentSearch.Crc dCrc = ContrastConsistentSearch.crc(dFit.pooled().pos(), dFit.pooled().neg(),
                    dFit.template());
            double dTruth = 0;
            double dFeature = 0;
            double cTruth = 0;
            double cFeature = 0;
            for (int k = 0; k < t; k++) {
                Pairs p = dTest.perTemplate()[k];
                int[] tk = constant(p.truth().length, k);
                double[] c1 = dCcs.credences(p.pos(), p.neg(), tk);
                double[] c2 = dCrc.credences(p.pos(), p.neg(), tk);
                dTruth += ContrastConsistentSearch.signFreeAccuracy(c1, p.truth()) / t;
                dFeature += ContrastConsistentSearch.signFreeAccuracy(c1, p.distractorOnYes()) / t;
                cTruth += ContrastConsistentSearch.signFreeAccuracy(c2, p.truth()) / t;
                cFeature += ContrastConsistentSearch.signFreeAccuracy(c2, p.distractorOnYes()) / t;
            }
            rows.add(new FollowUpRow(t, pooledAcc, perAcc, ContrastConsistentSearch.ensembleAccuracy(ccsCred,
                    p0.truth()), ContrastConsistentSearch.ensembleAccuracy(crcCred, p0.truth()), supAcc, zero, rnd,
                    ContrastConsistentSearch.ensembleAccuracy(sycCred, syc.perTemplate()[0].truth()), dTruth,
                    dFeature, cTruth, cFeature));
        }
        return rows;
    }

    /**
     * The follow-up's Run: the judge trained 4x longer than wave 9 (1800 steps) on 8 templates,
     * compared with 1, 4 and 8 templates pooled, for three seeds.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        for (long seed = 1; seed <= 3; seed++) {
            for (FollowUpRow row : followUp(1800, new int[] {1, 4, 8}, seed)) {
                System.out.println("seed " + seed + " " + row);
            }
        }
    }
}
