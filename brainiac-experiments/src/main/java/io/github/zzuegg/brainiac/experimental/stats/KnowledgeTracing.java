package io.github.zzuegg.brainiac.experimental.stats;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Knowledge tracing: predict whether a student answers the next question correctly from their
 * answer history, and estimate their mastery of each skill.
 *
 * <ul>
 *   <li>{@link Model#BKT} (Corbett and Anderson, 1994): a two-state hidden Markov model per skill
 *       (learn, forget, slip, guess, initial mastery) fitted by Baum-Welch EM; mastery is the
 *       filtered posterior.</li>
 *   <li>{@link Model#DKT} (Piech et al., NeurIPS 2015,
 *       <a href="https://arxiv.org/abs/1506.05908">arXiv:1506.05908</a>): an LSTM over one-hot
 *       (skill, correct) tokens with a sigmoid output per skill.</li>
 *   <li>{@link Model#SIMPLE_KT} (Liu et al., ICLR 2023,
 *       <a href="https://arxiv.org/abs/2302.06881">arXiv:2302.06881</a>): Rasch-style question
 *       embeddings {@code c_skill + mu_q d_skill} and causal dot-product attention from the next
 *       question to past interactions.</li>
 *   <li>{@link Model#DKT2_LITE} (after DKT2, Zhou, Han and Chen, 2025,
 *       <a href="https://arxiv.org/abs/2501.14256">arXiv:2501.14256</a>): Rasch-embedded
 *       interactions into an LSTM (in place of the paper's xLSTM blocks), read out IRT-style as a
 *       per-skill ability minus the next question's difficulty, so the ability doubles as a
 *       mastery estimate. The paper's familiar/unfamiliar split is not reproduced.</li>
 * </ul>
 */
public final class KnowledgeTracing {

    /** The model. */
    public enum Model {
        /** Bayesian knowledge tracing. */
        BKT,
        /** Deep knowledge tracing (LSTM). */
        DKT,
        /** simpleKT (attention with Rasch embeddings). */
        SIMPLE_KT,
        /** DKT2 with an LSTM and an IRT readout. */
        DKT2_LITE
    }

    /**
     * One student's sequence.
     *
     * @param question the questions
     * @param skill their skills
     * @param correct 1 for a correct answer
     */
    public record Sequence(int[] question, int[] skill, int[] correct) {
    }

    /**
     * Per-step predictions for one sequence.
     *
     * @param correct the predicted probability of a correct answer at each step, from the
     *     history before it
     * @param mastery the estimated mastery of the step's skill before the answer
     */
    public record Prediction(double[] correct, double[] mastery) {
    }

    private final Model model;
    private final int skills;
    private final int questions;
    // BKT parameters per skill: {L0, learn, forget, slip, guess}
    private double[][] bkt;
    // neural parameters
    private Layer lstm;
    private Var wOut;
    private Var bOut;
    private Var skillBase;
    private Var skillVar;
    private Var questionMu;
    private Var answerEmb;
    private Var difficulty;
    private Var w1;
    private Var b1;
    private Var w2;
    private Var b2;
    private int dim;

    private KnowledgeTracing(Model model, int skills, int questions) {
        this.model = model;
        this.skills = skills;
        this.questions = questions;
    }

    /**
     * Fits a model.
     *
     * @param model the model
     * @param train training sequences (equal lengths for the neural models)
     * @param skills the number of skills
     * @param questions the number of questions
     * @param epochs EM iterations (BKT) or full-batch Adam steps (neural)
     * @param seed the seed
     * @return the fitted model
     */
    public static KnowledgeTracing fit(Model model, List<Sequence> train, int skills, int questions, int epochs,
            long seed) {
        KnowledgeTracing kt = new KnowledgeTracing(model, skills, questions);
        if (model == Model.BKT) {
            kt.fitBkt(train, epochs);
        } else {
            kt.fitNeural(train, epochs, seed);
        }
        return kt;
    }

    // ------------------------------------------------------------------ BKT

    private void fitBkt(List<Sequence> train, int iterations) {
        bkt = new double[skills][];
        for (int s = 0; s < skills; s++) {
            List<int[]> obs = new ArrayList<>();
            for (Sequence q : train) {
                int c = 0;
                for (int v : q.skill()) {
                    if (v == s) {
                        c++;
                    }
                }
                int[] o = new int[c];
                int k = 0;
                for (int t = 0; t < q.skill().length; t++) {
                    if (q.skill()[t] == s) {
                        o[k++] = q.correct()[t];
                    }
                }
                if (c > 0) {
                    obs.add(o);
                }
            }
            bkt[s] = baumWelch(obs, iterations);
        }
    }

    /** EM for a 2-state HMM: state 1 = mastered. Returns {L0, learn, forget, slip, guess}. */
    private static double[] baumWelch(List<int[]> seqs, int iterations) {
        double l0 = 0.3;
        double learn = 0.1;
        double forget = 0.02;
        double slip = 0.1;
        double guess = 0.25;
        for (int it = 0; it < iterations; it++) {
            double sL0 = 0;
            double nSeq = 0;
            double trans01 = 0;
            double from0 = 0;
            double trans10 = 0;
            double from1 = 0;
            double slipNum = 0;
            double in1 = 0;
            double guessNum = 0;
            double in0 = 0;
            for (int[] o : seqs) {
                int t1 = o.length;
                double[][] a = new double[t1][2];
                double[][] b = new double[t1][2];
                double[] scale = new double[t1];
                for (int t = 0; t < t1; t++) {
                    double e0 = o[t] == 1 ? guess : 1 - guess;
                    double e1 = o[t] == 1 ? 1 - slip : slip;
                    double p0;
                    double p1;
                    if (t == 0) {
                        p0 = 1 - l0;
                        p1 = l0;
                    } else {
                        p0 = a[t - 1][0] * (1 - learn) + a[t - 1][1] * forget;
                        p1 = a[t - 1][0] * learn + a[t - 1][1] * (1 - forget);
                    }
                    a[t][0] = p0 * e0;
                    a[t][1] = p1 * e1;
                    scale[t] = a[t][0] + a[t][1];
                    a[t][0] /= scale[t];
                    a[t][1] /= scale[t];
                }
                b[t1 - 1][0] = 1;
                b[t1 - 1][1] = 1;
                for (int t = t1 - 2; t >= 0; t--) {
                    double e0 = o[t + 1] == 1 ? guess : 1 - guess;
                    double e1 = o[t + 1] == 1 ? 1 - slip : slip;
                    b[t][0] = ((1 - learn) * e0 * b[t + 1][0] + learn * e1 * b[t + 1][1]) / scale[t + 1];
                    b[t][1] = (forget * e0 * b[t + 1][0] + (1 - forget) * e1 * b[t + 1][1]) / scale[t + 1];
                }
                for (int t = 0; t < t1; t++) {
                    double g1 = a[t][1] * b[t][1];
                    double g0 = a[t][0] * b[t][0];
                    double z = g0 + g1;
                    g0 /= z;
                    g1 /= z;
                    if (t == 0) {
                        sL0 += g1;
                        nSeq++;
                    }
                    if (o[t] == 1) {
                        guessNum += g0;
                    } else {
                        slipNum += g1;
                    }
                    in0 += g0;
                    in1 += g1;
                    if (t + 1 < t1) {
                        double e0 = o[t + 1] == 1 ? guess : 1 - guess;
                        double e1 = o[t + 1] == 1 ? 1 - slip : slip;
                        double x00 = a[t][0] * (1 - learn) * e0 * b[t + 1][0];
                        double x01 = a[t][0] * learn * e1 * b[t + 1][1];
                        double x10 = a[t][1] * forget * e0 * b[t + 1][0];
                        double x11 = a[t][1] * (1 - forget) * e1 * b[t + 1][1];
                        double zz = x00 + x01 + x10 + x11;
                        trans01 += x01 / zz;
                        from0 += (x00 + x01) / zz;
                        trans10 += x10 / zz;
                        from1 += (x10 + x11) / zz;
                    }
                }
            }
            l0 = clamp(sL0 / Math.max(1, nSeq), 0.01, 0.99);
            learn = clamp(trans01 / Math.max(1e-9, from0), 0.001, 0.5);
            forget = clamp(trans10 / Math.max(1e-9, from1), 0.0, 0.3);
            slip = clamp(slipNum / Math.max(1e-9, in1), 0.01, 0.45);
            guess = clamp(guessNum / Math.max(1e-9, in0), 0.01, 0.45);
        }
        return new double[] {l0, learn, forget, slip, guess};
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private Prediction predictBkt(Sequence s) {
        int t1 = s.correct().length;
        double[] pl = new double[skills];
        for (int k = 0; k < skills; k++) {
            pl[k] = bkt[k][0];
        }
        double[] pc = new double[t1];
        double[] mastery = new double[t1];
        for (int t = 0; t < t1; t++) {
            int k = s.skill()[t];
            double[] p = bkt[k];
            mastery[t] = pl[k];
            pc[t] = pl[k] * (1 - p[3]) + (1 - pl[k]) * p[4];
            double post = s.correct()[t] == 1 ? pl[k] * (1 - p[3]) / pc[t] : pl[k] * p[3] / (1 - pc[t]);
            pl[k] = post * (1 - p[2]) + (1 - post) * p[1];
        }
        return new Prediction(pc, mastery);
    }

    // ------------------------------------------------------------------ neural models

    private record Batch(Tensor skillIn, Tensor answerIn, Tensor questionIn, Tensor nextSkill, Tensor nextQuestion,
            Tensor tokens, Tensor target, Tensor querySkill, Tensor queryQuestion) {
    }

    /** One-hot encodings of a batch of equal-length sequences. */
    private Batch encode(List<Sequence> seqs) {
        int b = seqs.size();
        int t1 = seqs.get(0).correct().length;
        int t = t1 - 1;
        float[] sk = new float[b * t * skills];
        float[] an = new float[b * t * 2];
        float[] qu = new float[b * t * questions];
        float[] ns = new float[b * t * skills];
        float[] nq = new float[b * t * questions];
        float[] tok = new float[b * t * 2 * skills];
        float[] tg = new float[b * t];
        // attention queries cover every step 0..t1-1 (the first one only sees the start token)
        float[] qs = new float[b * t1 * skills];
        float[] qq = new float[b * t1 * questions];
        for (int i = 0; i < b; i++) {
            Sequence s = seqs.get(i);
            for (int k = 0; k < t; k++) {
                int o = i * t + k;
                sk[o * skills + s.skill()[k]] = 1;
                an[o * 2 + s.correct()[k]] = 1;
                qu[o * questions + s.question()[k]] = 1;
                ns[o * skills + s.skill()[k + 1]] = 1;
                nq[o * questions + s.question()[k + 1]] = 1;
                tok[o * 2 * skills + s.skill()[k] * 2 + s.correct()[k]] = 1;
                tg[o] = s.correct()[k + 1];
            }
            for (int k = 0; k < t1; k++) {
                int o = i * t1 + k;
                qs[o * skills + s.skill()[k]] = 1;
                qq[o * questions + s.question()[k]] = 1;
            }
        }
        return new Batch(Tensor.of(Shape.of(b, t, skills), sk), Tensor.of(Shape.of(b, t, 2), an),
                Tensor.of(Shape.of(b, t, questions), qu), Tensor.of(Shape.of(b, t, skills), ns),
                Tensor.of(Shape.of(b, t, questions), nq), Tensor.of(Shape.of(b, t, 2 * skills), tok),
                Tensor.of(Shape.of(b, t, 1), tg), Tensor.of(Shape.of(b, t1, skills), qs),
                Tensor.of(Shape.of(b, t1, questions), qq));
    }

    private List<Var> neuralParameters() {
        List<Var> p = new ArrayList<>();
        if (lstm != null) {
            p.addAll(lstm.parameters());
        }
        for (Var v : new Var[] {wOut, bOut, skillBase, skillVar, questionMu, answerEmb, difficulty, w1, b1, w2, b2}) {
            if (v != null) {
                p.add(v);
            }
        }
        return p;
    }

    private void fitNeural(List<Sequence> train, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int t = train.get(0).correct().length - 1;
        int h = 32;
        dim = 16;
        switch (model) {
            case DKT -> {
                lstm = Recurrent.lstm(h).returnSequences(true).create(Shape.of(t, 2 * skills), r);
                wOut = glorot(h, skills, r);
                bOut = Var.param(Tensor.zeros(Shape.of(1, skills)));
            }
            case SIMPLE_KT -> {
                rasch(r);
                w1 = glorot(2 * dim, h, r);
                b1 = Var.param(Tensor.zeros(Shape.of(1, h)));
                w2 = glorot(h, 1, r);
                b2 = Var.param(Tensor.zeros(Shape.of(1, 1)));
            }
            case DKT2_LITE -> {
                rasch(r);
                lstm = Recurrent.lstm(h).returnSequences(true).create(Shape.of(t, dim), r);
                wOut = glorot(h, skills, r);
                bOut = Var.param(Tensor.zeros(Shape.of(1, skills)));
                difficulty = Var.param(Tensor.zeros(Shape.of(questions, 1)));
            }
            default -> throw new IllegalStateException();
        }
        Batch batch = encode(train);
        Optimizer opt = Optimizer.adam(0.01f);
        List<Var> params = neuralParameters();
        for (int s = 0; s < steps; s++) {
            Var logits = logits(batch);
            Var loss = logits.softplus().sub(logits.mul(batch.target())).mean();
            loss.backward();
            opt.step(params);
        }
    }

    private void rasch(SplittableRandom r) {
        skillBase = glorot(skills, dim, r);
        skillVar = glorot(skills, dim, r);
        questionMu = Var.param(Tensor.zeros(Shape.of(questions, 1)));
        answerEmb = glorot(2, dim, r);
    }

    private static Var glorot(int in, int out, SplittableRandom r) {
        float bound = (float) Math.sqrt(6.0 / (in + out));
        return Var.param(Tensor.uniform(Shape.of(in, out), -bound, bound, r));
    }

    /** Rasch question embeddings {@code c_s + mu_q d_s}. */
    private Var questionEmbedding(Tensor skillOneHot, Tensor questionOneHot) {
        Var s = Var.of(skillOneHot);
        Var q = Var.of(questionOneHot);
        return s.matmul(skillBase).add(q.matmul(questionMu).mul(s.matmul(skillVar)));
    }

    /** Next-step logits {@code [B, T, 1]} for steps 1..T. */
    private Var logits(Batch b) {
        return switch (model) {
            case DKT -> lstm.forward(Var.of(b.tokens())).matmul(wOut).add(bOut).mul(b.nextSkill()).sum(2, true);
            case DKT2_LITE -> {
                Var x = questionEmbedding(b.skillIn(), b.questionIn()).add(Var.of(b.answerIn()).matmul(answerEmb));
                Var ability = lstm.forward(x).matmul(wOut).add(bOut).mul(b.nextSkill()).sum(2, true);
                yield ability.sub(Var.of(b.nextQuestion()).matmul(difficulty));
            }
            case SIMPLE_KT -> {
                Var all = attentionLogits(b);
                int t1 = all.shape().dim(1);
                yield all.slice(1, 1, t1);
            }
            default -> throw new IllegalStateException();
        };
    }

    /** simpleKT logits for every step 0..T (step 0 attends to the start token only). */
    private Var attentionLogits(Batch b) {
        Var query = questionEmbedding(b.querySkill(), b.queryQuestion()); // [B, T+1, d]
        Var interaction = questionEmbedding(b.skillIn(), b.questionIn())
                .add(Var.of(b.answerIn()).matmul(answerEmb)); // [B, T, d]
        Var pastKeys = questionEmbedding(b.skillIn(), b.questionIn());
        int bs = query.shape().dim(0);
        int t1 = query.shape().dim(1);
        Var start = Var.of(Tensor.zeros(Shape.of(bs, 1, dim)));
        Var keys = Var.concat(1, start, pastKeys);
        Var values = Var.concat(1, start, interaction);
        float[] mask = new float[t1 * t1];
        for (int i = 0; i < t1; i++) {
            for (int j = 0; j < t1; j++) {
                mask[i * t1 + j] = j > i ? -1e9f : 0;
            }
        }
        Var scores = query.matmul(keys.permute(0, 2, 1)).div((float) Math.sqrt(dim)).add(Tensor.of(Shape.of(t1, t1),
                mask));
        Var attended = scores.softmax(2).matmul(values);
        return Var.concat(2, query, attended).matmul(w1).add(b1).relu().matmul(w2).add(b2);
    }

    // ------------------------------------------------------------------ prediction

    /**
     * Predicts a batch of sequences.
     *
     * @param seqs sequences (equal lengths for the neural models)
     * @return one prediction per sequence; for neural models step 0 is predicted at the base
     *     rate 0.5 (DKT, DKT2-lite) or from the start token (simpleKT)
     */
    public List<Prediction> predict(List<Sequence> seqs) {
        List<Prediction> out = new ArrayList<>();
        if (model == Model.BKT) {
            for (Sequence s : seqs) {
                out.add(predictBkt(s));
            }
            return out;
        }
        Batch b = encode(seqs);
        int t1 = seqs.get(0).correct().length;
        float[] lg;
        float[] mastery;
        if (model == Model.SIMPLE_KT) {
            lg = Var.noGrad(() -> attentionLogits(b)).value().toArray();
            mastery = lg;
        } else {
            lg = Var.noGrad(() -> logits(b)).value().toArray();
            // mastery of the next step's skill: the readout before the difficulty term
            mastery = model == Model.DKT2_LITE ? Var.noGrad(() -> {
                Var x = questionEmbedding(b.skillIn(), b.questionIn()).add(Var.of(b.answerIn()).matmul(answerEmb));
                return lstm.forward(x).matmul(wOut).add(bOut).mul(b.nextSkill()).sum(2, true);
            }).value().toArray() : lg;
        }
        for (int i = 0; i < seqs.size(); i++) {
            double[] pc = new double[t1];
            double[] ms = new double[t1];
            for (int k = 0; k < t1; k++) {
                if (model == Model.SIMPLE_KT) {
                    pc[k] = sigmoid(lg[i * t1 + k]);
                    ms[k] = pc[k];
                } else if (k == 0) {
                    pc[k] = 0.5;
                    ms[k] = 0.5;
                } else {
                    pc[k] = sigmoid(lg[i * (t1 - 1) + k - 1]);
                    ms[k] = sigmoid(mastery[i * (t1 - 1) + k - 1]);
                }
            }
            out.add(new Prediction(pc, ms));
        }
        return out;
    }

    private static double sigmoid(double z) {
        return 1 / (1 + Math.exp(-z));
    }

    /**
     * The fitted BKT parameters of a skill.
     *
     * @param skill the skill
     * @return {@code {L0, learn, forget, slip, guess}}, or {@code null} for neural models
     */
    public double[] bktParameters(int skill) {
        return bkt == null ? null : bkt[skill].clone();
    }
}
