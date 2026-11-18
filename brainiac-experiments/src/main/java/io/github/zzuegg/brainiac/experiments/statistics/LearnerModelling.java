package io.github.zzuegg.brainiac.experiments.statistics;

import io.github.zzuegg.brainiac.experimental.calibrate.Recalibrators;
import io.github.zzuegg.brainiac.experimental.data.LearnerTraces;
import io.github.zzuegg.brainiac.experimental.llm.IrtEvaluator;
import io.github.zzuegg.brainiac.experimental.stats.KnowledgeTracing;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Knowledge tracing (item 12.1.15): BKT (EM with forgetting), DKT (LSTM), simpleKT (causal
 * attention with Rasch embeddings) and a DKT2-style LSTM with an IRT readout against three
 * classical baselines (per-skill running accuracy, PFA logistic regression on success and failure
 * counts, and a static Rasch model whose ability is re-estimated from all earlier answers), on
 * simulated students with known mastery. Scored by next-answer AUC on steps after the first and
 * by the point-biserial correlation of the estimated mastery with the true hidden mastery.
 */
public final class LearnerModelling {

    private LearnerModelling() {
    }

    private static List<KnowledgeTracing.Sequence> sequences(LearnerTraces.Cohort c) {
        List<KnowledgeTracing.Sequence> out = new ArrayList<>();
        for (LearnerTraces.Trace t : c.traces()) {
            out.add(new KnowledgeTracing.Sequence(t.question(), t.skill(), t.correct()));
        }
        return out;
    }

    /**
     * Fits every method on one cohort and scores it on another drawn from the same items.
     *
     * @param trainStudents training students
     * @param testStudents test students
     * @param skills skills
     * @param questions questions
     * @param length answers per student
     * @param forget forgetting probability
     * @param transfer prerequisite transfer
     * @param epochs Adam steps of the neural models (0 skips them)
     * @param seed the seed
     * @return {@code {AUC, mastery correlation, fit seconds}} per method
     */
    public static Map<String, double[]> run(int trainStudents, int testStudents, int skills, int questions,
            int length, double forget, boolean transfer, int epochs, long seed) {
        LearnerTraces.Cohort train = LearnerTraces.simulate(trainStudents, skills, questions, length, forget,
                transfer, seed, seed + 1);
        LearnerTraces.Cohort test = LearnerTraces.simulate(testStudents, skills, questions, length, forget,
                transfer, seed, seed + 2);
        Map<String, double[]> out = new LinkedHashMap<>();
        Map<String, double[][]> base = baselines(train, test);
        base.forEach((k, v) -> out.put(k, score(test, v, v, 0)));
        List<KnowledgeTracing.Sequence> tr = sequences(train);
        List<KnowledgeTracing.Sequence> te = sequences(test);
        for (KnowledgeTracing.Model m : KnowledgeTracing.Model.values()) {
            if (epochs == 0 && m != KnowledgeTracing.Model.BKT) {
                continue;
            }
            long t0 = System.nanoTime();
            KnowledgeTracing kt = KnowledgeTracing.fit(m, tr, skills, questions,
                    m == KnowledgeTracing.Model.BKT ? 30 : epochs, seed);
            double secs = (System.nanoTime() - t0) / 1e9;
            List<KnowledgeTracing.Prediction> p = kt.predict(te);
            double[][] pc = new double[p.size()][];
            double[][] ms = new double[p.size()][];
            for (int i = 0; i < p.size(); i++) {
                pc[i] = p.get(i).correct();
                ms[i] = p.get(i).mastery();
            }
            out.put(m.name(), score(test, pc, ms, secs));
        }
        return out;
    }

    private static double[] score(LearnerTraces.Cohort test, double[][] pc, double[][] ms, double secs) {
        List<Double> s = new ArrayList<>();
        List<Boolean> y = new ArrayList<>();
        double sx = 0;
        double sy = 0;
        double sxx = 0;
        double syy = 0;
        double sxy = 0;
        int n = 0;
        for (int i = 0; i < test.traces().size(); i++) {
            LearnerTraces.Trace t = test.traces().get(i);
            for (int k = 1; k < t.correct().length; k++) {
                s.add(pc[i][k]);
                y.add(t.correct()[k] == 1);
                double a = ms[i][k];
                double b = t.mastered()[k] ? 1 : 0;
                sx += a;
                sy += b;
                sxx += a * a;
                syy += b * b;
                sxy += a * b;
                n++;
            }
        }
        double[] sc = new double[s.size()];
        boolean[] pos = new boolean[s.size()];
        for (int i = 0; i < sc.length; i++) {
            sc[i] = s.get(i);
            pos[i] = y.get(i);
        }
        double cov = sxy / n - sx / n * sy / n;
        double corr = cov / Math.sqrt(Math.max(1e-12, (sxx / n - sx * sx / n / n) * (syy / n - sy * sy / n / n)));
        return new double[] {Scores.auc(sc, pos), corr, secs};
    }

    private static double logit(double p) {
        return Math.log(p / (1 - p));
    }

    private static Map<String, double[][]> baselines(LearnerTraces.Cohort train, LearnerTraces.Cohort test) {
        int skills = train.skills();
        int nq = train.questions();
        Map<String, double[][]> out = new LinkedHashMap<>();
        // running accuracy per skill with a Laplace prior
        double[][] run = new double[test.traces().size()][];
        for (int i = 0; i < run.length; i++) {
            LearnerTraces.Trace t = test.traces().get(i);
            double[] ok = new double[skills];
            double[] n = new double[skills];
            run[i] = new double[t.correct().length];
            for (int k = 0; k < t.correct().length; k++) {
                int s = t.skill()[k];
                run[i][k] = (ok[s] + 1) / (n[s] + 2);
                ok[s] += t.correct()[k];
                n[s]++;
            }
        }
        out.put("running accuracy", run);
        // PFA: skill intercepts plus per-skill weights on log(1 + successes) and log(1 + failures)
        List<double[]> rows = new ArrayList<>();
        List<Double> ys = new ArrayList<>();
        for (LearnerTraces.Trace t : train.traces()) {
            double[] ok = new double[skills];
            double[] bad = new double[skills];
            for (int k = 0; k < t.correct().length; k++) {
                rows.add(pfaRow(t.skill()[k], ok, bad, skills));
                ys.add((double) t.correct()[k]);
                if (t.correct()[k] == 1) {
                    ok[t.skill()[k]]++;
                } else {
                    bad[t.skill()[k]]++;
                }
            }
        }
        double[] yy = new double[ys.size()];
        for (int i = 0; i < yy.length; i++) {
            yy[i] = ys.get(i);
        }
        double[] w = Recalibrators.logistic(rows.toArray(new double[0][]), yy, 1e-3);
        double[][] pfa = new double[test.traces().size()][];
        for (int i = 0; i < pfa.length; i++) {
            LearnerTraces.Trace t = test.traces().get(i);
            double[] ok = new double[skills];
            double[] bad = new double[skills];
            pfa[i] = new double[t.correct().length];
            for (int k = 0; k < t.correct().length; k++) {
                double[] f = pfaRow(t.skill()[k], ok, bad, skills);
                double z = 0;
                for (int j = 0; j < f.length; j++) {
                    z += w[j] * f[j];
                }
                pfa[i][k] = 1 / (1 + Math.exp(-z));
                if (t.correct()[k] == 1) {
                    ok[t.skill()[k]]++;
                } else {
                    bad[t.skill()[k]]++;
                }
            }
        }
        out.put("PFA", pfa);
        // static Rasch: difficulties from training accuracy per question, ability from all earlier answers
        double[] okq = new double[nq];
        double[] nqc = new double[nq];
        for (LearnerTraces.Trace t : train.traces()) {
            for (int k = 0; k < t.correct().length; k++) {
                okq[t.question()[k]] += t.correct()[k];
                nqc[t.question()[k]]++;
            }
        }
        double[] diff = new double[nq];
        for (int q = 0; q < nq; q++) {
            diff[q] = -logit((okq[q] + 1) / (nqc[q] + 2));
        }
        double[] ones = new double[nq];
        Arrays.fill(ones, 1);
        IrtEvaluator irt = new IrtEvaluator(ones, diff);
        double[][] rasch = new double[test.traces().size()][];
        for (int i = 0; i < rasch.length; i++) {
            LearnerTraces.Trace t = test.traces().get(i);
            int len = t.correct().length;
            rasch[i] = new double[len];
            boolean[] c = new boolean[len];
            for (int k = 0; k < len; k++) {
                c[k] = t.correct()[k] == 1;
            }
            for (int k = 0; k < len; k++) {
                double theta = k == 0 ? 0
                        : irt.ability(Arrays.copyOf(t.question(), k), Arrays.copyOf(c, k));
                rasch[i][k] = 1 / (1 + Math.exp(-(theta - diff[t.question()[k]])));
            }
        }
        out.put("static Rasch", rasch);
        return out;
    }

    private static double[] pfaRow(int s, double[] ok, double[] bad, int skills) {
        double[] f = new double[3 * skills];
        f[s] = 1;
        f[skills + s] = Math.log1p(ok[s]);
        f[2 * skills + s] = Math.log1p(bad[s]);
        return f;
    }
}
