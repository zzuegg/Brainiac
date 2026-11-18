package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Simulated students answering skill-tagged questions, for knowledge tracing with known ground
 * truth. Each question belongs to one skill and has a Rasch difficulty {@code b_q}; each student
 * has an ability {@code theta}. Per skill, a student is in a binary mastery state (BKT-style):
 * a mastered skill is answered correctly with probability {@code 1 - slip}, an unmastered one
 * with {@code guess}, both modulated by {@code theta - b_q} (2-PL style), so item effects exist
 * that pure BKT cannot model. After each practice an unmastered skill is learnt with
 * probability {@code learn} (higher when the previous skill is mastered, if {@code transfer} is
 * on: knowledge transfers along a prerequisite chain) and a mastered one is forgotten with
 * probability {@code forget}. Students practise in runs, mostly staying on a skill. The mastery
 * state before each answer is kept for scoring.
 */
public final class LearnerTraces {

    private LearnerTraces() {
    }

    /**
     * One student's answers.
     *
     * @param question the question at each step
     * @param skill its skill
     * @param correct 1 if answered correctly
     * @param mastered whether the skill was mastered just before the answer
     * @param ability the student's ability
     */
    public record Trace(int[] question, int[] skill, int[] correct, boolean[] mastered, double ability) {
    }

    /**
     * A simulated cohort.
     *
     * @param traces one trace per student
     * @param questionSkill the skill of each question
     * @param difficulty the Rasch difficulty of each question
     * @param skills the number of skills
     */
    public record Cohort(List<Trace> traces, int[] questionSkill, double[] difficulty, int skills) {

        /**
         * The number of questions.
         *
         * @return the question count
         */
        public int questions() {
            return questionSkill.length;
        }
    }

    /**
     * Item parameters shared by cohorts drawn from the same curriculum.
     *
     * @param skills the number of skills
     * @param questions the number of questions (spread evenly over skills)
     * @param seed the seed
     * @return {@code {question skill (as doubles), difficulty}}
     */
    private static double[][] items(int skills, int questions, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[] qs = new double[questions];
        double[] b = new double[questions];
        for (int q = 0; q < questions; q++) {
            qs[q] = q % skills;
            b[q] = r.nextGaussian();
        }
        return new double[][] {qs, b};
    }

    /**
     * Simulates a cohort.
     *
     * @param students students
     * @param skills skills
     * @param questions questions
     * @param length answers per student
     * @param forget the forgetting probability per practice of a mastered skill
     * @param transfer whether mastering skill {@code k - 1} speeds up learning skill {@code k}
     * @param itemSeed the seed of the item parameters (share it between train and test cohorts)
     * @param seed the seed of the students
     * @return the cohort
     */
    public static Cohort simulate(int students, int skills, int questions, int length, double forget,
            boolean transfer, long itemSeed, long seed) {
        double[][] it = items(skills, questions, itemSeed);
        int[] qskill = new int[questions];
        for (int q = 0; q < questions; q++) {
            qskill[q] = (int) it[0][q];
        }
        double[] b = it[1];
        List<List<Integer>> bySkill = new ArrayList<>();
        for (int s = 0; s < skills; s++) {
            bySkill.add(new ArrayList<>());
        }
        for (int q = 0; q < questions; q++) {
            bySkill.get(qskill[q]).add(q);
        }
        SplittableRandom r = new SplittableRandom(seed);
        List<Trace> traces = new ArrayList<>();
        for (int st = 0; st < students; st++) {
            double theta = 0.8 * r.nextGaussian();
            boolean[] know = new boolean[skills];
            for (int s = 0; s < skills; s++) {
                know[s] = r.nextDouble() < 0.15;
            }
            int[] qs = new int[length];
            int[] ss = new int[length];
            int[] cs = new int[length];
            boolean[] ms = new boolean[length];
            int cur = r.nextInt(Math.max(1, skills / 3));
            for (int t = 0; t < length; t++) {
                if (t > 0 && r.nextDouble() < 0.25) {
                    cur = r.nextDouble() < 0.6 ? Math.min(skills - 1, cur + 1) : r.nextInt(skills);
                }
                List<Integer> pool = bySkill.get(cur);
                int q = pool.get(r.nextInt(pool.size()));
                double z = theta - b[q];
                double slip = 0.2 / (1 + Math.exp(z));
                double guess = 0.45 / (1 + Math.exp(-z));
                boolean m = know[cur];
                boolean ok = r.nextDouble() < (m ? 1 - slip : guess);
                qs[t] = q;
                ss[t] = cur;
                cs[t] = ok ? 1 : 0;
                ms[t] = m;
                if (!m) {
                    double learn = 0.12 + (transfer && cur > 0 && know[cur - 1] ? 0.15 : 0)
                            - (transfer && cur > 0 && !know[cur - 1] ? 0.08 : 0);
                    if (r.nextDouble() < learn) {
                        know[cur] = true;
                    }
                } else if (r.nextDouble() < forget) {
                    know[cur] = false;
                }
            }
            traces.add(new Trace(qs, ss, cs, ms, theta));
        }
        return new Cohort(traces, qskill, b, skills);
    }
}
