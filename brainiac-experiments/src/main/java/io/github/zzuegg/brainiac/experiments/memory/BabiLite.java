package io.github.zzuegg.brainiac.experiments.memory;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A bAbI-like story generator with one, two and three supporting facts (Weston et al., "Towards
 * AI-Complete Question Answering", <a href="https://arxiv.org/abs/1502.05698">arXiv:1502.05698</a>).
 * Four people move between four places and pick up and drop three objects:
 * {@code "P went L"}, {@code "P got O"}, {@code "P dropped O"}. Questions:
 *
 * <ul>
 *   <li>task 1, one fact: {@code "where is P"} (the person's last move);</li>
 *   <li>task 2, two facts: {@code "where is O"} (who holds it, and where they are; or where it
 *       was dropped);</li>
 *   <li>task 3, three facts: {@code "where was O before L"} (the object's place before it
 *       reached {@code L}).</li>
 * </ul>
 *
 * <p>Stories are right-aligned in {@link #SENTENCES} slots (the newest sentence last), each of
 * up to {@link #WORDS} ids with 0 as padding. Answers are place indices 0 to 3.
 */
public final class BabiLite {
    /** Memory slots per story. */
    public static final int SENTENCES = 10;
    /** Words per sentence or question. */
    public static final int WORDS = 5;
    /** The vocabulary size. */
    public static final int VOCAB = 19;
    /** The number of answers (places). */
    public static final int ANSWERS = 4;

    static final int WENT = 1;
    static final int GOT = 2;
    static final int DROPPED = 3;
    static final int WHERE = 4;
    static final int IS = 5;
    static final int WAS = 6;
    static final int BEFORE = 7;
    static final int PERSON = 8;
    static final int PLACE = 12;
    static final int OBJECT = 16;

    /**
     * A batch of questions.
     *
     * @param stories {@code [batch][SENTENCES][WORDS]}
     * @param questions {@code [batch][WORDS]}
     * @param values the value token of each memory slot (its last word), {@code [batch][SENTENCES]}
     * @param answers the place index of each answer
     * @param tasks the task (1 to 3) of each question
     */
    public record Batch(int[][][] stories, int[][] questions, int[][] values, int[] answers, int[] tasks) {
    }

    private BabiLite() {
    }

    /**
     * Generates questions of the given tasks, drawn uniformly.
     *
     * @param n the questions
     * @param random the random source
     * @param tasks the tasks to draw from (1, 2, 3)
     * @return the batch
     */
    public static Batch generate(int n, SplittableRandom random, int... tasks) {
        int[][][] stories = new int[n][][];
        int[][] questions = new int[n][];
        int[][] values = new int[n][];
        int[] answers = new int[n];
        int[] taskOf = new int[n];
        for (int i = 0; i < n; i++) {
            int task = tasks[random.nextInt(tasks.length)];
            Object[] q = null;
            while (q == null) {
                q = story(task, random);
            }
            stories[i] = (int[][]) q[0];
            questions[i] = (int[]) q[1];
            answers[i] = (int) q[2];
            taskOf[i] = task;
            values[i] = new int[SENTENCES];
            for (int s = 0; s < SENTENCES; s++) {
                values[i][s] = stories[i][s][2];
            }
        }
        return new Batch(stories, questions, values, answers, taskOf);
    }

    /** Simulates one story and asks one question of the task, or returns null if none fits. */
    private static Object[] story(int task, SplittableRandom r) {
        int[] personAt = {-1, -1, -1, -1};
        int[] holder = {-1, -1, -1};
        int[] objectAt = {-1, -1, -1};
        List<List<Integer>> history = new ArrayList<>();
        for (int o = 0; o < 3; o++) {
            history.add(new ArrayList<>());
        }
        List<int[]> sentences = new ArrayList<>();
        int length = 6 + r.nextInt(SENTENCES - 5);
        while (sentences.size() < length) {
            int p = r.nextInt(4);
            int kind = r.nextInt(10);
            if (kind < 5 || personAt[p] < 0) {
                int l = r.nextInt(4);
                if (l == personAt[p]) {
                    continue;
                }
                personAt[p] = l;
                sentences.add(new int[] {PERSON + p, WENT, PLACE + l});
                for (int o = 0; o < 3; o++) {
                    if (holder[o] == p) {
                        note(history.get(o), l);
                    }
                }
            } else if (kind < 8) {
                int o = r.nextInt(3);
                if (holder[o] >= 0 || (objectAt[o] >= 0 && objectAt[o] != personAt[p])) {
                    continue;
                }
                holder[o] = p;
                objectAt[o] = -1;
                sentences.add(new int[] {PERSON + p, GOT, OBJECT + o});
                note(history.get(o), personAt[p]);
            } else {
                int o = -1;
                for (int c = 0; c < 3; c++) {
                    if (holder[c] == p) {
                        o = c;
                    }
                }
                if (o < 0) {
                    continue;
                }
                holder[o] = -1;
                objectAt[o] = personAt[p];
                sentences.add(new int[] {PERSON + p, DROPPED, OBJECT + o});
            }
        }
        int[] question;
        int answer;
        switch (task) {
            case 1 -> {
                int p = pick(r, i -> personAt[i] >= 0, 4);
                if (p < 0) {
                    return null;
                }
                question = new int[] {WHERE, IS, PERSON + p};
                answer = personAt[p];
            }
            case 2 -> {
                int o = pick(r, i -> holder[i] >= 0 || objectAt[i] >= 0, 3);
                if (o < 0) {
                    return null;
                }
                question = new int[] {WHERE, IS, OBJECT + o};
                answer = holder[o] >= 0 ? personAt[holder[o]] : objectAt[o];
            }
            default -> {
                int o = pick(r, i -> history.get(i).size() >= 2, 3);
                if (o < 0) {
                    return null;
                }
                List<Integer> h = history.get(o);
                int idx = -1;
                for (int k = h.size() - 1; k >= 1 && idx < 0; k--) {
                    if (h.indexOf(h.get(k)) == k && h.lastIndexOf(h.get(k)) == k) {
                        idx = k;
                    }
                }
                if (idx < 0) {
                    return null;
                }
                question = new int[] {WHERE, WAS, OBJECT + o, BEFORE, PLACE + h.get(idx)};
                answer = h.get(idx - 1);
            }
        }
        int[][] story = new int[SENTENCES][WORDS];
        int offset = SENTENCES - sentences.size();
        for (int s = 0; s < sentences.size(); s++) {
            System.arraycopy(sentences.get(s), 0, story[offset + s], 0, 3);
        }
        int[] q = new int[WORDS];
        System.arraycopy(question, 0, q, 0, question.length);
        return new Object[] {story, q, answer};
    }

    private static void note(List<Integer> h, int place) {
        if (place >= 0 && (h.isEmpty() || h.get(h.size() - 1).intValue() != place)) {
            h.add(place);
        }
    }

    private static int pick(SplittableRandom r, java.util.function.IntPredicate ok, int n) {
        List<Integer> c = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (ok.test(i)) {
                c.add(i);
            }
        }
        return c.isEmpty() ? -1 : c.get(r.nextInt(c.size()));
    }
}
