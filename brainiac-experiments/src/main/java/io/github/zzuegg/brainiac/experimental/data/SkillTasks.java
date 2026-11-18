package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * Arithmetic and string skills with a known prerequisite graph, for skill-ordered data
 * selection and curricula: single-digit addition without a carry, then with a carry, then
 * multi-digit addition (which needs both), then multiplication of a two-digit number by a digit
 * (which needs multi-digit addition), plus a decoy skill, reversing a number, that none of them
 * needs.
 *
 * <p>Every problem has the same fixed layout so that skills share input and output slots: an
 * operation one-hot (add, multiply, reverse), then {@value #SLOTS} digit slots for each operand
 * and {@value #OUT} answer slots, least significant digit first, each slot a one-hot over ten
 * digits or blank. A model reads {@link #INPUT} features and predicts {@code OUT} slots of
 * {@value #CLASSES} classes.
 */
public final class SkillTasks {

    private SkillTasks() {
    }

    /** Operand digit slots. */
    public static final int SLOTS = 4;
    /** Answer slots. */
    public static final int OUT = 5;
    /** Classes per slot: ten digits and blank. */
    public static final int CLASSES = 11;
    private static final int BLANK = 10;
    private static final int OPS = 3;
    /** Input features. */
    public static final int INPUT = OPS + 2 * SLOTS * CLASSES;

    /** The skills. */
    public enum Skill {
        /** {@code a + b} for digits with {@code a + b < 10}. */
        ADD_DIGIT,
        /** {@code a + b} for digits with {@code a + b >= 10}. */
        CARRY,
        /** Two-digit plus two-digit addition. */
        ADD_MULTI,
        /** A two-digit number times a digit. */
        MULTIPLY,
        /** The decoy: the digits of a three-digit number reversed. */
        REVERSE
    }

    /**
     * The planted prerequisite graph: {@code [i][j]} is true when skill {@code i} is a
     * prerequisite of skill {@code j}.
     *
     * @return the adjacency matrix over {@link Skill#values()}
     */
    public static boolean[][] prerequisites() {
        int k = Skill.values().length;
        boolean[][] g = new boolean[k][k];
        g[Skill.ADD_DIGIT.ordinal()][Skill.CARRY.ordinal()] = true;
        g[Skill.ADD_DIGIT.ordinal()][Skill.ADD_MULTI.ordinal()] = true;
        g[Skill.CARRY.ordinal()][Skill.ADD_MULTI.ordinal()] = true;
        g[Skill.ADD_MULTI.ordinal()][Skill.MULTIPLY.ordinal()] = true;
        return g;
    }

    /**
     * A batch of problems.
     *
     * @param x inputs {@code [n, INPUT]}
     * @param y answer slot ids {@code [n * OUT]}
     * @param n problems
     */
    public record Batch(Tensor x, Tensor y, int n) {
    }

    private static void digits(float[] x, int row, int operand, long value, int count) {
        for (int s = 0; s < SLOTS; s++) {
            int d = s < count ? (int) (value % 10) : BLANK;
            value /= 10;
            x[row * INPUT + OPS + (operand * SLOTS + s) * CLASSES + d] = 1;
        }
    }

    private static void answer(float[] y, int row, long value, int count) {
        for (int s = 0; s < OUT; s++) {
            y[row * OUT + s] = s < count ? (int) (value % 10) : BLANK;
            value /= 10;
        }
    }

    private static int length(long v) {
        return v == 0 ? 1 : (int) Math.floor(Math.log10(v)) + 1;
    }

    /**
     * Samples problems of one skill.
     *
     * @param skill the skill
     * @param n problems
     * @param r the random source
     * @return the batch
     */
    public static Batch sample(Skill skill, int n, SplittableRandom r) {
        float[] x = new float[n * INPUT];
        float[] y = new float[n * OUT];
        for (int i = 0; i < n; i++) {
            switch (skill) {
                case ADD_DIGIT, CARRY -> {
                    int a;
                    int b;
                    do {
                        a = r.nextInt(10);
                        b = r.nextInt(10);
                    } while ((a + b >= 10) != (skill == Skill.CARRY));
                    x[i * INPUT] = 1;
                    digits(x, i, 0, a, 1);
                    digits(x, i, 1, b, 1);
                    answer(y, i, a + b, length(a + b));
                }
                case ADD_MULTI -> {
                    int a = 10 + r.nextInt(90);
                    int b = 10 + r.nextInt(90);
                    x[i * INPUT] = 1;
                    digits(x, i, 0, a, 2);
                    digits(x, i, 1, b, 2);
                    answer(y, i, a + b, length(a + b));
                }
                case MULTIPLY -> {
                    int a = 10 + r.nextInt(90);
                    int b = 2 + r.nextInt(8);
                    x[i * INPUT + 1] = 1;
                    digits(x, i, 0, a, 2);
                    digits(x, i, 1, b, 1);
                    answer(y, i, (long) a * b, length((long) a * b));
                }
                case REVERSE -> {
                    int a = 100 + r.nextInt(900);
                    x[i * INPUT + 2] = 1;
                    digits(x, i, 0, a, 3);
                    digits(x, i, 1, 0, 0);
                    int rev = (a % 10) * 100 + (a / 10 % 10) * 10 + a / 100;
                    for (int s = 0; s < OUT; s++) {
                        y[i * OUT + s] = s < 3 ? (s == 0 ? rev % 10 : s == 1 ? rev / 10 % 10 : rev / 100) : BLANK;
                    }
                }
            }
        }
        return new Batch(Tensor.of(Shape.of(n, INPUT), x), Tensor.of(Shape.of(n * OUT), y), n);
    }

    /**
     * Addition of two {@code digits}-digit numbers (1 to 4), the tasks of a length curriculum.
     *
     * @param digits operand length
     * @param n problems
     * @param r the random source
     * @return the batch
     */
    public static Batch addition(int digits, int n, SplittableRandom r) {
        float[] x = new float[n * INPUT];
        float[] y = new float[n * OUT];
        long lo = digits == 1 ? 0 : (long) Math.pow(10, digits - 1);
        long hi = (long) Math.pow(10, digits);
        for (int i = 0; i < n; i++) {
            long a = lo + r.nextLong(hi - lo);
            long b = lo + r.nextLong(hi - lo);
            x[i * INPUT] = 1;
            digits(x, i, 0, a, digits);
            digits(x, i, 1, b, digits);
            answer(y, i, a + b, length(a + b));
        }
        return new Batch(Tensor.of(Shape.of(n, INPUT), x), Tensor.of(Shape.of(n * OUT), y), n);
    }

    /**
     * Unlearnable problems: two-digit additions whose answers are random digits.
     *
     * @param n problems
     * @param r the random source
     * @return the batch
     */
    public static Batch randomLabels(int n, SplittableRandom r) {
        Batch b = addition(2, n, r);
        float[] y = b.y().toArray();
        for (int i = 0; i < y.length; i++) {
            y[i] = r.nextInt(10);
        }
        return new Batch(b.x(), Tensor.of(Shape.of(y.length), y), n);
    }

    /**
     * The mean cross-entropy over answer slots.
     *
     * @param logits {@code [n, OUT * CLASSES]}
     * @param batch the batch
     * @return the loss
     */
    public static Var loss(Var logits, Batch batch) {
        return Functional.crossEntropy(logits.reshape(batch.n() * OUT, CLASSES), batch.y());
    }

    /**
     * Per-slot and whole-answer accuracy.
     *
     * @param logits {@code [n, OUT * CLASSES]}
     * @param batch the batch
     * @return {@code {slot accuracy, exact-match accuracy}}
     */
    public static double[] accuracy(Tensor logits, Batch batch) {
        float[] l = logits.toArray();
        float[] y = batch.y().toArray();
        int slots = 0;
        int exact = 0;
        for (int i = 0; i < batch.n(); i++) {
            boolean all = true;
            for (int s = 0; s < OUT; s++) {
                int off = (i * OUT + s) * CLASSES;
                int best = 0;
                for (int c = 1; c < CLASSES; c++) {
                    best = l[off + c] > l[off + best] ? c : best;
                }
                boolean ok = best == (int) y[i * OUT + s];
                slots += ok ? 1 : 0;
                all &= ok;
            }
            exact += all ? 1 : 0;
        }
        return new double[] {slots / (double) (batch.n() * OUT), exact / (double) batch.n()};
    }
}
