package io.github.zzuegg.brainiac.experiments.security;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.robust.AdversarialAttacks;
import io.github.zzuegg.brainiac.experimental.robust.AdversarialTraining;
import io.github.zzuegg.brainiac.experimental.security.AutoAttack;
import io.github.zzuegg.brainiac.experimental.security.BlackBoxAttacks;
import io.github.zzuegg.brainiac.experimental.security.DecisionAttacks;
import io.github.zzuegg.brainiac.experimental.security.QueryOracle;
import io.github.zzuegg.brainiac.experimental.security.TransferAttacks;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Wave 11 items 11.1.7 to 11.1.10: black-box, decision-based, transfer and AutoAttack
 * evaluations of MLP victims on {@link ToyTasks#digits} (8x8, 10 classes), trained normally or
 * with PGD adversarial training.
 *
 * <ul>
 *   <li>{@link #scoreBased}: NES, SimBA (pixel and DCT) and Square against white-box PGD and
 *       random sign noise of the same {@code ℓ∞} norm: success rate and median queries.</li>
 *   <li>{@link #decisionBased}: the boundary attack and HopSkipJump (label only, {@code ℓ2})
 *       against the white-box minimal {@code ℓ2} found by a PGD binary search.</li>
 *   <li>{@link #transfer}: PGD and MI-FGSM crafted on one surrogate or a three-model ensemble,
 *       transferred to a held-out target, plus a universal perturbation's fooling rate.</li>
 *   <li>{@link #autoAttack}: PGD-10, PGD-50 with restarts and the AutoAttack ensemble.</li>
 * </ul>
 *
 * <p>The MLP replaces the {@code VisionShootout} CNN to keep each test near 10 s on a shared CPU.
 */
public final class AttackShootout {

    /**
     * One attack's result.
     *
     * @param attack the attack
     * @param success the success rate on correctly classified test images
     * @param medianQueries the median queries per example ({@code 0} for white-box)
     */
    public record Score(String attack, double success, double medianQueries) {
    }

    /**
     * Median {@code ℓ2} distance against queries.
     *
     * @param attack the attack
     * @param budgets the query budgets
     * @param median the median distance at each budget
     */
    public record Curve(String attack, long[] budgets, double[] median) {
    }

    /**
     * Transfer results.
     *
     * @param attack the attack
     * @param success the target's misclassification rate on images it got right
     */
    public record Transfer(String attack, double success) {
    }

    /**
     * An evaluation row.
     *
     * @param evaluation the evaluation
     * @param robust robust accuracy
     */
    public record Robust(String evaluation, double robust) {
    }

    private static final int SIDE = 8;

    private AttackShootout() {
    }

    /** A victim and its data. */
    record Victim(UnaryOperator<Var> logits, float[][] x, int[] y, float[][] pool) {
    }

    static ToyTasks.Data[] data(long seed) {
        return new ToyTasks.Data[] {ToyTasks.digits(10, 100, SIDE, 0.05, seed, seed + 1),
                ToyTasks.digits(10, 30, SIDE, 0.05, seed, seed + 2)};
    }

    static UnaryOperator<Var> train(ToyTasks.Data train, int hidden, AdversarialTraining.Method method,
            float epsilon, int steps, long seed) {
        SimpleMlp m = SimpleMlp.relu(seed, SIDE * SIDE, hidden, 10);
        UnaryOperator<Var> f = m::forward;
        AdversarialTraining at = AdversarialTraining.of(f, m.parameters(), Optimizer.adam(3e-3f)).method(method)
                .epsilon(epsilon).steps(5).seed(seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            float[][] x = new float[64][];
            float[] y = new float[64];
            for (int i = 0; i < 64; i++) {
                int j = r.nextInt(train.x().length);
                x[i] = train.x()[j];
                y[i] = train.y()[j];
            }
            at.step(Tensor.of(x), Tensor.of(Shape.of(64), y));
        }
        return f;
    }

    /** Correctly classified test images, at most {@code max}. */
    static Victim victim(UnaryOperator<Var> f, ToyTasks.Data test, ToyTasks.Data train, int max) {
        int[] pred = TransferAttacks.predict(f, test.x());
        List<Integer> keep = new ArrayList<>();
        for (int i = 0; i < pred.length && keep.size() < max; i++) {
            if (pred[i] == test.y()[i]) {
                keep.add(i);
            }
        }
        float[][] x = new float[keep.size()][];
        int[] y = new int[keep.size()];
        for (int k = 0; k < x.length; k++) {
            x[k] = test.x()[keep.get(k)];
            y[k] = test.y()[keep.get(k)];
        }
        return new Victim(f, x, y, train.x());
    }

    static Tensor ids(int[] y) {
        float[] f = new float[y.length];
        for (int i = 0; i < y.length; i++) {
            f[i] = y[i];
        }
        return Tensor.of(Shape.of(y.length), f);
    }

    /**
     * Score-based attacks.
     *
     * @param robustVictim attack the PGD-trained victim
     * @param epsilon the {@code ℓ∞} budget
     * @param budget queries per example
     * @param examples attacked images
     * @param seed the seed
     * @return PGD, RANDOM, NES, SIMBA_PIXEL, SIMBA_DCT, SQUARE
     */
    public static List<Score> scoreBased(boolean robustVictim, float epsilon, int budget, int examples, long seed) {
        ToyTasks.Data[] d = data(seed);
        UnaryOperator<Var> f = train(d[0], 64, robustVictim ? AdversarialTraining.Method.PGD_AT
                : AdversarialTraining.Method.STANDARD, 0.1f, 300, seed);
        Victim v = victim(f, d[1], d[0], examples);
        List<Score> out = new ArrayList<>();
        Tensor x = Tensor.of(v.x());
        Tensor ids = ids(v.y());
        Tensor adv = AdversarialAttacks.pgd(f, x, ids, AdversarialAttacks.Norm.LINF, epsilon, 2.5f * epsilon / 20, 20,
                1, new SplittableRandom(seed));
        out.add(new Score("PGD-20 (white-box)", 1 - AdversarialAttacks.accuracy(f, adv, ids), 0));
        out.add(row("RANDOM", BlackBoxAttacks.randomNoise(new QueryOracle(f, v.x().length), v.x(), v.y(), epsilon,
                seed)));
        out.add(row("NES", BlackBoxAttacks.nes(new QueryOracle(f, v.x().length), v.x(), v.y(), epsilon, budget, 10,
                0.01f, seed)));
        out.add(row("SIMBA_PIXEL", BlackBoxAttacks.simba(new QueryOracle(f, v.x().length), v.x(), v.y(), SIDE,
                BlackBoxAttacks.Basis.PIXEL, epsilon, budget, seed)));
        out.add(row("SIMBA_DCT", BlackBoxAttacks.simba(new QueryOracle(f, v.x().length), v.x(), v.y(), SIDE,
                BlackBoxAttacks.Basis.DCT, epsilon, budget, seed)));
        out.add(row("SQUARE", BlackBoxAttacks.square(new QueryOracle(f, v.x().length), v.x(), v.y(), SIDE, true,
                epsilon, budget, 0.3, seed)));
        return out;
    }

    private static Score row(String name, BlackBoxAttacks.Result r) {
        return new Score(name, r.successRate(), r.medianQueries());
    }

    /**
     * Decision-based attacks on the standard victim.
     *
     * @param budgets query budgets reported (the largest is the run budget)
     * @param examples attacked images
     * @param seed the seed
     * @return BOUNDARY, HOP_SKIP_JUMP and the white-box reference ({@code PGD_L2_MIN}, one value)
     */
    public static List<Curve> decisionBased(long[] budgets, int examples, long seed) {
        ToyTasks.Data[] d = data(seed);
        UnaryOperator<Var> f = train(d[0], 64, AdversarialTraining.Method.STANDARD, 0.1f, 300, seed);
        Victim v = victim(f, d[1], d[0], examples);
        int budget = (int) budgets[budgets.length - 1];
        List<Curve> out = new ArrayList<>();
        for (DecisionAttacks.Method m : DecisionAttacks.Method.values()) {
            DecisionAttacks.Result r = DecisionAttacks.run(m, new QueryOracle(f, v.x().length), v.x(), v.y(), v.pool(),
                    budget, seed);
            double[] med = new double[budgets.length];
            for (int b = 0; b < budgets.length; b++) {
                med[b] = r.medianDistance(budgets[b]);
            }
            out.add(new Curve(m.name(), budgets, med));
        }
        out.add(new Curve("PGD_L2_MIN", new long[] {0}, new double[] {minimalL2(f, v.x(), v.y(), seed)}));
        return out;
    }

    /** The median minimal {@code ℓ2} radius at which PGD-20 succeeds, by bisection per example. */
    static double minimalL2(UnaryOperator<Var> f, float[][] x, int[] y, long seed) {
        int n = x.length;
        double[] lo = new double[n];
        double[] hi = new double[n];
        java.util.Arrays.fill(hi, 8);
        Tensor xs = Tensor.of(x);
        Tensor ids = ids(y);
        for (int it = 0; it < 7; it++) {
            double[] mid = new double[n];
            for (int i = 0; i < n; i++) {
                mid[i] = 0.5 * (lo[i] + hi[i]);
            }
            boolean[] fooled = fooledAt(f, xs, ids, mid, seed + it);
            for (int i = 0; i < n; i++) {
                if (fooled[i]) {
                    hi[i] = mid[i];
                } else {
                    lo[i] = mid[i];
                }
            }
        }
        java.util.Arrays.sort(hi);
        return n % 2 == 1 ? hi[n / 2] : 0.5 * (hi[n / 2 - 1] + hi[n / 2]);
    }

    private static boolean[] fooledAt(UnaryOperator<Var> f, Tensor x, Tensor ids, double[] radius, long seed) {
        int n = radius.length;
        boolean[] out = new boolean[n];
        float[][] rows = toRows(x);
        int[] y = new int[n];
        float[] idv = ids.toArray();
        for (int i = 0; i < n; i++) {
            y[i] = (int) idv[i];
        }
        for (int i = 0; i < n; i++) {
            float eps = (float) radius[i];
            Tensor xi = Tensor.of(new float[][] {rows[i]});
            Tensor yi = Tensor.of(Shape.of(1), new float[] {y[i]});
            Tensor adv = AdversarialAttacks.pgd(f, xi, yi, AdversarialAttacks.Norm.L2, eps, 2.5f * eps / 20, 20, 1,
                    new SplittableRandom(seed + i));
            out[i] = AdversarialAttacks.accuracy(f, adv, yi) < 0.5;
        }
        return out;
    }

    private static float[][] toRows(Tensor x) {
        int n = x.shape().dim(0);
        int d = Math.toIntExact(x.size() / n);
        float[] a = x.toArray();
        float[][] out = new float[n][d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a, i * d, out[i], 0, d);
        }
        return out;
    }

    /**
     * Transfer attacks and a universal perturbation.
     *
     * @param epsilon the {@code ℓ∞} budget
     * @param uapEpsilon the universal perturbation's budget
     * @param seed the seed
     * @return RANDOM, PGD (1 surrogate), MI-FGSM (1 surrogate), PGD (ensemble), MI-FGSM
     *     (ensemble), WHITE_BOX_PGD and UAP (fooling rate on held-out images)
     */
    public static List<Transfer> transfer(float epsilon, float uapEpsilon, long seed) {
        ToyTasks.Data[] d = data(seed);
        UnaryOperator<Var> target = train(d[0], 64, AdversarialTraining.Method.STANDARD, 0.1f, 300, seed + 100);
        List<UnaryOperator<Var>> surrogates = List.of(
                train(d[0], 32, AdversarialTraining.Method.STANDARD, 0.1f, 300, seed + 200),
                train(d[0], 48, AdversarialTraining.Method.STANDARD, 0.1f, 300, seed + 300),
                train(d[0], 96, AdversarialTraining.Method.STANDARD, 0.1f, 300, seed + 400));
        Victim v = victim(target, d[1], d[0], 200);
        Tensor x = Tensor.of(v.x());
        Tensor ids = ids(v.y());
        int steps = 10;
        List<Transfer> out = new ArrayList<>();
        BlackBoxAttacks.Result noise = BlackBoxAttacks.randomNoise(new QueryOracle(target, v.x().length), v.x(), v.y(),
                epsilon, seed);
        out.add(new Transfer("RANDOM", noise.successRate()));
        UnaryOperator<Var> one = surrogates.get(0);
        UnaryOperator<Var> ens = TransferAttacks.ensemble(surrogates);
        out.add(new Transfer("PGD (1 surrogate)", TransferAttacks.successRate(target, x,
                TransferAttacks.miFgsm(one, x, ids, epsilon, steps, 0), ids)));
        out.add(new Transfer("MI-FGSM (1 surrogate)", TransferAttacks.successRate(target, x,
                TransferAttacks.miFgsm(one, x, ids, epsilon, steps, 1), ids)));
        out.add(new Transfer("PGD (ensemble of 3)", TransferAttacks.successRate(target, x,
                TransferAttacks.miFgsm(ens, x, ids, epsilon, steps, 0), ids)));
        out.add(new Transfer("MI-FGSM (ensemble of 3)", TransferAttacks.successRate(target, x,
                TransferAttacks.miFgsm(ens, x, ids, epsilon, steps, 1), ids)));
        Tensor wb = AdversarialAttacks.pgd(target, x, ids, AdversarialAttacks.Norm.LINF, epsilon, 2.5f * epsilon / 20,
                20, 1, new SplittableRandom(seed));
        out.add(new Transfer("WHITE_BOX_PGD", TransferAttacks.successRate(target, x, wb, ids)));
        float[][] fit = new float[200][];
        SplittableRandom r = new SplittableRandom(seed + 9);
        for (int i = 0; i < fit.length; i++) {
            fit[i] = d[0].x()[r.nextInt(d[0].x().length)];
        }
        float[] uap = TransferAttacks.universal(target, fit, uapEpsilon, 3, 0.02f);
        out.add(new Transfer("UAP", TransferAttacks.foolingRate(target, d[1].x(), uap)));
        return out;
    }

    /**
     * AutoAttack against fixed-step PGD.
     *
     * @param method the victim's training
     * @param epsilon the training and evaluation {@code ℓ∞} budget
     * @param examples evaluated test images
     * @param seed the seed
     * @return CLEAN, PGD-10, PGD-50x3, APGD-CE, +APGD-DLR-T, AUTOATTACK (+SQUARE)
     */
    public static List<Robust> autoAttack(AdversarialTraining.Method method, float epsilon, int examples,
            long seed) {
        ToyTasks.Data[] d = data(seed);
        UnaryOperator<Var> f = train(d[0], 64, method, epsilon, 300, seed);
        float[][] x = new float[examples][];
        int[] y = new int[examples];
        System.arraycopy(d[1].x(), 0, x, 0, examples);
        System.arraycopy(d[1].y(), 0, y, 0, examples);
        return evaluate(f, x, y, epsilon, seed);
    }

    /**
     * The evaluation rows for one model.
     *
     * @param f the model
     * @param x test inputs
     * @param y test classes
     * @param epsilon the {@code ℓ∞} budget
     * @param seed the seed
     * @return CLEAN, PGD-10, PGD-50x3, APGD-CE, +APGD-DLR-T, AUTOATTACK
     */
    public static List<Robust> evaluate(UnaryOperator<Var> f, float[][] x, int[] y, float epsilon, long seed) {
        Tensor xs = Tensor.of(x);
        Tensor ids = ids(y);
        List<Robust> out = new ArrayList<>();
        out.add(new Robust("CLEAN", AdversarialAttacks.accuracy(f, xs, ids)));
        Tensor p10 = AdversarialAttacks.pgd(f, xs, ids, AdversarialAttacks.Norm.LINF, epsilon, 2.5f * epsilon / 10, 10,
                1, new SplittableRandom(seed));
        out.add(new Robust("PGD-10", AdversarialAttacks.accuracy(f, p10, ids)));
        Tensor p50 = AdversarialAttacks.pgd(f, xs, ids, AdversarialAttacks.Norm.LINF, epsilon, 2.5f * epsilon / 50, 50,
                3, new SplittableRandom(seed + 1));
        out.add(new Robust("PGD-50x3", Math.min(AdversarialAttacks.accuracy(f, p50, ids),
                AdversarialAttacks.accuracy(f, p10, ids))));
        AutoAttack.Report rep = AutoAttack.evaluate(f, x, y, SIDE, epsilon, 50, 500, seed + 2);
        out.add(new Robust("APGD-CE", rep.afterCe()));
        out.add(new Robust("+APGD-DLR-T", rep.afterDlr()));
        out.add(new Robust("AUTOATTACK", rep.afterSquare()));
        return out;
    }
}
