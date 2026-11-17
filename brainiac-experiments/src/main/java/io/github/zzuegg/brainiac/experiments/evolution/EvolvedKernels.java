package io.github.zzuegg.brainiac.experiments.evolution;

import io.github.zzuegg.brainiac.experimental.evolve.MapElites;
import io.github.zzuegg.brainiac.experimental.evolve.ProgramEvolution;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * AlphaEvolve-style evolution of the two kernels every network leans on, softmax and matrix
 * multiply (item 5.5), with {@link ProgramEvolution}.
 *
 * <p><b>Softmax</b> kernels are straight-line programs over vector and scalar registers
 * ({@code s0 = max(v0); v1 = v0 - s0; v1 = exp(v1); ...}). The evaluator runs a kernel on
 * test vectors, half of them with logits large enough to overflow {@code exp}, and scores
 * correctness first and cost (memory passes, an {@code exp} counting four, a division two)
 * second. The cheap naive kernel {@code exp, sum, divide} is right only on the small half; the
 * best kernel subtracts the maximum first and multiplies by a reciprocal.
 *
 * <p><b>Matmul</b> kernels are loop nests: a loop order and tile sizes for {@code i, j, k}.
 * The evaluator runs the nest on real matrices (checking the product) through a simulated
 * LRU data cache and scores the misses; the baseline is the textbook {@code ijk} loop.
 *
 * <p>Three searches get the same number of evaluations: the AlphaEvolve loop, where the
 * proposer (a stand-in for the code model) sees the parent and inspirations from the
 * MAP-Elites archive and splices in their code; the same archive with blind random mutation;
 * and AutoML-Zero's regularized evolution with random mutation.
 */
public final class EvolvedKernels {

    private EvolvedKernels() {
    }

    // ---------------------------------------------------------------- softmax

    /** A softmax kernel's operations. */
    public enum Op {
        /** {@code s[d] = max(v[a])}. */
        MAX,
        /** {@code s[d] = sum(v[a])}. */
        SUM,
        /** {@code v[d] = v[a] - s[b]}. */
        SUBS,
        /** {@code v[d] = v[a] * s[b]}. */
        MULS,
        /** {@code v[d] = v[a] / s[b]}. */
        DIVS,
        /** {@code v[d] = exp(v[a])}. */
        EXP,
        /** {@code s[d] = 1 / s[a]}. */
        RECIP
    }

    /**
     * One instruction.
     *
     * @param op the operation
     * @param d the destination register
     * @param a the first source
     * @param b the scalar source, for SUBS, MULS and DIVS
     */
    public record Instr(Op op, int d, int a, int b) {
        @Override
        public String toString() {
            return switch (op) {
                case MAX -> "s" + d + " = max(v" + a + ")";
                case SUM -> "s" + d + " = sum(v" + a + ")";
                case SUBS -> "v" + d + " = v" + a + " - s" + b;
                case MULS -> "v" + d + " = v" + a + " * s" + b;
                case DIVS -> "v" + d + " = v" + a + " / s" + b;
                case EXP -> "v" + d + " = exp(v" + a + ")";
                case RECIP -> "s" + d + " = 1 / s" + a;
            };
        }
    }

    static final int REGS = 3;
    static final int MAX_LENGTH = 10;
    static final int WIDTH = 16;

    /**
     * Runs a kernel; v0 holds the input, v1 is the output.
     *
     * @param program the kernel
     * @param x the logits
     * @return the output
     */
    public static double[] runSoftmax(List<Instr> program, double[] x) {
        double[][] v = new double[REGS][x.length];
        double[] s = new double[REGS];
        v[0] = x.clone();
        for (Instr in : program) {
            switch (in.op()) {
                case MAX -> s[in.d()] = java.util.Arrays.stream(v[in.a()]).max().orElse(0);
                case SUM -> s[in.d()] = java.util.Arrays.stream(v[in.a()]).sum();
                case RECIP -> s[in.d()] = 1 / s[in.a()];
                default -> {
                    double[] src = v[in.a()];
                    double[] out = new double[src.length];
                    for (int i = 0; i < out.length; i++) {
                        out[i] = switch (in.op()) {
                            case SUBS -> src[i] - s[in.b()];
                            case MULS -> src[i] * s[in.b()];
                            case DIVS -> src[i] / s[in.b()];
                            default -> Math.exp(src[i]);
                        };
                    }
                    v[in.d()] = out;
                }
            }
        }
        return v[1];
    }

    /**
     * @param program a kernel
     * @return its cost per element: passes over memory, exp counting 4 and division 2
     */
    public static double softmaxCost(List<Instr> program) {
        double c = 0;
        for (Instr in : program) {
            c += switch (in.op()) {
                case EXP -> 4;
                case DIVS -> 2;
                case RECIP -> 2.0 / WIDTH;
                default -> 1;
            };
        }
        return c;
    }

    static List<double[]> softmaxTests() {
        SplittableRandom r = new SplittableRandom(99);
        List<double[]> tests = new ArrayList<>();
        for (int t = 0; t < 8; t++) {
            double scale = t < 4 ? 2 : 400;   // the second half overflows a naive exp
            double[] x = new double[WIDTH];
            for (int i = 0; i < WIDTH; i++) {
                x[i] = scale * (r.nextDouble() * 2 - 1) + (t >= 4 ? 500 : 0);
            }
            tests.add(x);
        }
        return tests;
    }

    private static final List<double[]> TESTS = softmaxTests();

    /**
     * @param program a kernel
     * @return the share of test vectors it gets right (max error 1e-9, no NaN)
     */
    public static double softmaxCorrectness(List<Instr> program) {
        int right = 0;
        for (double[] x : TESTS) {
            double m = java.util.Arrays.stream(x).max().orElse(0);
            double z = 0;
            for (double xi : x) {
                z += Math.exp(xi - m);
            }
            double[] out = runSoftmax(program, x);
            boolean ok = true;
            for (int i = 0; i < x.length; i++) {
                double want = Math.exp(x[i] - m) / z;
                ok &= Math.abs(out[i] - want) < 1e-9;
            }
            right += ok ? 1 : 0;
        }
        return right / (double) TESTS.size();
    }

    /**
     * Partial credit: the mean cosine similarity of the output to the true softmax (0 for
     * NaN or infinite outputs), which already rewards an unnormalized {@code exp}.
     *
     * @param program a kernel
     * @return the mean cosine over the test vectors
     */
    public static double softmaxShape(List<Instr> program) {
        double total = 0;
        for (double[] x : TESTS) {
            double m = java.util.Arrays.stream(x).max().orElse(0);
            double[] out = runSoftmax(program, x);
            double dot = 0;
            double nOut = 0;
            double nWant = 0;
            for (int i = 0; i < x.length; i++) {
                double want = Math.exp(x[i] - m);
                dot += out[i] * want;
                nOut += out[i] * out[i];
                nWant += want * want;
            }
            double cos = dot / Math.sqrt(nOut * nWant);
            total += Double.isFinite(cos) ? Math.max(0, cos) : 0;
        }
        return total / TESTS.size();
    }

    /**
     * Exact correctness counts most, then the shape of the output, then the cost.
     *
     * @param program a kernel
     * @return its fitness
     */
    public static double softmaxFitness(List<Instr> program) {
        return 10 * softmaxCorrectness(program) + 3 * softmaxShape(program) - softmaxCost(program) / 10;
    }

    static MapElites.Evaluation softmaxEvaluation(List<Instr> program) {
        return new MapElites.Evaluation(softmaxFitness(program),
                new double[] {program.size(), Math.min(softmaxCost(program), 24)});
    }

    static Instr randomInstr(SplittableRandom r) {
        Op op = Op.values()[r.nextInt(Op.values().length)];
        return new Instr(op, r.nextInt(REGS), r.nextInt(REGS), r.nextInt(REGS));
    }

    /**
     * Blind mutation: insert, delete or replace one instruction, or change one register.
     *
     * @param p the parent
     * @param r the random source
     * @return the child
     */
    public static List<Instr> mutateSoftmax(List<Instr> p, SplittableRandom r) {
        List<Instr> c = new ArrayList<>(p);
        int kind = r.nextInt(4);
        if (kind == 0 && c.size() < MAX_LENGTH || c.isEmpty()) {
            c.add(r.nextInt(c.size() + 1), randomInstr(r));
        } else if (kind == 1 && c.size() > 1) {
            c.remove(r.nextInt(c.size()));
        } else if (kind == 2) {
            c.set(r.nextInt(c.size()), randomInstr(r));
        } else {
            int i = r.nextInt(c.size());
            Instr in = c.get(i);
            int reg = r.nextInt(REGS);
            c.set(i, switch (r.nextInt(3)) {
                case 0 -> new Instr(in.op(), reg, in.a(), in.b());
                case 1 -> new Instr(in.op(), in.d(), reg, in.b());
                default -> new Instr(in.op(), in.d(), in.a(), reg);
            });
        }
        return c;
    }

    /**
     * An edit a code model would write: a new or replaced instruction reads only registers
     * that hold a value at that point (the input or something written earlier), and writes
     * the output register more often than not.
     */
    static List<Instr> editCoherently(List<Instr> p, SplittableRandom r) {
        List<Instr> c = new ArrayList<>(p);
        int kind = r.nextInt(3);
        if (kind == 1 && c.size() > 1) {
            c.remove(r.nextInt(c.size()));
            return c;
        }
        boolean insert = kind == 0 && c.size() < MAX_LENGTH || c.isEmpty();
        // code mostly grows at the end, and mostly reads what was written last
        int at = insert ? (r.nextBoolean() ? c.size() : r.nextInt(c.size() + 1)) : r.nextInt(c.size());
        List<Integer> vectors = new ArrayList<>(List.of(0));
        List<Integer> scalars = new ArrayList<>();
        for (int i = 0; i < at; i++) {
            Instr in = c.get(i);
            boolean scalarOut = in.op() == Op.MAX || in.op() == Op.SUM || in.op() == Op.RECIP;
            (scalarOut ? scalars : vectors).add(in.d());
        }
        Op op = Op.values()[r.nextInt(Op.values().length)];
        boolean readsScalar = op == Op.RECIP;
        if ((readsScalar || op == Op.SUBS || op == Op.MULS || op == Op.DIVS) && scalars.isEmpty()) {
            op = r.nextBoolean() ? Op.MAX : Op.SUM;   // nothing to read yet: make a scalar first
            readsScalar = false;
        }
        int a = readsScalar ? recent(scalars, r) : recent(vectors, r);
        int b = scalars.isEmpty() ? 0 : recent(scalars, r);
        int d = op == Op.MAX || op == Op.SUM || op == Op.RECIP ? r.nextInt(REGS) : r.nextDouble() < 0.7 ? 1
                : r.nextInt(REGS);
        Instr in = new Instr(op, d, a, b);
        if (insert) {
            c.add(at, in);
        } else {
            c.set(at, in);
        }
        return c;
    }

    private static int recent(List<Integer> written, SplittableRandom r) {
        return r.nextDouble() < 0.6 ? written.getLast() : written.get(r.nextInt(written.size()));
    }

    /**
     * The stand-in for the code model: with the parent and inspirations in its prompt, it
     * either splices a run of instructions from an inspiration into the parent (reusing code
     * that works elsewhere) or makes a small coherent edit ({@link #editCoherently}).
     *
     * @param parent the parent
     * @param inspirations good kernels from the archive
     * @param r the random source
     * @return the child
     */
    static List<Instr> proposeSoftmax(ProgramEvolution.Scored<List<Instr>> parent,
            List<ProgramEvolution.Scored<List<Instr>>> inspirations, SplittableRandom r) {
        List<Instr> p = parent.program();
        if (inspirations.isEmpty() || r.nextDouble() < 0.5) {
            return editCoherently(p, r);
        }
        List<Instr> donor = inspirations.get(r.nextInt(inspirations.size())).program();
        int from = r.nextInt(donor.size());
        int to = from + 1 + r.nextInt(donor.size() - from);
        List<Instr> c = new ArrayList<>();
        int cut = r.nextInt(p.size() + 1);
        int resume = Math.min(p.size(), cut + r.nextInt(to - from + 1));   // replace part of the parent
        c.addAll(p.subList(0, cut));
        c.addAll(donor.subList(from, to));
        c.addAll(p.subList(resume, p.size()));
        while (c.size() > MAX_LENGTH) {
            c.remove(c.size() - 1);
        }
        return c;
    }

    /** @return the naive kernel {@code exp, sum, divide} */
    public static List<Instr> naiveSoftmax() {
        return List.of(new Instr(Op.EXP, 1, 0, 0), new Instr(Op.SUM, 0, 1, 0), new Instr(Op.DIVS, 1, 1, 0));
    }

    // ---------------------------------------------------------------- matmul

    /**
     * A tiled matmul loop nest.
     *
     * @param order the loop order, a permutation of {@code "ijk"}
     * @param ti the tile size along i
     * @param tj the tile size along j
     * @param tk the tile size along k
     */
    public record Matmul(String order, int ti, int tj, int tk) {
        @Override
        public String toString() {
            return "for " + order + " tiles " + ti + "x" + tj + "x" + tk;
        }
    }

    static final String[] ORDERS = {"ijk", "ikj", "jik", "jki", "kij", "kji"};
    static final int[] TILES = {1, 2, 4, 8, 16, 32};
    static final int N = 32;

    /** A fully associative LRU cache of 32 lines of 8 floats (1 KiB). */
    private static final class Cache {
        private final long[] tags = new long[32];
        private final long[] used = new long[32];
        private long clock;
        private long misses;

        Cache() {
            java.util.Arrays.fill(tags, -1);
        }

        void touch(long address) {
            long line = address / 8;
            clock++;
            int lru = 0;
            for (int i = 0; i < tags.length; i++) {
                if (tags[i] == line) {
                    used[i] = clock;
                    return;
                }
                if (used[i] < used[lru]) {
                    lru = i;
                }
            }
            misses++;
            tags[lru] = line;
            used[lru] = clock;
        }
    }

    /**
     * Runs the nest on fixed matrices through the simulated cache.
     *
     * @param m the kernel
     * @return the cache misses, or -1 if the product came out wrong
     */
    public static long matmulMisses(Matmul m) {
        float[] a = new float[N * N];
        float[] b = new float[N * N];
        for (int i = 0; i < a.length; i++) {
            a[i] = (i * 7 % 13) - 6;
            b[i] = (i * 5 % 11) - 5;
        }
        float[] c = new float[N * N];
        Cache cache = new Cache();
        int[] tile = new int[3];
        for (int axis = 0; axis < 3; axis++) {
            char ch = m.order().charAt(axis);
            tile[axis] = ch == 'i' ? m.ti() : ch == 'j' ? m.tj() : m.tk();
        }
        int[] idx = new int[3];   // i, j, k
        for (int o0 = 0; o0 < N; o0 += tile[0]) {
            for (int o1 = 0; o1 < N; o1 += tile[1]) {
                for (int o2 = 0; o2 < N; o2 += tile[2]) {
                    for (int x0 = o0; x0 < Math.min(N, o0 + tile[0]); x0++) {
                        for (int x1 = o1; x1 < Math.min(N, o1 + tile[1]); x1++) {
                            for (int x2 = o2; x2 < Math.min(N, o2 + tile[2]); x2++) {
                                idx["ijk".indexOf(m.order().charAt(0))] = x0;
                                idx["ijk".indexOf(m.order().charAt(1))] = x1;
                                idx["ijk".indexOf(m.order().charAt(2))] = x2;
                                int i = idx[0];
                                int j = idx[1];
                                int k = idx[2];
                                cache.touch(i * N + k);
                                cache.touch(N * N + k * N + j);
                                cache.touch(2L * N * N + i * N + j);
                                c[i * N + j] += a[i * N + k] * b[k * N + j];
                            }
                        }
                    }
                }
            }
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                float want = 0;
                for (int k = 0; k < N; k++) {
                    want += a[i * N + k] * b[k * N + j];
                }
                if (want != c[i * N + j]) {
                    return -1;
                }
            }
        }
        return cache.misses;
    }

    static double matmulFitness(Matmul m) {
        long misses = matmulMisses(m);
        return misses < 0 ? -1e9 : -misses / 1000.0;
    }

    static int tileIndex(int t) {
        return Integer.numberOfTrailingZeros(t);
    }

    static Matmul mutateMatmul(Matmul m, SplittableRandom r) {
        return switch (r.nextInt(4)) {
            case 0 -> new Matmul(ORDERS[r.nextInt(ORDERS.length)], m.ti(), m.tj(), m.tk());
            case 1 -> new Matmul(m.order(), TILES[r.nextInt(TILES.length)], m.tj(), m.tk());
            case 2 -> new Matmul(m.order(), m.ti(), TILES[r.nextInt(TILES.length)], m.tk());
            default -> new Matmul(m.order(), m.ti(), m.tj(), TILES[r.nextInt(TILES.length)]);
        };
    }

    static Matmul proposeMatmul(ProgramEvolution.Scored<Matmul> parent, List<ProgramEvolution.Scored<Matmul>> insp,
            SplittableRandom r) {
        Matmul p = parent.program();
        if (insp.isEmpty() || r.nextDouble() < 0.5) {
            return mutateMatmul(p, r);
        }
        Matmul d = insp.get(r.nextInt(insp.size())).program();   // borrow one decision from a good kernel
        return switch (r.nextInt(4)) {
            case 0 -> new Matmul(d.order(), p.ti(), p.tj(), p.tk());
            case 1 -> new Matmul(p.order(), d.ti(), p.tj(), p.tk());
            case 2 -> new Matmul(p.order(), p.ti(), d.tj(), p.tk());
            default -> new Matmul(p.order(), p.ti(), p.tj(), d.tk());
        };
    }

    // ---------------------------------------------------------------- the comparison

    /**
     * Best results of each search at equal evaluations.
     *
     * @param softmaxGuided best softmax fitness, AlphaEvolve loop
     * @param softmaxRandom best softmax fitness, archive with blind mutation
     * @param softmaxRegularized best softmax fitness, regularized evolution
     * @param softmaxNaive the naive kernel's fitness
     * @param softmaxBest the best kernel found by the AlphaEvolve loop
     * @param softmaxGuidedHalf best fitness of the AlphaEvolve loop at half the evaluations
     * @param softmaxRandomHalf the same for blind mutation
     * @param matmulGuided fewest misses, AlphaEvolve loop
     * @param matmulRandom fewest misses, blind mutation
     * @param matmulRegularized fewest misses, regularized evolution
     * @param matmulNaive misses of the textbook ijk loop
     * @param matmulBest the best nest found by the AlphaEvolve loop
     */
    public record Result(double softmaxGuided, double softmaxRandom, double softmaxRegularized, double softmaxNaive,
            String softmaxBest, double softmaxGuidedHalf, double softmaxRandomHalf, long matmulGuided,
            long matmulRandom, long matmulRegularized, long matmulNaive, String matmulBest) {
    }

    /**
     * @param softmaxEvaluations evaluations per softmax search
     * @param matmulEvaluations evaluations per matmul search
     * @param seed the seed
     * @return the comparison
     */
    public static Result run(int softmaxEvaluations, int matmulEvaluations, long seed) {
        List<Instr> seedKernel = List.of(new Instr(Op.EXP, 1, 0, 0));
        int[] sBins = {MAX_LENGTH, 8};
        double[] sLow = {1, 0};
        double[] sHigh = {MAX_LENGTH, 24};
        ProgramEvolution<List<Instr>> guided = ProgramEvolution.of(seedKernel, EvolvedKernels::proposeSoftmax,
                EvolvedKernels::softmaxEvaluation, sBins, sLow, sHigh).seed(seed);
        double[] g = guided.run(softmaxEvaluations);
        ProgramEvolution<List<Instr>> blind = ProgramEvolution.<List<Instr>>of(seedKernel,
                (p, insp, r) -> mutateSoftmax(p.program(), r), EvolvedKernels::softmaxEvaluation, sBins, sLow, sHigh)
                .seed(seed);
        double[] b = blind.run(softmaxEvaluations);
        double[] reg = ProgramEvolution.regularizedEvolution(seedKernel, EvolvedKernels::mutateSoftmax,
                EvolvedKernels::softmaxFitness, 50, 10, softmaxEvaluations, new SplittableRandom(seed));

        Matmul naive = new Matmul("ijk", 32, 32, 32);
        int[] mBins = {6, 6};
        double[] mLow = {0, 0};
        double[] mHigh = {5, 15};
        java.util.function.Function<Matmul, MapElites.Evaluation> mEval = m -> new MapElites.Evaluation(
                matmulFitness(m), new double[] {java.util.Arrays.asList(ORDERS).indexOf(m.order()),
                    tileIndex(m.ti()) + tileIndex(m.tj()) + tileIndex(m.tk())});
        ProgramEvolution<Matmul> mGuided = ProgramEvolution.of(naive, EvolvedKernels::proposeMatmul, mEval, mBins,
                mLow, mHigh).seed(seed);
        double[] mg = mGuided.run(matmulEvaluations);
        ProgramEvolution<Matmul> mBlind = ProgramEvolution.<Matmul>of(naive, (p, insp, r) -> mutateMatmul(p.program(),
                r), mEval, mBins, mLow, mHigh).seed(seed);
        double[] mb = mBlind.run(matmulEvaluations);
        double[] mr = ProgramEvolution.regularizedEvolution(naive, EvolvedKernels::mutateMatmul,
                EvolvedKernels::matmulFitness, 20, 5, matmulEvaluations, new SplittableRandom(seed));
        return new Result(g[g.length - 1], b[b.length - 1], reg[reg.length - 1], softmaxFitness(naiveSoftmax()),
                String.join("; ", guided.best().program().stream().map(Instr::toString).toList()),
                g[g.length / 2], b[b.length / 2], Math.round(-mg[mg.length - 1] * 1000),
                Math.round(-mb[mb.length - 1] * 1000), Math.round(-mr[mr.length - 1] * 1000), matmulMisses(naive),
                mGuided.best().program().toString());
    }
}
