package io.github.zzuegg.brainiac.experiments.parallel;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SyntheticCorpus;
import io.github.zzuegg.brainiac.experimental.nn.ExpertChoiceMoe;
import io.github.zzuegg.brainiac.experimental.nn.FineGrainedMoe;
import io.github.zzuegg.brainiac.experimental.parallel.Collectives;
import io.github.zzuegg.brainiac.experimental.parallel.BuildingBlockSchedules;
import io.github.zzuegg.brainiac.experimental.parallel.ExpertParallelMoe;
import io.github.zzuegg.brainiac.experimental.parallel.MemoryAwareScheduler;
import io.github.zzuegg.brainiac.experimental.parallel.ParallelCostModel;
import io.github.zzuegg.brainiac.experimental.parallel.PipelineRunner;
import io.github.zzuegg.brainiac.experimental.parallel.PipelineSchedule;
import io.github.zzuegg.brainiac.experimental.parallel.ShardedDataParallel;
import io.github.zzuegg.brainiac.experimental.parallel.SplitBackward;
import io.github.zzuegg.brainiac.experimental.parallel.TensorParallelBlock;
import io.github.zzuegg.brainiac.experimental.train.LowBitOptimizerStates;
import io.github.zzuegg.brainiac.experiments.language.CountingTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.DataParallel;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.BiFunction;
import java.util.function.Supplier;

/**
 * Parallelism strategies on simulated devices (threads), each checked against single-device
 * gradients and measured by per-rank memory, bytes communicated and idle fraction:
 * <ul>
 *   <li>9.6.1 pipeline schedules (GPipe, 1F1B, interleaved 1F1B) and 9.6.2 zero-bubble ZB-H1,
 *       ZB-H2 and DualPipe, on a 16-block MLP over {@link CountingTasks} sequences, run for real
 *       by {@link PipelineRunner} and replayed by a {@link ParallelCostModel} calibrated with
 *       measured per-stage CPU times; baseline core {@link DataParallel} at equal devices;</li>
 *   <li>9.6.3 Megatron tensor parallelism and sequence parallelism on a transformer block;</li>
 *   <li>9.6.4 ZeRO stages 1 to 3 on a toy language model over {@link SyntheticCorpus} with AdamW,
 *       against {@link DataParallel}, offloaded states and 4-bit states;</li>
 *   <li>9.6.6 expert parallelism with capacity limits on a toy MoE language model, against a
 *       single-device {@link FineGrainedMoe} and {@link ExpertChoiceMoe}.</li>
 *   <li>10.6.15 ({@link #pipelineMemoryFollowUp}): memory-aware schedules: the
 *       {@link MemoryAwareScheduler} (greedy and branch and bound under an explicit activation
 *       limit, so ZB-H2's memory is actually used) and the V-shape building-block schedules
 *       V-Min, V-Half and V-ZB ({@link BuildingBlockSchedules}), against wave 9's 1F1B, ZB-H1
 *       and ZB-H2, as bubble against peak activation memory, optionally run on threads.</li>
 * </ul>
 */
public final class ParallelismShootout {

    private static final int VOCAB = 16;
    private static final int MICRO_ROWS = 8;
    private static final BiFunction<Var, Tensor, Var> CE = Functional::crossEntropy;

    private ParallelismShootout() {
    }

    // ---------------------------------------------------------------- 9.6.1 / 9.6.2

    /**
     * A pipeline result.
     *
     * @param schedule the schedule
     * @param devices {@code p}
     * @param microBatches {@code m}
     * @param simulatedBubble the cost model's idle fraction
     * @param theoryBubble {@code (p-1)/(vm+p-1)} (for reference)
     * @param measuredBubble the idle fraction measured on threads
     * @param peakActivations the largest per-device peak, in micro-batch stage activations
     * @param messages point-to-point messages per step
     * @param gradientError largest gradient difference from one device ({@code NaN} if not run)
     * @param stepTime the simulated step time relative to one device doing all the work
     */
    public record PipelineRow(String schedule, int devices, int microBatches, double simulatedBubble,
            double theoryBubble, double measuredBubble, double peakActivations, long messages, double gradientError,
            double stepTime) {
    }

    private static Tensor[] counting(int rows, long seed) {
        int[][] seq = CountingTasks.sequences(new SplittableRandom(seed), rows, 3, VOCAB, 1, 2, 3);
        float[] x = new float[rows * 2 * VOCAB];
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            x[i * 2 * VOCAB + seq[i][0]] = 1;
            x[i * 2 * VOCAB + VOCAB + seq[i][1]] = 1;
            y[i] = seq[i][2];
        }
        return new Tensor[] {Tensor.of(Shape.of(rows, 2 * VOCAB), x), Tensor.of(Shape.of(rows), y)};
    }

    /** A stage: consecutive layers. */
    private static final class Stage implements Layer {
        private final List<Layer> layers;

        Stage(List<Layer> layers) {
            this.layers = layers;
        }

        @Override
        public Var forward(Var input) {
            Var h = input;
            for (Layer l : layers) {
                h = l.forward(h);
            }
            return h;
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>();
            layers.forEach(l -> p.addAll(l.parameters()));
            return p;
        }
    }

    /** Builds {@code stages} stages of a {@code blocks}-block MLP; each stage has its own tape. */
    private static List<Layer> mlpStages(int stages, int blocks, int width, long seed, List<SplitBackward> tapes) {
        List<Layer> out = new ArrayList<>();
        int per = blocks / stages;
        for (int s = 0; s < stages; s++) {
            SplitBackward tape = new SplitBackward();
            tapes.add(tape);
            SplittableRandom r = new SplittableRandom(seed * 7919 + s);
            List<Layer> layers = new ArrayList<>();
            if (s == 0) {
                layers.add(tape.linear(2 * VOCAB, width, Activation.RELU, r));
            }
            for (int b = 0; b < per; b++) {
                layers.add(tape.linear(width, width, Activation.RELU, r));
            }
            if (s == stages - 1) {
                layers.add(tape.linear(width, VOCAB, Activation.IDENTITY, r));
            }
            out.add(new Stage(layers));
        }
        return out;
    }

    /**
     * Calibrates the cost model on one stage of the MLP: median forward, B and W times of one
     * micro-batch, with a simulated link whose latency is 2% of the forward time (a fast
     * interconnect relative to compute) and 10 GB/s.
     *
     * @param devices {@code p}
     * @param blocks MLP blocks
     * @param width MLP width
     * @param seed the seed
     * @return the model
     */
    public static ParallelCostModel calibrate(int devices, int blocks, int width, long seed) {
        List<SplitBackward> tapes = new ArrayList<>();
        Layer stage = mlpStages(devices, blocks, width, seed, tapes).get(1 % devices);
        SplitBackward tape = tapes.get(1 % devices);
        Tensor x = Tensor.normal(Shape.of(MICRO_ROWS, width), 0, 1, new SplittableRandom(seed));
        Tensor seedGrad = Tensor.ones(Shape.of(MICRO_ROWS, width));
        if (devices == 1) {
            x = counting(MICRO_ROWS, seed)[0];
            seedGrad = Tensor.ones(Shape.of(MICRO_ROWS, VOCAB));
        }
        Tensor in = x;
        Tensor g = seedGrad;
        double f = ParallelCostModel.median(() -> stage.forward(Var.param(in)), 15);
        double fb = ParallelCostModel.median(() -> {
            Var v = Var.param(in);
            tape.backwardInput(0, stage.forward(v), g, v);
        }, 15);
        double fbw = ParallelCostModel.median(() -> {
            Var v = Var.param(in);
            tape.backwardInput(0, stage.forward(v), g, v);
            tape.backwardWeight(0);
        }, 15);
        tape.backwardWeight(0);
        double b = Math.max(fb - f, 0.2 * f);
        double w = Math.max(fbw - fb, 0.2 * f);
        // a link whose latency is 2% of a stage's forward pass, at 10 GB/s
        return new ParallelCostModel(f, b, w, 0.02 * f, 10e9, (long) MICRO_ROWS * width * Float.BYTES, true);
    }

    /**
     * Simulated bubble of every schedule over a grid of depths and micro-batch counts.
     *
     * @param cost the calibrated cost model
     * @param devices pipeline depths
     * @param microBatches micro-batch counts
     * @return rows (gradient error and measured bubble {@code NaN})
     */
    public static List<PipelineRow> simulatedGrid(ParallelCostModel cost, int[] devices, int[] microBatches) {
        List<PipelineRow> out = new ArrayList<>();
        for (int p : devices) {
            for (int m : microBatches) {
                for (PipelineSchedule s : schedules(p, m, cost)) {
                    out.add(row(s, cost, Double.NaN, Double.NaN));
                }
            }
        }
        return out;
    }

    private static List<PipelineSchedule> schedules(int p, int m, ParallelCostModel cost) {
        List<PipelineSchedule> out = new ArrayList<>();
        out.add(PipelineSchedule.gpipe(p, m));
        out.add(PipelineSchedule.oneFOneB(p, m));
        if (m % p == 0) {
            out.add(PipelineSchedule.interleaved(p, m, 2));
        }
        out.add(PipelineSchedule.zeroBubble(p, m, false, cost));
        out.add(PipelineSchedule.zeroBubble(p, m, true, cost));
        if (m % 2 == 0) {
            out.add(PipelineSchedule.dualPipe(p, m, cost));
        }
        return out;
    }

    private static PipelineRow row(PipelineSchedule s, ParallelCostModel cost, double measured, double error) {
        ParallelCostModel.Result r = cost.simulate(s);
        double single = s.microBatches() * s.devices() * (cost.forward() + cost.backwardInput() + cost.backwardWeight());
        return new PipelineRow(s.name(), s.devices(), s.microBatches(), r.bubble(),
                ParallelCostModel.theoreticalBubble(s.devices(), s.microBatches(), s.chunks()), measured,
                r.maxPeakActivations(), r.messages(), error, r.stepTime() / single);
    }

    /**
     * Runs every schedule for real on {@code p} threads and checks its gradients against one
     * device.
     *
     * @param devices {@code p}
     * @param microBatches {@code m}
     * @param blocks MLP blocks, divisible by {@code 2p}
     * @param width MLP width
     * @param seed the seed
     * @return one row per schedule, plus the core data-parallel baseline
     */
    public static List<PipelineRow> pipelines(int devices, int microBatches, int blocks, int width, long seed) {
        ParallelCostModel cost = calibrate(devices, blocks, width, seed);
        List<Tensor> xs = new ArrayList<>();
        List<Tensor> ys = new ArrayList<>();
        for (int mb = 0; mb < microBatches; mb++) {
            Tensor[] d = counting(MICRO_ROWS, seed * 1000 + mb);
            xs.add(d[0]);
            ys.add(d[1]);
        }
        List<PipelineRow> out = new ArrayList<>();
        for (PipelineSchedule s : schedules(devices, microBatches, cost)) {
            s.validate();
            List<SplitBackward> tapes = new ArrayList<>();
            List<Layer> stages = mlpStages(s.stages(), blocks, width, seed, tapes);
            List<SplitBackward> rTapes = new ArrayList<>();
            List<Layer> reverse = s.bidirectional() ? mlpStages(s.stages(), blocks, width, seed, rTapes) : null;
            PipelineRunner runner = new PipelineRunner(stages, tapes, reverse, s.bidirectional() ? rTapes : null, CE);
            PipelineRunner.Result got = runner.run(s, xs, ys);
            got = runner.run(s, xs, ys);   // second run: warm threads and code
            PipelineRunner.Result ref = PipelineRunner.referenceGradients(stages, xs, ys, CE);
            double err = PipelineRunner.maxDifference(got.gradients(), ref.gradients());
            out.add(row(s, cost, got.measuredBubble(), err));
        }
        out.add(dataParallelRow(devices, microBatches, blocks, width, seed, xs, ys, cost));
        return out;
    }

    /** Core DataParallel on the whole MLP: no bubble, an all-reduce of every gradient. */
    private static PipelineRow dataParallelRow(int devices, int microBatches, int blocks, int width, long seed,
            List<Tensor> xs, List<Tensor> ys, ParallelCostModel cost) {
        List<SplitBackward> tapes = new ArrayList<>();
        List<Layer> stages = mlpStages(1, blocks, width, seed, tapes);
        List<Var> params = stages.getFirst().parameters();
        List<Device> ds = new ArrayList<>();
        for (int i = 0; i < devices; i++) {
            ds.add(Device.cpu());
        }
        long bytes = 0;
        for (Var p : params) {
            bytes += p.value().size() * Float.BYTES;
        }
        try (DataParallel dp = new DataParallel(ds, params, (ps, x, y) -> {
            Var h = Var.of(x);
            for (int i = 0; i < ps.size(); i += 2) {
                h = h.matmul(ps.get(i)).add(ps.get(i + 1));
                if (i + 2 < ps.size()) {
                    h = h.relu();
                }
            }
            return Functional.crossEntropy(h, y);
        }, Optimizer.sgd(0.01f))) {
            dp.step(Tensor.concat(0, xs.toArray(Tensor[]::new)), Tensor.concat(0, ys.toArray(Tensor[]::new)));
        }
        double compute = microBatches * (cost.forward() + cost.backwardInput() + cost.backwardWeight());
        double comm = cost.collectiveTime(Collectives.allReduceBytes(bytes, devices), 2 * (devices - 1));
        double single = microBatches * devices * (cost.forward() + cost.backwardInput() + cost.backwardWeight());
        // every device holds every layer's activations for its m / p micro-batches
        return new PipelineRow("DataParallel (core)", devices, microBatches, 0, 0, Double.NaN,
                (double) devices * microBatches / devices, 2L * (devices - 1), Double.NaN, (compute + comm) / single);
    }

    // ---------------------------------------------------------------- 9.6.3

    /**
     * A tensor-parallel result.
     *
     * @param mode "tensor" or "tensor+sequence"
     * @param ranks {@code t}
     * @param bytesPerRank bytes each rank sent in one forward and backward pass
     * @param messages messages each rank sent in one step
     * @param parametersPerRank parameters held per rank
     * @param activationFloatsPerRank activation floats kept per rank
     * @param gradientError largest gradient difference from one device
     */
    public record TensorRow(String mode, int ranks, long bytesPerRank, long messages, long parametersPerRank,
            long activationFloatsPerRank, double gradientError) {
    }

    /**
     * Tensor and sequence parallelism of one transformer block (width 32, 8 heads, MLP 64,
     * sequence 16, batch 2) for each rank count.
     *
     * @param ranks rank counts, dividing 8
     * @param seed the seed
     * @return rows
     */
    public static List<TensorRow> tensorParallel(int[] ranks, long seed) {
        int width = 32;
        int heads = 8;
        int hidden = 64;
        int seq = 16;
        int batch = 2;
        int rows = seq * batch;
        TensorParallelBlock block = new TensorParallelBlock(width, heads, hidden, true, new SplittableRandom(seed));
        SplittableRandom r = new SplittableRandom(seed + 1);
        Tensor x = Tensor.normal(Shape.of(rows, width), 0, 1, r);
        Tensor target = Tensor.normal(Shape.of(rows, width), 0, 1, r);
        TensorParallelBlock.Shard ref = block.shard(0, new Collectives(1), false).batch(batch);
        ref.forward(Var.of(x)).mul(target).sum().backward();
        List<TensorRow> out = new ArrayList<>();
        for (boolean sp : new boolean[] {false, true}) {
            for (int t : ranks) {
                Collectives c = new Collectives(t);
                List<TensorParallelBlock.Shard> shards = new ArrayList<>();
                for (int i = 0; i < t; i++) {
                    shards.add(block.shard(i, c, sp).batch(batch));
                }
                List<Double> errs = Collectives.run(t, rank -> {
                    TensorParallelBlock.Shard s = shards.get(rank);
                    Tensor in = sp ? x.slice(0, rank * rows / t, (rank + 1) * rows / t) : x;
                    Tensor tg = sp ? target.slice(0, rank * rows / t, (rank + 1) * rows / t) : target;
                    s.forward(Var.of(in)).mul(tg).sum().backward();
                    s.syncNormGradients();
                    return TensorParallelBlock.maxGradientDifference(s, ref);
                });
                double err = errs.stream().mapToDouble(Double::doubleValue).max().orElse(0);
                out.add(new TensorRow(sp ? "tensor+sequence" : "tensor", t, c.totalBytes() / t,
                        c.totalMessages() / t, block.parametersPerRank(t),
                        block.activationFloatsPerRank(seq, batch, t, sp), err));
            }
        }
        return out;
    }

    // ---------------------------------------------------------------- 9.6.4

    /**
     * A ZeRO result.
     *
     * @param method the method
     * @param ranks {@code N}
     * @param parameterBytes per-rank parameter bytes
     * @param gradientBytes per-rank gradient bytes
     * @param stateBytes per-rank optimizer-state bytes (on the device)
     * @param bytesPerRankPerStep bytes each rank sends per step
     * @param maxLossDifference largest loss difference from core DataParallel over the steps
     * @param finalLoss the loss at the last step
     */
    public record ZeroRow(String method, int ranks, long parameterBytes, long gradientBytes, long stateBytes,
            long bytesPerRankPerStep, double maxLossDifference, float finalLoss) {

        /** @return per-rank bytes in total */
        public long total() {
            return parameterBytes + gradientBytes + stateBytes;
        }
    }

    private static final int LM_HIDDEN_LAYERS = 3;

    /**
     * The toy LM: the two previous tokens, one-hot, through {@code 1 + LM_HIDDEN_LAYERS} ReLU
     * layers to the next token (so no single layer dominates the parameter count, as in FSDP's
     * setting).
     */
    private static List<Var> lmParameters(int vocab, int hidden, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Var> p = new ArrayList<>();
        int in = 2 * vocab;
        for (int l = 0; l <= LM_HIDDEN_LAYERS; l++) {
            p.add(Var.param(Tensor.normal(Shape.of(in, hidden), 0, (float) (1 / Math.sqrt(in)), r)));
            p.add(Var.param(Tensor.zeros(Shape.of(hidden))));
            in = hidden;
        }
        p.add(Var.param(Tensor.normal(Shape.of(hidden, vocab), 0, (float) (1 / Math.sqrt(hidden)), r)));
        p.add(Var.param(Tensor.zeros(Shape.of(vocab))));
        return p;
    }

    private static Var lmLoss(List<Var> p, Tensor x, Tensor y) {
        Var h = Var.of(x);
        for (int i = 0; i < p.size(); i += 2) {
            h = h.matmul(p.get(i)).add(p.get(i + 1));
            if (i + 2 < p.size()) {
                h = h.relu();
            }
        }
        return Functional.crossEntropy(h, y);
    }

    private static Tensor[] corpusBigrams(SyntheticCorpus corpus, int rows, long seed) {
        int v = corpus.vocabulary();
        int[][] docs = corpus.documents();
        SplittableRandom r = new SplittableRandom(seed);
        float[] x = new float[rows * 2 * v];
        float[] y = new float[rows];
        for (int i = 0; i < rows; i++) {
            int[] d = docs[r.nextInt(docs.length)];
            int t = 2 + r.nextInt(d.length - 2);
            x[i * 2 * v + d[t - 2]] = 1;
            x[i * 2 * v + v + d[t - 1]] = 1;
            y[i] = d[t];
        }
        return new Tensor[] {Tensor.of(Shape.of(rows, 2 * v), x), Tensor.of(Shape.of(rows), y)};
    }

    /**
     * ZeRO stages 1 to 3 against core DataParallel on the toy LM with AdamW, plus the offloaded
     * and 4-bit-state variants.
     *
     * @param ranks {@code N}
     * @param steps training steps
     * @param seed the seed
     * @return rows
     */
    public static List<ZeroRow> zero(int ranks, int steps, long seed) {
        SyntheticCorpus corpus = SyntheticCorpus.generate(4, 6, 6, 0.5, seed);
        int v = corpus.vocabulary();
        int hidden = 128;
        List<Tensor[]> batches = new ArrayList<>();
        for (int s = 0; s < steps; s++) {
            batches.add(corpusBigrams(corpus, 64, seed * 100 + s));
        }
        Supplier<Optimizer> adamw = () -> Optimizer.adamW(0.01f, 0.01f);
        // the baseline: core DataParallel
        List<Var> base = lmParameters(v, hidden, seed);
        List<Device> ds = new ArrayList<>();
        for (int i = 0; i < ranks; i++) {
            ds.add(Device.cpu());
        }
        float[] baseLoss = new float[steps];
        try (DataParallel dp = new DataParallel(ds, base, ParallelismShootout::lmLoss, adamw.get())) {
            for (int s = 0; s < steps; s++) {
                baseLoss[s] = dp.step(batches.get(s)[0], batches.get(s)[1]);
            }
        }
        long psi = 0;
        long largest = 0;
        for (Var p : base) {
            psi += p.value().size();
            largest = Math.max(largest, p.value().size());
        }
        List<ZeroRow> out = new ArrayList<>();
        ShardedDataParallel.Memory dpMem = ShardedDataParallel.memory(psi, ranks, 0, 8, largest);
        out.add(new ZeroRow("DataParallel (core)", ranks, dpMem.parameters(), dpMem.gradients(),
                dpMem.optimizerStates(), Collectives.allReduceBytes(psi * 4, ranks), 0, baseLoss[steps - 1]));
        out.add(new ZeroRow("Offloaded (core, states on host)", ranks, dpMem.parameters(), dpMem.gradients(), 0,
                Collectives.allReduceBytes(psi * 4, ranks), 0, baseLoss[steps - 1]));
        for (int stage = 1; stage <= 3; stage++) {
            out.add(runZero("ZeRO-" + stage, stage, ranks, v, hidden, seed, batches, adamw, 8, baseLoss));
        }
        // ZeRO-1 with 4-bit states: measure the packed state size on one shard
        ZeroRow fourBit = runZero("ZeRO-1 + 4-bit states", 1, ranks, v, hidden, seed, batches,
                () -> LowBitOptimizerStates.fourBit(0.01f, 0.01f), 1, baseLoss);
        LowBitOptimizerStates probe = LowBitOptimizerStates.fourBit(0.01f, 0.01f);
        Var shard = Var.param(Tensor.zeros(Shape.of((int) ((psi + ranks - 1) / ranks))));
        shard.setGrad(Tensor.ones(shard.shape()));
        probe.step(List.of(shard));
        out.add(new ZeroRow(fourBit.method(), ranks, fourBit.parameterBytes(), fourBit.gradientBytes(),
                probe.stateBytes(), fourBit.bytesPerRankPerStep(), fourBit.maxLossDifference(), fourBit.finalLoss()));
        return out;
    }

    private static ZeroRow runZero(String name, int stage, int ranks, int v, int hidden, long seed,
            List<Tensor[]> batches, Supplier<Optimizer> opt, double stateBytes, float[] baseLoss) {
        List<Var> params = lmParameters(v, hidden, seed);
        ShardedDataParallel z = new ShardedDataParallel(ranks, stage, params, ParallelismShootout::lmLoss, opt,
                stateBytes);
        double worst = 0;
        float last = 0;
        for (int s = 0; s < batches.size(); s++) {
            last = z.step(batches.get(s)[0], batches.get(s)[1]);
            worst = Math.max(worst, Math.abs(last - baseLoss[s]));
        }
        ShardedDataParallel.Memory m = z.memoryPerRank();
        return new ZeroRow(name, ranks, m.parameters(), m.gradients(), m.optimizerStates(),
                z.collectives().totalBytes() / ranks / batches.size(), worst, last);
    }

    // ---------------------------------------------------------------- 9.6.6

    /**
     * An expert-parallel result.
     *
     * @param method the routing and balancing
     * @param capacityFactor the capacity factor ({@code NaN} without a capacity)
     * @param dropped the share of assignments dropped (mean over training)
     * @param allToAllBytes all-to-all bytes per step (needed)
     * @param paddedBytes all-to-all bytes per step with capacity-padded static buffers
     * @param slowestRank the slowest rank's work over the mean (mean over training)
     * @param validationLoss held-out next-token loss
     */
    public record MoeRow(String method, double capacityFactor, double dropped, long allToAllBytes, long paddedBytes,
            double slowestRank, double validationLoss) {
    }

    /** The toy MoE LM: embed the previous token, add an MoE layer, predict the next token. */
    private static final class MoeLm {
        final Var embed;
        final Var out;
        final Layer moe;

        MoeLm(int vocab, int width, Layer moe, long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            this.embed = Var.param(Tensor.normal(Shape.of(vocab, width), 0, 0.3f, r));
            this.out = Var.param(Tensor.normal(Shape.of(width, vocab), 0, 0.3f, r));
            this.moe = moe;
        }

        Var loss(Tensor prev, Tensor next) {
            Var h = embed.gather(0, prev);
            h = h.add(moe.forward(h));
            Var l = Functional.crossEntropy(h.matmul(out), next);
            Var aux = moe.auxiliaryLoss();
            return aux == null ? l : l.add(aux);
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(List.of(embed, out));
            p.addAll(moe.parameters());
            return p;
        }
    }

    private static Tensor[] corpusPairs(SyntheticCorpus corpus, int rows, SplittableRandom r) {
        int[][] docs = corpus.documents();
        float[] a = new float[rows];
        float[] b = new float[rows];
        for (int i = 0; i < rows; i++) {
            int[] d = docs[r.nextInt(docs.length)];
            int t = 1 + r.nextInt(d.length - 1);
            a[i] = d[t - 1];
            b[i] = d[t];
        }
        return new Tensor[] {Tensor.of(Shape.of(rows), a), Tensor.of(Shape.of(rows), b)};
    }

    /**
     * Expert parallelism over 4 ranks with 8 experts, top-2, at each capacity factor, with
     * auxiliary-loss or loss-free balancing, plus the single-device and expert-choice baselines.
     *
     * @param capacities capacity factors
     * @param steps training steps per run
     * @param seed the seed
     * @return rows
     */
    public static List<MoeRow> experts(float[] capacities, int steps, long seed) {
        SyntheticCorpus corpus = SyntheticCorpus.generate(4, 6, 6, 0.5, seed);
        int v = corpus.vocabulary();
        int width = 16;
        int e = 8;
        int k = 2;
        int ranks = 4;
        Tensor[] val = corpusPairs(corpus, 512, new SplittableRandom(seed + 99));
        List<MoeRow> out = new ArrayList<>();
        Map<String, Float> settings = new LinkedHashMap<>();
        for (float cf : capacities) {
            settings.put("aux loss|" + cf, cf);
            settings.put("loss-free bias|" + cf, cf);
        }
        settings.put("aux loss, no drop|0", 0f);
        for (Map.Entry<String, Float> en : settings.entrySet()) {
            String method = en.getKey().split("\\|")[0];
            float cf = en.getValue();
            boolean drop = cf > 0;
            FineGrainedMoe inner = FineGrainedMoe.of(width, e, 0, 16, k, new SplittableRandom(seed + 5));
            ExpertParallelMoe ep = new ExpertParallelMoe(inner, k, ranks, drop ? cf : 1, drop);
            boolean bias = method.startsWith("loss-free");
            if (!bias) {
                ep.balanceWeight(0.01f);
            }
            MoeLm lm = new MoeLm(v, width, ep, seed);
            double[] stats = train(lm, corpus, steps, seed, () -> {
                if (bias) {
                    ep.updateBalanceBias(0.002f);
                }
                return new double[] {ep.droppedFraction(), ep.allToAllBytes(), ep.paddedAllToAllBytes(),
                        ep.slowestRankFactor()};
            });
            double loss = Var.noGrad(() -> lm.loss(val[0], val[1])).value().item();
            out.add(new MoeRow("EP " + method, drop ? cf : Double.NaN, stats[0], (long) stats[1], (long) stats[2],
                    stats[3], loss));
        }
        // single-device FineGrainedMoe: no capacity, no communication
        FineGrainedMoe single = FineGrainedMoe.of(width, e, 0, 16, k, new SplittableRandom(seed + 5)).balanceWeight(0.01f);
        MoeLm lm = new MoeLm(v, width, single, seed);
        train(lm, corpus, steps, seed, () -> new double[] {0, 0, 0, 1});
        out.add(new MoeRow("single-device FineGrainedMoe", Double.NaN, 0, 0, 0, 1,
                Var.noGrad(() -> lm.loss(val[0], val[1])).value().item()));
        // expert choice: perfect balance by construction, some tokens unprocessed
        ExpertChoiceMoe ec = ExpertChoiceMoe.of(width, e, 16, ExpertChoiceMoe.Routing.EXPERT_CHOICE, k, 1f,
                new SplittableRandom(seed + 5));
        MoeLm lmEc = new MoeLm(v, width, ec, seed);
        double[] ecStats = train(lmEc, corpus, steps, seed, () -> {
            int[] per = ec.expertsPerToken();
            int none = 0;
            for (int c : per) {
                none += c == 0 ? 1 : 0;
            }
            int[] load = ec.expertLoad();
            double mean = java.util.Arrays.stream(load).average().orElse(1);
            double max = java.util.Arrays.stream(load).max().orElse(0);
            return new double[] {none / (double) Math.max(1, per.length), 0, 0, max / Math.max(mean, 1e-9)};
        });
        out.add(new MoeRow("ExpertChoiceMoe (tokens unprocessed)", 1.0, ecStats[0], 0, 0, ecStats[3],
                Var.noGrad(() -> lmEc.loss(val[0], val[1])).value().item()));
        return out;
    }

    private static double[] train(MoeLm lm, SyntheticCorpus corpus, int steps, long seed, Supplier<double[]> after) {
        Optimizer opt = Optimizer.adam(0.02f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        double[] sum = new double[4];
        List<Var> params = lm.parameters();
        for (int s = 0; s < steps; s++) {
            Tensor[] b = corpusPairs(corpus, 64, r);
            lm.loss(b[0], b[1]).backward();
            opt.step(params);
            double[] st = after.get();
            for (int i = 0; i < 4; i++) {
                sum[i] += st[i] / steps;
            }
        }
        return sum;
    }

    // ---------------------------------------------------------------- 10.6.15

    /**
     * Memory-aware pipeline schedules against wave 9's (10.6.15): for every depth and
     * micro-batch count, 1F1B, ZB-H1 and ZB-H2 from the wave 9 generators; the
     * {@link MemoryAwareScheduler} at ZB-H1's and ZB-H2's activation limits (greedy, then branch
     * and bound with {@code nodeBudget} nodes); the V-shape V-Min, V-Half and V-ZB; and the
     * greedy memory-aware scheduler on the V-shape placement at every limit of
     * {@code memoryFractions} (the memory sweep). Bubble and peak activations come from the cost
     * model; with {@code runThreads} every new schedule is also run by {@link PipelineRunner} on a
     * {@code 2 stages}-block MLP and checked against one device's gradients.
     *
     * @param cost the operation and link times
     * @param devices pipeline depths
     * @param microBatches micro-batch counts
     * @param memoryFractions activation limits for the sweep, as fractions of 1F1B's {@code p}
     * @param nodeBudget the branch-and-bound node budget (0 to skip the exact search)
     * @param runThreads whether to run the new schedules for real
     * @param seed the seed of the threaded runs
     * @return the rows ("exact" rows are marked "(optimal)" when the search finished)
     */
    public static List<PipelineRow> pipelineMemoryFollowUp(ParallelCostModel cost, int[] devices, int[] microBatches,
            double[] memoryFractions, long nodeBudget, boolean runThreads, long seed) {
        List<PipelineRow> out = new ArrayList<>();
        for (int p : devices) {
            for (int m : microBatches) {
                List<PipelineSchedule> list = new ArrayList<>();
                list.add(PipelineSchedule.oneFOneB(p, m));
                list.add(PipelineSchedule.zeroBubble(p, m, false, cost));
                list.add(PipelineSchedule.zeroBubble(p, m, true, cost));
                int baselines = list.size();
                list.add(MemoryAwareScheduler.greedy("memory-aware greedy @ZB-H1 memory", p, p, m, false, p, cost));
                list.add(MemoryAwareScheduler.greedy("memory-aware greedy @ZB-H2 memory", p, p, m, false, 2 * p - 1,
                        cost));
                if (nodeBudget > 0) {
                    MemoryAwareScheduler.Result exact = MemoryAwareScheduler.search("exact @ZB-H2 memory", p, p, m,
                            false, 2 * p - 1, cost, nodeBudget);
                    list.add(exact.optimal() ? PipelineSchedule.placed("exact @ZB-H2 memory (optimal)", p, p, m, false,
                            orders(exact.schedule())) : exact.schedule());
                }
                list.add(BuildingBlockSchedules.vMin(p, m, cost));
                list.add(BuildingBlockSchedules.vHalf(p, m, cost));
                list.add(BuildingBlockSchedules.vZb(p, m, cost));
                for (double f : memoryFractions) {
                    list.add(MemoryAwareScheduler.greedy(String.format("V-shape greedy @%.2f memory", f), p, 2 * p, m,
                            true, f * p, cost));
                }
                for (int i = 0; i < list.size(); i++) {
                    PipelineSchedule s = list.get(i);
                    s.validate();
                    double err = Double.NaN;
                    double measured = Double.NaN;
                    if (runThreads && i >= baselines) {
                        double[] r = runForReal(s, seed);
                        measured = r[0];
                        err = r[1];
                    }
                    out.add(row(s, cost, measured, err));
                }
            }
        }
        return out;
    }

    private static List<List<PipelineSchedule.Op>> orders(PipelineSchedule s) {
        List<List<PipelineSchedule.Op>> o = new ArrayList<>();
        for (int d = 0; d < s.devices(); d++) {
            o.add(s.ops(d));
        }
        return o;
    }

    /** Runs a split schedule on threads: {measured bubble, gradient error}. */
    private static double[] runForReal(PipelineSchedule s, long seed) {
        int blocks = 2 * s.stages();
        int width = 16;
        List<Tensor> xs = new ArrayList<>();
        List<Tensor> ys = new ArrayList<>();
        for (int mb = 0; mb < s.microBatches(); mb++) {
            Tensor[] d = counting(MICRO_ROWS, seed * 1000 + mb);
            xs.add(d[0]);
            ys.add(d[1]);
        }
        List<SplitBackward> tapes = new ArrayList<>();
        List<Layer> stages = mlpStages(s.stages(), blocks, width, seed, tapes);
        PipelineRunner runner = new PipelineRunner(stages, tapes, null, null, CE);
        PipelineRunner.Result got = runner.run(s, xs, ys);
        PipelineRunner.Result ref = PipelineRunner.referenceGradients(stages, xs, ys, CE);
        return new double[] {got.measuredBubble(), PipelineRunner.maxDifference(got.gradients(), ref.gradients())};
    }

    /**
     * The long 10.6.15 Run: memory-aware schedules on a calibrated cost model for
     * {@code p} in {4, 8} and {@code m} in {8, 16, 32}.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        ParallelCostModel cost = calibrate(4, 16, 32, 1);
        for (PipelineRow r : pipelineMemoryFollowUp(cost, new int[] {4, 8}, new int[] {8, 16, 32},
                new double[] {0.34, 0.5, 0.75, 1.0}, 2_000_000, true, 1)) {
            System.out.printf("%-34s p=%d m=%2d bubble %.3f measured %.3f peak %.2f err %.2e%n", r.schedule(),
                    r.devices(), r.microBatches(), r.simulatedBubble(), r.measuredBubble(), r.peakActivations(),
                    r.gradientError());
        }
    }
}
