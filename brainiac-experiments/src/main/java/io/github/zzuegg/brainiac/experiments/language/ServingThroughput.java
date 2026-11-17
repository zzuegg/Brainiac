package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.ChunkedPrefillScheduler;
import io.github.zzuegg.brainiac.experimental.llm.LogitMath;
import io.github.zzuegg.brainiac.experimental.llm.OffloadingKvCache;
import io.github.zzuegg.brainiac.experimental.llm.PagedKvCache;
import io.github.zzuegg.brainiac.experimental.llm.PagedLlama;
import io.github.zzuegg.brainiac.experimental.llm.PrefixCache;
import io.github.zzuegg.brainiac.llm.LanguageModel;
import io.github.zzuegg.brainiac.llm.Llama;
import io.github.zzuegg.brainiac.llm.LlamaConfig;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Serving a small {@link Llama} to many requests, the serving prototypes against the core
 * {@code Llama} + {@code Generator} path:
 * <ul>
 * <li>{@link #prefixReuse}: {@link PrefixCache} with a shared system prompt, time to first
 *     token against reading every prompt whole;</li>
 * <li>{@link #pagedMemory}: {@link PagedKvCache} peak memory and how many sequences fit a
 *     memory budget, against the core cache's contiguous buffers that grow by doubling;</li>
 * <li>{@link #continuousBatching}: the core {@code Generator.batcher} (continuous batching,
 *     over the core {@code Llama} and over {@link PagedLlama}) against static batching under
 *     a Poisson arrival trace, tokens per model pass and p95 latency in passes;</li>
 * <li>{@link #chunkedPrefill}: {@link ChunkedPrefillScheduler} with long prompts arriving
 *     among running decodes, decode-latency jitter against monolithic prefill;</li>
 * <li>{@link #offload}: {@link OffloadingKvCache}, bytes kept on the device and output error
 *     against device-only full KV, the core {@code Llama.slidingWindow} and the core
 *     StreamingLLM state at the same device budget.</li>
 * </ul>
 * Wall-clock numbers are measured on a shared CPU and vary from run to run; the token counts
 * are exact.
 */
public final class ServingThroughput {

    private ServingThroughput() {
    }

    /**
     * A random float32 model on the CPU.
     *
     * @param vocab the vocabulary
     * @param hidden the hidden size
     * @param seed the seed
     * @return the model
     */
    public static Llama model(int vocab, int hidden, long seed) {
        return Llama.random(LlamaConfig.small(vocab, hidden, 2, 4, 2), seed, Device.cpu(), DType.FLOAT32);
    }

    static int[] randomTokens(SplittableRandom r, int n, int vocab) {
        int[] t = new int[n];
        for (int i = 0; i < n; i++) {
            t[i] = r.nextInt(vocab);
        }
        return t;
    }

    // ---------------------------------------------------------------- 4.1 prefix caching

    /**
     * Prefix caching results.
     *
     * @param ttftMillis mean time to first token per method
     * @param promptTokensRead the prompt tokens each method read
     * @param maxLogitDiff the largest logit difference of any cached method from no reuse
     */
    public record PrefixResult(Map<String, Double> ttftMillis, Map<String, Long> promptTokensRead,
            double maxLogitDiff) {
    }

    /**
     * Requests that share a system prompt, served with no reuse (the core path), with a
     * {@link PrefixCache} over the core {@code Llama}, and with one over {@link PagedLlama}.
     *
     * @param system the shared system prompt length
     * @param user each request's own tokens
     * @param requests the requests
     * @param seed the seed
     * @return the measurements
     */
    public static PrefixResult prefixReuse(int system, int user, int requests, long seed) {
        Llama core = model(64, 64, seed);
        SplittableRandom r = new SplittableRandom(seed);
        int[] shared = randomTokens(r, system, 64);
        int[][] prompts = new int[requests][];
        for (int i = 0; i < requests; i++) {
            prompts[i] = Arrays.copyOf(shared, system + user);
            System.arraycopy(randomTokens(r, user, 64), 0, prompts[i], system, user);
        }
        core.next(core.newState(1), new int[][] {prompts[0]}, null);   // warm up
        Map<String, Double> ttft = new java.util.LinkedHashMap<>();
        Map<String, Long> read = new java.util.LinkedHashMap<>();
        float[][] reference = new float[requests][];
        long start = System.nanoTime();
        for (int i = 0; i < requests; i++) {
            reference[i] = core.next(core.newState(1), new int[][] {prompts[i]}, null).toArray();
        }
        ttft.put("no reuse", (System.nanoTime() - start) / 1e6 / requests);
        read.put("no reuse", (long) requests * (system + user));
        double diff = 0;
        PrefixCache coreCache = PrefixCache.forCore(core, 16);
        PagedLlama paged = PagedLlama.of(core, 16, 256);
        PrefixCache pagedCache = PrefixCache.forPaged(paged, 16);
        for (Map.Entry<String, PrefixCache> e : List.of(Map.entry("core + prefix cache", coreCache),
                Map.entry("paged + prefix cache", pagedCache))) {
            PrefixCache cache = e.getValue();
            cache.prefill(shared);   // the system prompt is cached once, before the requests
            long t0 = System.nanoTime();
            for (int i = 0; i < requests; i++) {
                PrefixCache.Prefill p = cache.prefill(prompts[i]);
                float[] got = p.logits().toArray();
                for (int k = 0; k < got.length; k++) {
                    diff = Math.max(diff, Math.abs(got[k] - reference[i][k]));
                }
                if (cache == pagedCache) {
                    paged.release(p.state());
                }
            }
            ttft.put(e.getKey(), (System.nanoTime() - t0) / 1e6 / requests);
            read.put(e.getKey(), cache.computedTokens() - system);
        }
        return new PrefixResult(ttft, read, diff);
    }

    // ---------------------------------------------------------------- 4.2 paged KV

    /**
     * Paged against contiguous KV memory.
     *
     * @param pagedPeakBytes the paged pool's peak while serving the requests (measured)
     * @param contiguousBytes the core cache's buffers for the same requests, one state each
     * @param batchedBytes the core cache for the same requests in one padded batch state
     * @param pagedFit sequences a fixed budget admits, paged
     * @param contiguousFit sequences the same budget admits with the core rule
     * @param identical whether paged serving gave the core model's greedy tokens
     */
    public record MemoryResult(long pagedPeakBytes, long contiguousBytes, long batchedBytes, int pagedFit,
            int contiguousFit, boolean identical) {
    }

    /**
     * Slots the core {@code Llama} cache holds after reading a prompt and then decoding one
     * token at a time: it grows to {@code max(need, max(64, 2 * capacity))} when full.
     *
     * @param prompt the prompt length
     * @param generated tokens decoded after it
     * @return the slots allocated at the end
     */
    public static int coreCapacity(int prompt, int generated) {
        int capacity = 0;
        int length = 0;
        int[] reads = new int[generated + 1];
        reads[0] = prompt;
        Arrays.fill(reads, 1, reads.length, 1);
        for (int steps : reads) {
            int need = length + steps;
            if (need > capacity) {
                capacity = Math.max(need, Math.max(64, 2 * capacity));
            }
            length = need;
        }
        return capacity;
    }

    /**
     * Serves requests of mixed lengths through a {@link PagedKvCache}, measures its peak, and
     * compares it with the core cache's allocation rule; then counts how many such requests
     * fit a budget of {@code budgetBlocks} blocks each way.
     *
     * @param requests the requests served
     * @param blockSize the slots per block
     * @param budgetBlocks the budget for the fit count, in blocks
     * @param seed the seed
     * @return the measurements
     */
    public static MemoryResult pagedMemory(int requests, int blockSize, int budgetBlocks, long seed) {
        Llama core = model(64, 32, seed);
        SplittableRandom r = new SplittableRandom(seed);
        PagedLlama paged = PagedLlama.of(core, blockSize, 1024);
        List<ChunkedPrefillScheduler.Request> list = new ArrayList<>();
        int[][] lengths = new int[requests][2];
        for (int i = 0; i < requests; i++) {
            int p = 8 + r.nextInt(72);
            int g = 4 + r.nextInt(36);
            lengths[i] = new int[] {p, g};
            list.add(new ChunkedPrefillScheduler.Request(i, randomTokens(r, p, 64), g, 0));
        }
        ChunkedPrefillScheduler.Report report = new ChunkedPrefillScheduler(paged, 1 << 20, 0).run(list);
        LlamaConfig c = core.config();
        long perSlot = 2L * c.layers() * c.kvHeads() * c.headDim() * Float.BYTES;
        long contiguous = 0;
        int longest = 0;
        for (int[] l : lengths) {
            contiguous += coreCapacity(l[0], l[1] - 1) * perSlot;
            longest = Math.max(longest, l[0] + l[1] - 1);
        }
        // one batch state: every row padded to the longest prompt, grown as one buffer
        int maxPrompt = Arrays.stream(lengths).mapToInt(l -> l[0]).max().orElse(0);
        int maxGen = Arrays.stream(lengths).mapToInt(l -> l[1]).max().orElse(0);
        long batched = (long) requests * coreCapacity(maxPrompt, maxGen - 1) * perSlot;
        // the greedy tokens of a few requests must match the core path
        boolean same = true;
        for (int i = 0; i < 3; i++) {
            int[] want = new io.github.zzuegg.brainiac.llm.Generator(core, null).generate(
                    new int[][] {list.get(i).prompt()}, io.github.zzuegg.brainiac.llm.Generator.Options.greedy(
                    lengths[i][1]), null).getFirst().tokens();
            same &= Arrays.equals(want, report.completions().get(i).tokens());
        }
        long budgetSlots = (long) budgetBlocks * blockSize;
        int pagedFit = 0;
        int contiguousFit = 0;
        long usedBlocks = 0;
        long usedSlots = 0;
        SplittableRandom fit = new SplittableRandom(seed + 1);
        boolean pagedOpen = true;
        boolean coreOpen = true;
        while (pagedOpen || coreOpen) {
            int p = 8 + fit.nextInt(72);
            int g = 4 + fit.nextInt(36);
            int total = p + g - 1;
            if (pagedOpen && usedBlocks + (total + blockSize - 1) / blockSize <= budgetBlocks) {
                usedBlocks += (total + blockSize - 1) / blockSize;
                pagedFit++;
            } else {
                pagedOpen = false;
            }
            if (coreOpen && usedSlots + coreCapacity(p, g - 1) <= budgetSlots) {
                usedSlots += coreCapacity(p, g - 1);
                contiguousFit++;
            } else {
                coreOpen = false;
            }
        }
        PagedKvCache pool = (PagedKvCache) paged.store();
        return new MemoryResult(pool.peakResidentBytes(), contiguous, batched, pagedFit, contiguousFit, same);
    }

    // ---------------------------------------------------------------- 4.3 continuous batching

    /**
     * One serving policy under the arrival trace, on a clock that counts model passes (calls of
     * {@code LanguageModel.next}), so the comparison does not depend on how busy the CPU is.
     *
     * @param tokensPerPass requested tokens delivered per model pass, first arrival to last answer
     * @param p50Passes the median passes from arrival to complete answer
     * @param p95Passes the 95th percentile of that
     * @param tokensPerSecond requested tokens per second of wall time (noisy on a shared CPU)
     */
    public record Latency(double tokensPerPass, double p50Passes, double p95Passes, double tokensPerSecond) {
    }

    /** Counts the passes of a model, the clock of {@link #continuousBatching}. */
    private static final class CountingModel implements LanguageModel {
        final LanguageModel model;
        final java.util.concurrent.atomic.AtomicLong passes = new java.util.concurrent.atomic.AtomicLong();

        CountingModel(LanguageModel model) {
            this.model = model;
        }

        @Override
        public int vocabSize() {
            return model.vocabSize();
        }

        @Override
        public State newState(int batch) {
            return model.newState(batch);
        }

        @Override
        public io.github.zzuegg.brainiac.tensor.Tensor next(State state, int[][] tokens, boolean[][] padding) {
            io.github.zzuegg.brainiac.tensor.Tensor out = model.next(state, tokens, padding);
            passes.incrementAndGet();
            return out;
        }
    }

    /**
     * Requests with Poisson arrivals (in model passes) and mixed answer lengths, served by
     * static batching (batches of {@code maxBatch} in arrival order, started once full, each
     * decoded by the core {@code Generator} to its longest answer) and by the core continuous
     * {@code Generator.batcher}, over the core {@code Llama} and over {@link PagedLlama}.
     *
     * @param requests the requests
     * @param maxBatch the most sequences decoded together
     * @param meanGap the mean passes between arrivals
     * @param seed the seed
     * @return policy name to measurements
     */
    public static Map<String, Latency> continuousBatching(int requests, int maxBatch, double meanGap, long seed) {
        Llama core = model(64, 64, seed);
        SplittableRandom r = new SplittableRandom(seed);
        int[][] prompts = new int[requests][];
        int[] lengths = new int[requests];
        double[] arrival = new double[requests];
        double t = 0;
        long useful = 0;
        for (int i = 0; i < requests; i++) {
            prompts[i] = randomTokens(r, 8 + r.nextInt(16), 64);
            lengths[i] = 4 + r.nextInt(60);
            useful += lengths[i];
            t += -Math.log(1 - r.nextDouble()) * meanGap;
            arrival[i] = Math.floor(t);
        }
        Map<String, Latency> out = new LinkedHashMap<>();
        out.put("static batching", staticBatching(new CountingModel(core), prompts, lengths, arrival, maxBatch,
                useful));
        out.put("continuous (core Llama)", continuous(new CountingModel(core), prompts, lengths, arrival, maxBatch,
                useful));
        out.put("continuous (PagedLlama)", continuous(new CountingModel(PagedLlama.of(core, 16, 8192)), prompts,
                lengths, arrival, maxBatch, useful));
        return out;
    }

    private static io.github.zzuegg.brainiac.llm.Generator.Options greedy(int n) {
        return io.github.zzuegg.brainiac.llm.Generator.Options.greedy(n).withStopTokens(java.util.Set.of());
    }

    private static Latency staticBatching(CountingModel model, int[][] prompts, int[] lengths, double[] arrival,
            int maxBatch, long useful) {
        io.github.zzuegg.brainiac.llm.Generator gen = new io.github.zzuegg.brainiac.llm.Generator(model, null);
        int n = prompts.length;
        double[] latency = new double[n];
        double clock = 0;
        long start = System.nanoTime();
        for (int i = 0; i < n; i += maxBatch) {
            int end = Math.min(n, i + maxBatch);
            clock = Math.max(clock, arrival[end - 1]);   // a static batch starts once it is full
            int longest = 0;
            for (int j = i; j < end; j++) {
                longest = Math.max(longest, lengths[j]);
            }
            long before = model.passes.get();
            gen.generate(Arrays.copyOfRange(prompts, i, end), greedy(longest), null);
            clock += model.passes.get() - before;
            for (int j = i; j < end; j++) {
                latency[j] = clock - arrival[j];
            }
        }
        return latency(latency, useful, clock - arrival[0], (System.nanoTime() - start) / 1e6);
    }

    private static Latency continuous(CountingModel model, int[][] prompts, int[] lengths, double[] arrival,
            int maxBatch, long useful) {
        int n = prompts.length;
        double[] latency = new double[n];
        double[] finish = new double[n];
        io.github.zzuegg.brainiac.llm.Generator gen = new io.github.zzuegg.brainiac.llm.Generator(model, null);
        java.util.concurrent.atomic.AtomicInteger open = new java.util.concurrent.atomic.AtomicInteger();
        long start = System.nanoTime();
        try (io.github.zzuegg.brainiac.llm.Generator.Batcher batcher = gen.batcher(maxBatch)) {
            List<java.util.concurrent.CompletableFuture<?>> futures = new ArrayList<>();
            double offset = 0;   // idle time skipped while nothing runs
            for (int i = 0; i < n; i++) {
                while (model.passes.get() + offset < arrival[i]) {
                    if (open.get() == 0) {
                        offset = arrival[i] - model.passes.get();
                    } else {
                        Thread.onSpinWait();
                    }
                }
                final int request = i;
                final double skipped = offset;
                open.incrementAndGet();
                futures.add(batcher.submit(prompts[i], greedy(lengths[i]), null).thenRun(() -> {
                    finish[request] = model.passes.get() + skipped;
                    latency[request] = finish[request] - arrival[request];
                    open.decrementAndGet();
                }));
            }
            java.util.concurrent.CompletableFuture.allOf(futures.toArray(java.util.concurrent.CompletableFuture[]::new))
                    .join();
        }
        double last = 0;
        for (double f : finish) {
            last = Math.max(last, f);
        }
        return latency(latency, useful, last - arrival[0], (System.nanoTime() - start) / 1e6);
    }

    private static Latency latency(double[] latency, long useful, double passes, double millis) {
        double[] sorted = latency.clone();
        Arrays.sort(sorted);
        return new Latency(useful / passes, ChunkedPrefillScheduler.Report.quantile(sorted, 0.5),
                ChunkedPrefillScheduler.Report.quantile(sorted, 0.95), useful / (millis / 1000));
    }

    // ---------------------------------------------------------------- 4.4 chunked prefill

    /**
     * Chunked against monolithic prefill.
     *
     * @param monolithic the run that reads prompts whole
     * @param chunked the run with chunked prefill
     */
    public record ChunkResult(ChunkedPrefillScheduler.Report monolithic, ChunkedPrefillScheduler.Report chunked) {
    }

    /**
     * Short requests decode while long prompts arrive; both schedulers serve the same trace.
     *
     * @param longPrompt the long prompts' length
     * @param chunk the chunk size
     * @param seed the seed
     * @return both reports
     */
    public static ChunkResult chunkedPrefill(int longPrompt, int chunk, long seed) {
        Llama core = model(64, 64, seed);
        SplittableRandom r = new SplittableRandom(seed);
        List<ChunkedPrefillScheduler.Request> trace = new ArrayList<>();
        int id = 0;
        for (int i = 0; i < 6; i++) {
            trace.add(new ChunkedPrefillScheduler.Request(id++, randomTokens(r, 8, 64), 40, i));
        }
        for (int i = 0; i < 3; i++) {
            trace.add(new ChunkedPrefillScheduler.Request(id++, randomTokens(r, longPrompt, 64), 4, 8 + 10 * i));
        }
        int budget = chunk + 8;
        new ChunkedPrefillScheduler(PagedLlama.of(core, 16, 512), budget, chunk).run(trace);   // warm up
        ChunkedPrefillScheduler.Report mono = new ChunkedPrefillScheduler(PagedLlama.of(core, 16, 512), budget, 0)
                .run(trace);
        ChunkedPrefillScheduler.Report chunked = new ChunkedPrefillScheduler(PagedLlama.of(core, 16, 512), budget,
                chunk).run(trace);
        return new ChunkResult(mono, chunked);
    }

    // ---------------------------------------------------------------- 4.13 KV offload

    /**
     * KV offload against device-only alternatives at the same device budget.
     *
     * @param meanAbsError per method, the mean absolute logit error against full device KV
     * @param top1Agreement per method, how often its top token equals full KV's
     * @param hotBytes the offloading cache's peak device bytes while decoding
     * @param fullBytes what device-only full KV holds at the end
     * @param transferPerToken bytes fetched from the host per decoded token
     * @param tokensPerSecond decode speed: offloaded and full paged
     * @param windowTokens the tokens the windowed baselines keep: the offloading cache's device bytes
     */
    public record OffloadResult(Map<String, Double> meanAbsError, Map<String, Double> top1Agreement, long hotBytes,
            long fullBytes, double transferPerToken, double[] tokensPerSecond, int windowTokens) {
    }

    /**
     * A model whose attention is sharpened (query and key weights scaled up) so that decode
     * steps lean on a few far-back tokens, reads a long prompt and then decodes a fixed
     * continuation. Full KV is the reference; offloading keeps {@code hot} blocks of
     * {@code blockSize} per layer plus a 4-bit key sketch on the device; the core sliding window
     * and StreamingLLM state keep as many tokens as fit in the same device bytes.
     *
     * @param prompt the prompt length
     * @param decode the tokens decoded
     * @param blockSize slots per block
     * @param hot blocks per layer kept on the device
     * @param seed the seed
     * @return the measurements
     */
    public static OffloadResult offload(int prompt, int decode, int blockSize, int hot, long seed) {
        Llama core = model(64, 64, seed);
        for (Map.Entry<String, Var> e : core.namedParameters().entrySet()) {
            if (e.getKey().endsWith("q_proj.weight") || e.getKey().endsWith("k_proj.weight")) {
                e.getValue().assign(e.getValue().value().mul(20f));
            }
        }
        SplittableRandom r = new SplittableRandom(seed);
        int[] text = randomTokens(r, prompt + decode, 64);
        int[] head = Arrays.copyOf(text, prompt);
        List<float[]> exact = decodeAll(core, core.newState(1), head, text, prompt);
        Map<String, List<float[]>> runs = new java.util.LinkedHashMap<>();
        LlamaConfig c = core.config();
        OffloadingKvCache store = new OffloadingKvCache(c.layers(), c.kvHeads(), c.headDim(), blockSize, hot, 2,
                Device.cpu());
        PagedLlama offloaded = new PagedLlama(core, store);
        LanguageModel.State s = offloaded.newState(1);
        offloaded.next(s, new int[][] {head}, null);
        store.resetPeak();
        long moved = store.transferredBytes();
        long t0 = System.nanoTime();
        List<float[]> off = decodeAll(offloaded, s, null, text, prompt);
        double offSpeed = decode / ((System.nanoTime() - t0) / 1e9);
        long transfer = store.transferredBytes() - moved;
        runs.put("offload, key sketch", off);
        for (OffloadingKvCache.Selection rule : List.of(OffloadingKvCache.Selection.BOUNDS,
                OffloadingKvCache.Selection.ORACLE)) {
            PagedLlama other = new PagedLlama(core, new OffloadingKvCache(c.layers(), c.kvHeads(), c.headDim(),
                    blockSize, hot, 2, Device.cpu()).selection(rule));
            runs.put("offload, " + rule.name().toLowerCase(java.util.Locale.ROOT),
                    decodeAll(other, other.newState(1), head, text, prompt));
        }
        PagedLlama full = PagedLlama.of(core, blockSize, 256);
        LanguageModel.State fs = full.newState(1);
        full.next(fs, new int[][] {head}, null);
        long t1 = System.nanoTime();
        runs.put("paged full KV", decodeAll(full, fs, null, text, prompt));
        double fullSpeed = decode / ((System.nanoTime() - t1) / 1e9);
        long perToken = 2L * c.layers() * c.kvHeads() * c.headDim() * Float.BYTES;
        int budget = (int) (store.peakResidentBytes() / perToken);   // the same device bytes, in tokens
        core.slidingWindow(budget);
        runs.put("core slidingWindow", decodeAll(core, core.newState(1), head, text, prompt));
        core.slidingWindow(0);
        runs.put("core StreamingLLM state", decodeAll(core, core.newState(1, blockSize, (budget - blockSize) * 4 / 5), head,
                text, prompt));
        Map<String, Double> err = new java.util.LinkedHashMap<>();
        Map<String, Double> agree = new java.util.LinkedHashMap<>();
        for (Map.Entry<String, List<float[]>> e : runs.entrySet()) {
            double sum = 0;
            int same = 0;
            int n = 0;
            for (int i = 0; i < exact.size(); i++) {
                float[] a = exact.get(i);
                float[] b = e.getValue().get(i);
                for (int k = 0; k < a.length; k++) {
                    sum += Math.abs(a[k] - b[k]);
                    n++;
                }
                same += LogitMath.argmax(a) == LogitMath.argmax(b) ? 1 : 0;
            }
            err.put(e.getKey(), sum / n);
            agree.put(e.getKey(), same / (double) exact.size());
        }
        return new OffloadResult(err, agree, store.peakResidentBytes(), store.totalBytes(), transfer / (double) decode,
                new double[] {offSpeed, fullSpeed}, budget);
    }

    /** Reads the prompt (when given), then feeds the text's continuation one token at a time. */
    private static List<float[]> decodeAll(LanguageModel model, LanguageModel.State state, int[] head, int[] text,
            int prompt) {
        if (head != null) {
            model.next(state, new int[][] {head}, null);
        }
        List<float[]> out = new ArrayList<>();
        for (int t = prompt; t < text.length; t++) {
            out.add(model.next(state, new int[][] {{text[t]}}, null).toArray());
        }
        return out;
    }
}
