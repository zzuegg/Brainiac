package io.github.zzuegg.brainiac.experimental.llm;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Deque;
import java.util.List;
import java.util.Objects;

/**
 * Chunked prefill (Sarathi-Serve): an iteration-level scheduler that splits long prompts into
 * chunks and puts each chunk in the same model pass as the decode tokens of the sequences
 * already running, under a fixed token budget per pass. A long prompt arriving then delays
 * running sequences by at most one chunk per step instead of its whole length, so decode
 * latency stays smooth; the pass stays full, so throughput holds.
 *
 * <p>Every iteration it gives one decode token to each running sequence, then fills the rest
 * of the budget with prompt chunks, oldest request first, and runs them all as one ragged
 * {@link PagedLlama#forward} step. With chunk 0 it is the monolithic baseline: a waiting
 * prompt is read whole in the next pass, however long. Sequences join as they arrive and
 * leave as they finish (iteration-level batching), memory permitting: a request starts only
 * when the store can hold its prompt. Decoding is greedy.
 *
 * <p>The core {@code Llama.prefillChunk} splits one long read into pieces to bound the size of
 * the attention scores; this scheduler interleaves those pieces with other sequences' decode
 * steps across passes, which needs the per-sequence store of {@link PagedLlama}.
 *
 * <p>Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve"
 * (OSDI 2024), <a href="https://arxiv.org/abs/2403.02310">arXiv:2403.02310</a>.
 */
public final class ChunkedPrefillScheduler {

    /**
     * One request.
     *
     * @param id an id for the report
     * @param prompt the prompt
     * @param maxNew tokens to generate
     * @param arrival the iteration it arrives at
     */
    public record Request(int id, int[] prompt, int maxNew, int arrival) {
    }

    /**
     * A finished request.
     *
     * @param id the request's id
     * @param tokens the generated tokens
     * @param arrival the iteration it arrived at
     * @param tokenIterations the iteration that produced each token
     * @param tokenMillis the wall time since the run began at which each token came out
     * @param tokenWork the pass tokens run since the run began when each token came out
     */
    public record Completion(int id, int[] tokens, int arrival, int[] tokenIterations, double[] tokenMillis,
            long[] tokenWork) {
    }

    /**
     * What a run measured.
     *
     * @param completions the requests, by id
     * @param iterations the passes run
     * @param passTokens the tokens of each pass
     * @param millis the wall time
     */
    public record Report(List<Completion> completions, int iterations, int[] passTokens, double millis) {

        /** @return all generated tokens per second of wall time */
        public double tokensPerSecond() {
            long n = 0;
            for (Completion c : completions) {
                n += c.tokens().length;
            }
            return n / (millis / 1000);
        }

        /**
         * The gaps between consecutive tokens of each request, in pass tokens run in between
         * (the deterministic measure of compute a waiting sequence sat through).
         *
         * @return every gap
         */
        public long[] decodeGapsInWork() {
            List<Long> gaps = new ArrayList<>();
            for (Completion c : completions) {
                for (int i = 1; i < c.tokenWork().length; i++) {
                    gaps.add(c.tokenWork()[i] - c.tokenWork()[i - 1]);
                }
            }
            return gaps.stream().mapToLong(Long::longValue).sorted().toArray();
        }

        /** @return the gaps between consecutive tokens of each request, in milliseconds */
        public double[] decodeGapsInMillis() {
            List<Double> gaps = new ArrayList<>();
            for (Completion c : completions) {
                for (int i = 1; i < c.tokenMillis().length; i++) {
                    gaps.add(c.tokenMillis()[i] - c.tokenMillis()[i - 1]);
                }
            }
            return gaps.stream().mapToDouble(Double::doubleValue).sorted().toArray();
        }

        /**
         * @param sorted ascending values
         * @param q the quantile in [0, 1]
         * @return the nearest-rank quantile
         */
        public static double quantile(double[] sorted, double q) {
            return sorted.length == 0 ? 0 : sorted[Math.max(0, Math.min(sorted.length - 1,
                    (int) Math.ceil(q * sorted.length) - 1))];
        }

        /**
         * @param sorted ascending values
         * @param q the quantile in [0, 1]
         * @return the nearest-rank quantile
         */
        public static long quantile(long[] sorted, double q) {
            return sorted.length == 0 ? 0 : sorted[Math.max(0, Math.min(sorted.length - 1,
                    (int) Math.ceil(q * sorted.length) - 1))];
        }
    }

    private static final class Running {
        final Request request;
        final int sequence;
        int read;
        final List<Integer> out = new ArrayList<>();
        final List<Integer> iterations = new ArrayList<>();
        final List<Double> millis = new ArrayList<>();
        final List<Long> work = new ArrayList<>();

        Running(Request request, int sequence) {
            this.request = request;
            this.sequence = sequence;
        }

        boolean prefilling() {
            return read < request.prompt().length;
        }
    }

    private final PagedLlama model;
    private final int budget;
    private final int chunk;

    /**
     * @param model the model and its store
     * @param budget the most tokens per pass (decode tokens first, then prompt chunks)
     * @param chunk the most prompt tokens per request per pass, or 0 to read prompts whole
     */
    public ChunkedPrefillScheduler(PagedLlama model, int budget, int chunk) {
        if (budget < 1 || chunk < 0) {
            throw new IllegalArgumentException("budget must be positive and chunk at least 0");
        }
        this.model = Objects.requireNonNull(model, "model");
        this.budget = budget;
        this.chunk = chunk;
    }

    /**
     * Serves requests to the end.
     *
     * @param requests the requests, in any order
     * @return what happened
     */
    public Report run(List<Request> requests) {
        KvBlockStore store = model.store();
        Deque<Request> pending = new ArrayDeque<>(requests.stream()
                .sorted(Comparator.comparingInt(Request::arrival)).toList());
        List<Running> running = new ArrayList<>();
        List<Completion> done = new ArrayList<>();
        List<Integer> passTokens = new ArrayList<>();
        int vocab = model.vocabSize();
        long work = 0;
        long start = System.nanoTime();
        int iteration = 0;
        while (!pending.isEmpty() || !running.isEmpty()) {
            while (!pending.isEmpty() && pending.peek().arrival() <= iteration) {
                Request r = pending.poll();
                running.add(new Running(r, store.newSequence()));
            }
            List<PagedLlama.Pass> passes = new ArrayList<>();
            List<Running> owners = new ArrayList<>();
            int used = 0;
            for (Running r : running) {
                if (!r.prefilling() && store.canAppend(r.sequence, 1)) {
                    passes.add(PagedLlama.Pass.chain(r.sequence, r.out.getLast()));
                    owners.add(r);
                    used++;
                }
            }
            for (Running r : running) {
                if (!r.prefilling()) {
                    continue;
                }
                int left = r.request.prompt().length - r.read;
                int take = chunk == 0 ? left : Math.min(Math.min(left, chunk), budget - used);
                if (take <= 0 || !store.canAppend(r.sequence, take)) {
                    continue;
                }
                passes.add(PagedLlama.Pass.chain(r.sequence, Arrays.copyOfRange(r.request.prompt(), r.read,
                        r.read + take)));
                owners.add(r);
                used += take;
            }
            if (passes.isEmpty()) {
                if (running.isEmpty() && !pending.isEmpty()) {
                    iteration = Math.max(iteration + 1, pending.peek().arrival());
                    continue;
                }
                throw new IllegalStateException("the store cannot hold any running request");
            }
            List<PagedLlama.Output> outs = model.forward(passes, false);
            work += used;
            passTokens.add(used);
            double now = (System.nanoTime() - start) / 1e6;
            for (int i = 0; i < passes.size(); i++) {
                Running r = owners.get(i);
                boolean wasPrefill = r.prefilling();
                if (wasPrefill) {
                    r.read += passes.get(i).tokens().length;
                    if (r.prefilling()) {
                        continue;   // more chunks to go: no token yet
                    }
                }
                r.out.add(LogitMath.argmax(outs.get(i).row(0, vocab)));
                r.iterations.add(iteration);
                r.millis.add(now);
                r.work.add(work);
            }
            for (int i = running.size() - 1; i >= 0; i--) {
                Running r = running.get(i);
                if (r.out.size() >= r.request.maxNew()) {
                    store.free(r.sequence);
                    running.remove(i);
                    done.add(new Completion(r.request.id(), r.out.stream().mapToInt(Integer::intValue).toArray(),
                            r.request.arrival(), r.iterations.stream().mapToInt(Integer::intValue).toArray(),
                            r.millis.stream().mapToDouble(Double::doubleValue).toArray(),
                            r.work.stream().mapToLong(Long::longValue).toArray()));
                }
            }
            iteration++;
        }
        done.sort(Comparator.comparingInt(Completion::id));
        return new Report(done, iteration, passTokens.stream().mapToInt(Integer::intValue).toArray(),
                (System.nanoTime() - start) / 1e6);
    }
}
