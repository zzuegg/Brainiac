package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.llm.Llama;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * Prefill/decode disaggregation (DistServe, Splitwise): prompts are read on one worker and
 * tokens generated on another, with the KV cache handed from the first to the second.
 * Prefill is compute-bound and decode memory-bound; run together on one worker (even with
 * chunked prefill) every pass that carries prompt tokens slows the decode tokens riding in it,
 * so time-per-output-token (TPOT) suffers whenever prompts arrive, and the batch is tuned for
 * neither phase. Split, the decode worker's passes contain only decode tokens, and each worker
 * can be sized for its own latency goal (time-to-first-token, TTFT, for prefill). The price is
 * the KV transfer and a prefill queue on fewer workers. What counts is <em>goodput</em>: the
 * requests per second that meet both latency goals.
 *
 * <p>The workers run real {@link PagedLlama} passes on two separate stores (the KV hand-off
 * copies every layer's keys and values from the prefill store into the decode store), so the
 * tokens are exactly the colocated scheduler's; time is a simulated clock from a {@link Cost}
 * model (a pass costs {@code fixed + perToken * tokens}; a hand-off costs
 * {@code transferPerToken * promptLength} and overlaps the next prefill), so goodput is
 * deterministic on a shared CPU. {@link #colocated} is the baseline at the same number of
 * workers: replicas each running chunked prefill as {@link ChunkedPrefillScheduler} does
 * (decode tokens first, then prompt chunks up to a token budget), on the same clock.
 *
 * <p>Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large
 * Language Model Serving" (OSDI 2024), <a href="https://arxiv.org/abs/2401.09670">arXiv:2401.09670</a>.
 */
public final class DisaggregatedScheduler {

    /**
     * A request.
     *
     * @param id an id for the report
     * @param prompt the prompt
     * @param maxNew tokens to generate, at least 1
     * @param arrival when it arrives, in simulated milliseconds
     */
    public record Request(int id, int[] prompt, int maxNew, double arrival) {
    }

    /**
     * The simulated clock.
     *
     * @param fixed milliseconds per pass whatever its size (weights read once per pass)
     * @param perToken milliseconds per token in a pass
     * @param transferPerToken milliseconds to move one token's KV between workers
     */
    public record Cost(double fixed, double perToken, double transferPerToken) {

        /**
         * @param tokens the tokens of a pass
         * @return its milliseconds
         */
        public double pass(int tokens) {
            return fixed + perToken * tokens;
        }
    }

    /**
     * Latency goals.
     *
     * @param ttft the most milliseconds from arrival to the first token
     * @param tpot the most mean milliseconds between later tokens
     */
    public record Slo(double ttft, double tpot) {
    }

    /**
     * A finished request.
     *
     * @param id its id
     * @param tokens the generated tokens
     * @param arrival its arrival
     * @param firstToken when its first token came out
     * @param finish when its last token came out
     */
    public record Completion(int id, int[] tokens, double arrival, double firstToken, double finish) {

        /** @return milliseconds to the first token */
        public double ttft() {
            return firstToken - arrival;
        }

        /** @return mean milliseconds between later tokens (0 for one token) */
        public double tpot() {
            return tokens.length <= 1 ? 0 : (finish - firstToken) / (tokens.length - 1);
        }

        /**
         * @param slo the goals
         * @return whether it met both
         */
        public boolean meets(Slo slo) {
            return ttft() <= slo.ttft() && tpot() <= slo.tpot();
        }
    }

    /**
     * A run.
     *
     * @param completions every request, by id
     * @param makespan when the last token came out
     */
    public record Report(List<Completion> completions, double makespan) {

        /**
         * @param slo the goals
         * @return the share of requests that met both
         */
        public double attainment(Slo slo) {
            return completions.stream().filter(c -> c.meets(slo)).count() / (double) completions.size();
        }

        /**
         * @param slo the goals
         * @return requests meeting both goals per simulated second
         */
        public double goodput(Slo slo) {
            return completions.stream().filter(c -> c.meets(slo)).count() / (makespan / 1000);
        }

        /** @return the mean time per output token */
        public double meanTpot() {
            return completions.stream().mapToDouble(Completion::tpot).average().orElse(0);
        }

        /** @return the mean time to first token */
        public double meanTtft() {
            return completions.stream().mapToDouble(Completion::ttft).average().orElse(0);
        }
    }

    private static final class Running {
        final Request request;
        int sequence;
        int read;
        double ready;
        double first = Double.NaN;
        double last;
        final List<Integer> out = new ArrayList<>();

        Running(Request request) {
            this.request = request;
        }

        boolean prefilling() {
            return read < request.prompt().length;
        }

        Completion done() {
            return new Completion(request.id(), out.stream().mapToInt(Integer::intValue).toArray(), request.arrival(),
                    first, last);
        }
    }

    private final Llama model;
    private final Cost cost;
    private final int blockSize;
    private final int blocks;

    /**
     * @param model the weights every worker runs
     * @param cost the clock
     * @param blockSize slots per KV block on each worker
     * @param blocks blocks per worker
     */
    public DisaggregatedScheduler(Llama model, Cost cost, int blockSize, int blocks) {
        this.model = Objects.requireNonNull(model, "model");
        this.cost = Objects.requireNonNull(cost, "cost");
        this.blockSize = blockSize;
        this.blocks = blocks;
    }

    /**
     * One prefill worker and one decode worker.
     *
     * @param requests the requests
     * @param prefillTokens the most prompt tokens per prefill pass (at least one whole prompt
     *     always runs)
     * @return the report
     */
    public Report disaggregated(List<Request> requests, int prefillTokens) {
        PagedLlama prefill = PagedLlama.of(model, blockSize, blocks);
        PagedLlama decode = PagedLlama.of(model, blockSize, blocks);
        List<Running> queue = new ArrayList<>(requests.stream().sorted(Comparator.comparingDouble(Request::arrival))
                .map(Running::new).toList());
        int vocab = model.vocabSize();
        // the prefill worker: batches of whole prompts, first come first served
        double clock = 0;
        int next = 0;
        while (next < queue.size()) {
            clock = Math.max(clock, queue.get(next).request.arrival());
            List<Running> batch = new ArrayList<>();
            int tokens = 0;
            while (next < queue.size() && queue.get(next).request.arrival() <= clock
                    && (batch.isEmpty() || tokens + queue.get(next).request.prompt().length <= prefillTokens)) {
                Running r = queue.get(next++);
                r.sequence = prefill.store().newSequence();
                batch.add(r);
                tokens += r.request.prompt().length;
            }
            List<PagedLlama.Pass> passes = batch.stream().map(r -> PagedLlama.Pass.chain(r.sequence,
                    r.request.prompt())).toList();
            List<PagedLlama.Output> outs = prefill.forward(passes, false);
            clock += cost.pass(tokens);
            for (int i = 0; i < batch.size(); i++) {
                Running r = batch.get(i);
                r.read = r.request.prompt().length;
                r.out.add(LogitMath.argmax(outs.get(i).row(0, vocab)));
                r.first = clock;
                r.last = clock;
                r.ready = clock + cost.transferPerToken() * r.read;
                r.sequence = handOff(prefill.store(), decode.store(), r.sequence, r.read);
            }
        }
        // the decode worker: continuous batching of handed-off requests, one token each per pass
        List<Running> waiting = new ArrayList<>(queue);
        waiting.sort(Comparator.comparingDouble(r -> r.ready));
        List<Running> running = new ArrayList<>();
        List<Completion> done = new ArrayList<>();
        for (Running r : waiting) {
            if (r.out.size() >= r.request.maxNew()) {
                decode.store().free(r.sequence);
                done.add(r.done());
            }
        }
        waiting.removeIf(r -> r.out.size() >= r.request.maxNew());
        double t = 0;
        while (!waiting.isEmpty() || !running.isEmpty()) {
            if (running.isEmpty()) {
                t = Math.max(t, waiting.getFirst().ready);
            }
            while (!waiting.isEmpty() && waiting.getFirst().ready <= t) {
                running.add(waiting.removeFirst());
            }
            List<PagedLlama.Pass> passes = running.stream().map(r -> PagedLlama.Pass.chain(r.sequence,
                    r.out.getLast())).toList();
            List<PagedLlama.Output> outs = decode.forward(passes, false);
            t += cost.pass(passes.size());
            for (int i = 0; i < running.size(); i++) {
                Running r = running.get(i);
                r.out.add(LogitMath.argmax(outs.get(i).row(0, vocab)));
                r.last = t;
            }
            for (int i = running.size() - 1; i >= 0; i--) {
                Running r = running.get(i);
                if (r.out.size() >= r.request.maxNew()) {
                    decode.store().free(r.sequence);
                    done.add(r.done());
                    running.remove(i);
                }
            }
        }
        return report(done);
    }

    /** Copies a sequence's keys and values from one store to another and frees the source. */
    private static int handOff(KvBlockStore from, KvBlockStore to, int sequence, int length) {
        int target = to.newSequence();
        to.append(target, length);
        Tensor none = Tensor.zeros(Shape.of(1, 1, 1));
        for (int l = 0; l < from.layers(); l++) {
            KvBlockStore.View v = from.view(sequence, l, none);
            to.write(target, l, 0, v.keys(), v.values());
        }
        from.free(sequence);
        return target;
    }

    /**
     * The colocated baseline: {@code replicas} workers, requests assigned round-robin, each
     * running chunked prefill with decode tokens first.
     *
     * @param requests the requests
     * @param replicas the workers
     * @param budget the most tokens per pass
     * @param chunk the most prompt tokens per request per pass
     * @return the report
     */
    public Report colocated(List<Request> requests, int replicas, int budget, int chunk) {
        List<Request> sorted = requests.stream().sorted(Comparator.comparingDouble(Request::arrival)).toList();
        List<Completion> done = new ArrayList<>();
        for (int w = 0; w < replicas; w++) {
            List<Request> mine = new ArrayList<>();
            for (int i = w; i < sorted.size(); i += replicas) {
                mine.add(sorted.get(i));
            }
            done.addAll(chunked(mine, budget, chunk));
        }
        return report(done);
    }

    private List<Completion> chunked(List<Request> requests, int budget, int chunk) {
        PagedLlama worker = PagedLlama.of(model, blockSize, blocks);
        KvBlockStore store = worker.store();
        int vocab = model.vocabSize();
        List<Running> pending = new ArrayList<>(requests.stream().map(Running::new).toList());
        List<Running> running = new ArrayList<>();
        List<Completion> done = new ArrayList<>();
        double t = 0;
        while (!pending.isEmpty() || !running.isEmpty()) {
            if (running.isEmpty()) {
                t = Math.max(t, pending.getFirst().request.arrival());
            }
            while (!pending.isEmpty() && pending.getFirst().request.arrival() <= t) {
                Running r = pending.removeFirst();
                r.sequence = store.newSequence();
                running.add(r);
            }
            List<PagedLlama.Pass> passes = new ArrayList<>();
            List<Running> owners = new ArrayList<>();
            int used = 0;
            for (Running r : running) {
                if (!r.prefilling()) {
                    passes.add(PagedLlama.Pass.chain(r.sequence, r.out.getLast()));
                    owners.add(r);
                    used++;
                }
            }
            for (Running r : running) {
                if (!r.prefilling()) {
                    continue;
                }
                int take = Math.min(Math.min(r.request.prompt().length - r.read, chunk), budget - used);
                if (take <= 0) {
                    continue;
                }
                passes.add(PagedLlama.Pass.chain(r.sequence, Arrays.copyOfRange(r.request.prompt(), r.read,
                        r.read + take)));
                owners.add(r);
                used += take;
            }
            List<PagedLlama.Output> outs = worker.forward(passes, false);
            t += cost.pass(used);
            for (int i = 0; i < passes.size(); i++) {
                Running r = owners.get(i);
                if (r.prefilling()) {
                    r.read += passes.get(i).tokens().length;
                    if (r.prefilling()) {
                        continue;
                    }
                    r.first = t;
                }
                r.out.add(LogitMath.argmax(outs.get(i).row(0, vocab)));
                r.last = t;
            }
            for (int i = running.size() - 1; i >= 0; i--) {
                Running r = running.get(i);
                if (r.out.size() >= r.request.maxNew()) {
                    store.free(r.sequence);
                    done.add(r.done());
                    running.remove(i);
                }
            }
        }
        return done;
    }

    private static Report report(List<Completion> done) {
        done.sort(Comparator.comparingInt(Completion::id));
        double end = 0;
        for (Completion c : done) {
            end = Math.max(end, c.finish());
        }
        return new Report(done, end);
    }
}
