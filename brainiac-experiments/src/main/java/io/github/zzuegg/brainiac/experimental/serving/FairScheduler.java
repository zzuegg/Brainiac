package io.github.zzuegg.brainiac.experimental.serving;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.Deque;
import java.util.List;

/**
 * Fair admission for continuous batching with many clients: the virtual token counter (VTC,
 * Sheng et al., "Fairness in Serving Large Language Models", OSDI 2024,
 * <a href="https://arxiv.org/abs/2401.00588">arXiv:2401.00588</a>), against first come first
 * served and per-client request-rate limiting.
 * <ul>
 * <li>{@link #fcfs()}: admit the oldest waiting request; a heavy client starves the others.</li>
 * <li>{@link #vtc}: every client has a counter of the weighted service it received
 *     ({@code inputWeight} per prompt token when admitted, {@code outputWeight} per generated
 *     token); a free batch slot goes to the backlogged client with the smallest counter. When a
 *     client becomes backlogged again its counter is lifted to the smallest counter among the
 *     backlogged clients, so idle time is not banked. The service gap between two backlogged
 *     clients stays bounded while the batch stays full.</li>
 * <li>{@link #requestRateLimit}: each client may start at most {@code limit} requests per
 *     {@code window} iterations; fair under overload but it leaves slots idle when only one
 *     client is active.</li>
 * </ul>
 * {@link #run} drives an iteration-level engine with the cost model of
 * {@code ChunkedPrefillScheduler}'s monolithic mode: each iteration reads every newly admitted
 * prompt whole and one decode token per running sequence, up to {@code slots} sequences.
 */
public final class FairScheduler {

    private enum Kind { FCFS, VTC, RATE }

    private final Kind kind;
    private final double inputWeight;
    private final double outputWeight;
    private final int limit;
    private final int window;

    private FairScheduler(Kind kind, double inputWeight, double outputWeight, int limit, int window) {
        this.kind = kind;
        this.inputWeight = inputWeight;
        this.outputWeight = outputWeight;
        this.limit = limit;
        this.window = window;
    }

    /** @return first come, first served */
    public static FairScheduler fcfs() {
        return new FairScheduler(Kind.FCFS, 1, 2, 0, 1);
    }

    /**
     * @param inputWeight service per prompt token (1 in the paper)
     * @param outputWeight service per generated token (2 in the paper)
     * @return the virtual token counter
     */
    public static FairScheduler vtc(double inputWeight, double outputWeight) {
        return new FairScheduler(Kind.VTC, inputWeight, outputWeight, 0, 1);
    }

    /**
     * @param limit requests a client may start per window
     * @param window the window, in iterations
     * @return per-client request-rate limiting (served first come, first served)
     */
    public static FairScheduler requestRateLimit(int limit, int window) {
        return new FairScheduler(Kind.RATE, 1, 2, limit, window);
    }

    /**
     * A request.
     *
     * @param client the client
     * @param prompt prompt tokens
     * @param output tokens to generate
     * @param arrival the iteration it arrives at
     */
    public record Request(int client, int prompt, int output, int arrival) {
    }

    /**
     * What a run measured.
     *
     * @param iterations iterations until every request finished
     * @param tokens prompt plus generated tokens served
     * @param latency per request, iterations from arrival to its last token, in input order
     * @param service per client and iteration, the weighted service received so far
     *     ({@code [clients][iterations + 1]})
     * @param backlogged per client and iteration, whether it had requests waiting or running
     */
    public record Report(int iterations, long tokens, double[] latency, double[][] service,
            boolean[][] backlogged) {

        /** @return tokens served per iteration */
        public double throughput() {
            return tokens / (double) iterations;
        }

        /**
         * The largest difference in service two clients received over any window of iterations
         * in which both were backlogged throughout.
         *
         * @param a one client
         * @param b another
         * @param span the window
         * @return the gap
         */
        public double serviceGap(int a, int b, int span) {
            double gap = 0;
            for (int t = 0; t + span < service[a].length && t + span <= backlogged[a].length; t++) {
                boolean both = true;
                for (int u = t; u < t + span && both; u++) {
                    both = backlogged[a][u] && backlogged[b][u];
                }
                if (both) {
                    double da = service[a][t + span] - service[a][t];
                    double db = service[b][t + span] - service[b][t];
                    gap = Math.max(gap, Math.abs(da - db));
                }
            }
            return gap;
        }
    }

    private static final class Live {
        final Request r;
        final int index;
        int produced;
        boolean read;

        Live(Request r, int index) {
            this.r = r;
            this.index = index;
        }
    }

    /**
     * Serves requests to the end.
     *
     * @param requests the requests
     * @param clients the number of clients
     * @param slots the most sequences running at once
     * @return the report
     */
    public Report run(List<Request> requests, int clients, int slots) {
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < requests.size(); i++) {
            order.add(i);
        }
        order.sort(Comparator.comparingInt(i -> requests.get(i).arrival()));
        List<Deque<Integer>> queues = new ArrayList<>();
        for (int c = 0; c < clients; c++) {
            queues.add(new ArrayDeque<>());
        }
        double[] counter = new double[clients];
        int[] started = new int[clients];
        int[] runningPer = new int[clients];
        double[] latency = new double[requests.size()];
        List<Live> running = new ArrayList<>();
        List<double[]> service = new ArrayList<>();
        List<boolean[]> backlog = new ArrayList<>();
        double[] served = new double[clients];
        service.add(served.clone());
        int next = 0;
        int done = 0;
        long tokens = 0;
        int t = 0;
        while (done < requests.size()) {
            while (next < order.size() && requests.get(order.get(next)).arrival() <= t) {
                int i = order.get(next++);
                int c = requests.get(i).client();
                if (kind == Kind.VTC && queues.get(c).isEmpty() && runningPer[c] == 0) {
                    double min = Double.POSITIVE_INFINITY;
                    for (int o = 0; o < clients; o++) {
                        if (o != c && (!queues.get(o).isEmpty() || runningPer[o] > 0)) {
                            min = Math.min(min, counter[o]);
                        }
                    }
                    if (min < Double.POSITIVE_INFINITY) {
                        counter[c] = Math.max(counter[c], min);
                    }
                }
                queues.get(c).add(i);
            }
            if (kind == Kind.RATE && t % window == 0) {
                Arrays.fill(started, 0);
            }
            while (running.size() < slots) {
                int pick = choose(requests, queues, counter, started);
                if (pick < 0) {
                    break;
                }
                int i = queues.get(pick).poll();
                running.add(new Live(requests.get(i), i));
                runningPer[pick]++;
                started[pick]++;
            }
            boolean[] b = new boolean[clients];
            for (int c = 0; c < clients; c++) {
                b[c] = !queues.get(c).isEmpty() || runningPer[c] > 0;
            }
            backlog.add(b);
            for (int k = running.size() - 1; k >= 0; k--) {
                Live l = running.get(k);
                int c = l.r.client();
                if (!l.read) {
                    l.read = true;
                    tokens += l.r.prompt();
                    counter[c] += inputWeight * l.r.prompt();
                    served[c] += inputWeight * l.r.prompt();
                }
                l.produced++;
                tokens++;
                counter[c] += outputWeight;
                served[c] += outputWeight;
                if (l.produced >= l.r.output()) {
                    running.remove(k);
                    runningPer[c]--;
                    latency[l.index] = t + 1 - l.r.arrival();
                    done++;
                }
            }
            t++;
            service.add(served.clone());
        }
        double[][] s = new double[clients][service.size()];
        boolean[][] bl = new boolean[clients][backlog.size()];
        for (int u = 0; u < service.size(); u++) {
            for (int c = 0; c < clients; c++) {
                s[c][u] = service.get(u)[c];
            }
        }
        for (int u = 0; u < backlog.size(); u++) {
            for (int c = 0; c < clients; c++) {
                bl[c][u] = backlog.get(u)[c];
            }
        }
        return new Report(t, tokens, latency, s, bl);
    }

    private int choose(List<Request> requests, List<Deque<Integer>> queues, double[] counter, int[] started) {
        int best = -1;
        for (int c = 0; c < queues.size(); c++) {
            if (queues.get(c).isEmpty()) {
                continue;
            }
            switch (kind) {
                case FCFS -> {
                    if (best < 0 || requests.get(queues.get(c).peek()).arrival()
                            < requests.get(queues.get(best).peek()).arrival()) {
                        best = c;
                    }
                }
                case VTC -> {
                    if (best < 0 || counter[c] < counter[best]) {
                        best = c;
                    }
                }
                case RATE -> {
                    if (started[c] < limit && (best < 0 || requests.get(queues.get(c).peek()).arrival()
                            < requests.get(queues.get(best).peek()).arrival())) {
                        best = c;
                    }
                }
            }
        }
        return best;
    }

    @Override
    public String toString() {
        return switch (kind) {
            case FCFS -> "FCFS";
            case VTC -> "VTC(" + inputWeight + ", " + outputWeight + ")";
            case RATE -> "rate limit(" + limit + " per " + window + ")";
        };
    }
}
