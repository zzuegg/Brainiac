package io.github.zzuegg.brainiac.experimental.parallel;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicLongArray;
import java.util.function.IntFunction;

/**
 * A process group of simulated devices: {@code ranks} Java threads that exchange data through
 * collective operations (all-reduce, reduce-scatter, all-gather, all-to-all, broadcast) and
 * point-to-point links, counting the bytes and messages each rank sends.
 *
 * <p>Every collective is called by all ranks of the group from their own threads, as in NCCL or
 * MPI; a call blocks until every rank has arrived. Results are computed in rank order, so every
 * rank sees bit-identical sums and repeated runs are deterministic. Bytes are counted as the
 * bandwidth-optimal ring algorithms send them, which is what real libraries do: an all-reduce of
 * {@code B} bytes sends {@code 2 (N - 1) / N * B} per rank, a reduce-scatter or an all-gather
 * {@code (N - 1) / N * B} per rank, and an all-to-all every byte addressed to another rank.
 * Megatron-LM, ZeRO and expert parallelism are written in terms of exactly these operations,
 * so the counts are the communication volume those papers analyse.
 *
 * <p>Papers: Thakur et al., "Optimization of Collective Communication Operations in MPICH"
 * (2005); Shoeybi et al., "Megatron-LM", <a href="https://arxiv.org/abs/1909.08053">arXiv:1909.08053</a>.
 */
public final class Collectives {

    private static final int FLOAT = Float.BYTES;

    private final int ranks;
    private final CyclicBarrier barrier;
    private final Object[] slots;
    private final AtomicLongArray bytes;
    private final AtomicLongArray messages;
    private final List<List<LinkedBlockingQueue<float[]>>> links = new ArrayList<>();

    /**
     * Creates a group.
     *
     * @param ranks the number of ranks, at least 1
     */
    public Collectives(int ranks) {
        if (ranks < 1) {
            throw new IllegalArgumentException("ranks must be at least 1, got " + ranks);
        }
        this.ranks = ranks;
        this.barrier = new CyclicBarrier(ranks);
        this.slots = new Object[ranks];
        this.bytes = new AtomicLongArray(ranks);
        this.messages = new AtomicLongArray(ranks);
        for (int i = 0; i < ranks; i++) {
            List<LinkedBlockingQueue<float[]>> row = new ArrayList<>();
            for (int j = 0; j < ranks; j++) {
                row.add(new LinkedBlockingQueue<>());
            }
            links.add(row);
        }
    }

    /** @return the number of ranks */
    public int ranks() {
        return ranks;
    }

    /**
     * @param rank a rank
     * @return the bytes that rank has sent since the last {@link #reset()}
     */
    public long bytesSent(int rank) {
        return bytes.get(rank);
    }

    /** @return the bytes all ranks have sent since the last {@link #reset()} */
    public long totalBytes() {
        long s = 0;
        for (int i = 0; i < ranks; i++) {
            s += bytes.get(i);
        }
        return s;
    }

    /** @return the messages all ranks have sent since the last {@link #reset()} */
    public long totalMessages() {
        long s = 0;
        for (int i = 0; i < ranks; i++) {
            s += messages.get(i);
        }
        return s;
    }

    /** Clears the byte and message counters. */
    public void reset() {
        for (int i = 0; i < ranks; i++) {
            bytes.set(i, 0);
            messages.set(i, 0);
        }
    }

    /**
     * Adds traffic that was not sent through this group, such as a modelled transfer.
     *
     * @param rank the sending rank
     * @param sentBytes the bytes
     * @param sentMessages the messages
     */
    public void account(int rank, long sentBytes, long sentMessages) {
        bytes.addAndGet(rank, sentBytes);
        messages.addAndGet(rank, sentMessages);
    }

    private void await() {
        try {
            barrier.await(120, java.util.concurrent.TimeUnit.SECONDS);
        } catch (java.util.concurrent.TimeoutException e) {
            barrier.reset();
            throw new IllegalStateException("a rank did not arrive at a collective", e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        } catch (BrokenBarrierException e) {
            throw new IllegalStateException("another rank failed", e);
        }
    }

    /** Deposits this rank's value, waits for all, and returns everyone's values in rank order. */
    private Object[] exchange(int rank, Object value) {
        slots[rank] = value;
        await();
        Object[] all = slots.clone();
        await();   // nobody overwrites a slot before everyone has read it
        return all;
    }

    /**
     * Sums arrays over all ranks; every rank gets the sum.
     *
     * @param rank the calling rank
     * @param data this rank's values, the same length on every rank
     * @return the elementwise sum
     */
    public float[] allReduce(int rank, float[] data) {
        Object[] all = exchange(rank, data);
        float[] sum = new float[data.length];
        for (Object o : all) {
            float[] a = (float[]) o;
            for (int i = 0; i < sum.length; i++) {
                sum[i] += a[i];
            }
        }
        account(rank, ringBytes(2L * data.length * FLOAT), 2L * (ranks - 1));
        return sum;
    }

    /**
     * Sums tensors over all ranks.
     *
     * @param rank the calling rank
     * @param t this rank's tensor
     * @return the sum, of the same shape
     */
    public Tensor allReduce(int rank, Tensor t) {
        return Tensor.of(t.shape(), allReduce(rank, t.toArray()));
    }

    /**
     * Sums arrays over all ranks and gives each rank only its chunk of the sum (chunk {@code r} of
     * {@code N} equal parts; the length must divide evenly).
     *
     * @param rank the calling rank
     * @param data this rank's values
     * @return this rank's chunk of the sum
     */
    public float[] reduceScatter(int rank, float[] data) {
        if (data.length % ranks != 0) {
            throw new IllegalArgumentException("length " + data.length + " does not split over " + ranks + " ranks");
        }
        Object[] all = exchange(rank, data);
        int chunk = data.length / ranks;
        float[] out = new float[chunk];
        for (Object o : all) {
            float[] a = (float[]) o;
            for (int i = 0; i < chunk; i++) {
                out[i] += a[rank * chunk + i];
            }
        }
        account(rank, ringBytes((long) data.length * FLOAT), ranks - 1);
        return out;
    }

    /**
     * Reduce-scatter of a tensor along axis 0.
     *
     * @param rank the calling rank
     * @param t this rank's tensor {@code [N * rows, ...]}
     * @return this rank's rows of the sum {@code [rows, ...]}
     */
    public Tensor reduceScatter(int rank, Tensor t) {
        int[] dims = t.shape().toArray();
        dims[0] /= ranks;
        return Tensor.of(Shape.of(dims), reduceScatter(rank, t.toArray()));
    }

    /**
     * Concatenates every rank's chunk in rank order; chunks may differ in length.
     *
     * @param rank the calling rank
     * @param chunk this rank's chunk
     * @return all chunks, concatenated
     */
    public float[] allGather(int rank, float[] chunk) {
        Object[] all = exchange(rank, chunk);
        int total = 0;
        for (Object o : all) {
            total += ((float[]) o).length;
        }
        float[] out = new float[total];
        int at = 0;
        for (Object o : all) {
            float[] a = (float[]) o;
            System.arraycopy(a, 0, out, at, a.length);
            at += a.length;
        }
        account(rank, (long) chunk.length * FLOAT * (ranks - 1), ranks - 1);
        return out;
    }

    /**
     * All-gather of tensors along axis 0.
     *
     * @param rank the calling rank
     * @param t this rank's tensor {@code [rows, ...]}, the same shape on every rank
     * @return {@code [N * rows, ...]}
     */
    public Tensor allGather(int rank, Tensor t) {
        int[] dims = t.shape().toArray();
        dims[0] *= ranks;
        return Tensor.of(Shape.of(dims), allGather(rank, t.toArray()));
    }

    /**
     * Every rank sends one array to every rank (its own included, which costs nothing).
     *
     * @param rank the calling rank
     * @param send {@code send[j]} goes to rank {@code j}; may be empty
     * @return {@code received[i]}, what rank {@code i} sent to this rank
     */
    public float[][] allToAll(int rank, float[][] send) {
        if (send.length != ranks) {
            throw new IllegalArgumentException("need one array per rank");
        }
        Object[] all = exchange(rank, send);
        float[][] got = new float[ranks][];
        long sent = 0;
        long msgs = 0;
        for (int i = 0; i < ranks; i++) {
            got[i] = ((float[][]) all[i])[rank];
            if (i != rank && send[i].length > 0) {
                sent += (long) send[i].length * FLOAT;
                msgs++;
            }
        }
        account(rank, sent, msgs);
        return got;
    }

    /**
     * Copies rank {@code root}'s array to every rank.
     *
     * @param rank the calling rank
     * @param root the source rank
     * @param data the array on the root, ignored elsewhere
     * @return the root's array
     */
    public float[] broadcast(int rank, int root, float[] data) {
        Object[] all = exchange(rank, rank == root ? data : null);
        float[] src = (float[]) all[root];
        if (rank == root) {
            account(rank, ringBytes((long) src.length * FLOAT), ranks - 1);
        }
        return src.clone();
    }

    /** Waits until every rank has arrived. */
    public void barrier() {
        await();
    }

    /**
     * Sends an array over the point-to-point link {@code from -> to}; does not block.
     *
     * @param from the sending rank
     * @param to the receiving rank
     * @param data the payload
     */
    public void send(int from, int to, float[] data) {
        links.get(from).get(to).add(data);
        if (from != to) {
            account(from, (long) data.length * FLOAT, 1);
        }
    }

    /**
     * Receives the next array on the link {@code from -> to}, blocking until one arrives.
     *
     * @param to the receiving rank
     * @param from the sending rank
     * @return the payload
     */
    public float[] receive(int to, int from) {
        try {
            return links.get(from).get(to).take();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        }
    }

    private long ringBytes(long payload) {
        return payload * (ranks - 1) / ranks;
    }

    /**
     * The bytes one rank sends in a ring all-reduce of {@code bytes}.
     *
     * @param bytes the payload size
     * @param ranks the group size
     * @return {@code 2 (N - 1) / N * bytes}
     */
    public static long allReduceBytes(long bytes, int ranks) {
        return 2 * bytes * (ranks - 1) / ranks;
    }

    /**
     * Runs one task per rank on its own thread and waits for all of them; an exception in any
     * rank is rethrown (a rank that fails breaks the barrier, so the others fail too).
     *
     * @param ranks the number of ranks
     * @param body the work of rank {@code r}
     * @param <T> the result type
     * @return the results in rank order
     */
    public static <T> List<T> run(int ranks, IntFunction<T> body) {
        ExecutorService pool = Executors.newFixedThreadPool(ranks, r -> {
            Thread t = new Thread(r, "brainiac-rank");
            t.setDaemon(true);
            return t;
        });
        try {
            List<Future<T>> work = new ArrayList<>();
            for (int r = 0; r < ranks; r++) {
                int rank = r;
                work.add(pool.submit(() -> body.apply(rank)));
            }
            List<T> out = new ArrayList<>();
            for (Future<T> f : work) {
                out.add(f.get());
            }
            return out;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        } catch (ExecutionException e) {
            Throwable c = e.getCause();
            if (c instanceof RuntimeException re) {
                throw re;
            }
            if (c instanceof Error err) {
                throw err;
            }
            throw new IllegalStateException(c);
        } finally {
            pool.shutdownNow();
        }
    }
}
