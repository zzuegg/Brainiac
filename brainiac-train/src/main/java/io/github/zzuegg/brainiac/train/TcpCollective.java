package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.List;

/**
 * Collective operations between training processes over TCP, for data-parallel training on
 * several machines: rank 0 listens, the others connect, and each operation gathers on rank 0
 * and sends the result back to everyone. Every process must call the same operations in the
 * same order.
 *
 * <pre>{@code
 * try (TcpCollective c = TcpCollective.join(rank, world, new InetSocketAddress("10.0.0.1", 29500))) {
 *     c.broadcast(parameters);                 // start from rank 0's weights
 *     for (Batch b : myShard) {
 *         loss(b).backward();
 *         c.averageGradients(parameters);
 *         optimizer.step(parameters);
 *     }
 * }
 * }</pre>
 */
public final class TcpCollective implements AutoCloseable {

    private final int rank;
    private final int world;
    private final ServerSocket server;
    private final List<Socket> peers = new ArrayList<>();
    private final List<DataInputStream> ins = new ArrayList<>();
    private final List<DataOutputStream> outs = new ArrayList<>();

    private TcpCollective(int rank, int world, ServerSocket server) {
        this.rank = rank;
        this.world = world;
        this.server = server;
    }

    /**
     * Joins a group: rank 0 waits for the others, which retry connecting for up to a minute.
     *
     * @param rank this process's rank, from 0
     * @param world the number of processes
     * @param coordinator where rank 0 listens
     * @return the connected group
     */
    public static TcpCollective join(int rank, int world, InetSocketAddress coordinator) {
        if (world < 1 || rank < 0 || rank >= world) {
            throw new IllegalArgumentException("rank " + rank + " of " + world);
        }
        try {
            if (rank == 0) {
                ServerSocket s = new ServerSocket();
                s.setReuseAddress(true);
                s.bind(coordinator);
                TcpCollective c = new TcpCollective(0, world, s);
                Socket[] byRank = new Socket[world];
                DataInputStream[] in = new DataInputStream[world];
                for (int i = 1; i < world; i++) {
                    Socket p = s.accept();
                    p.setTcpNoDelay(true);
                    DataInputStream d = new DataInputStream(new BufferedInputStream(p.getInputStream(), 1 << 16));
                    int r = d.readInt();
                    if (r <= 0 || r >= world || byRank[r] != null) {
                        p.close();
                        throw new IOException("bad or repeated rank " + r);
                    }
                    byRank[r] = p;
                    in[r] = d;
                }
                for (int r = 1; r < world; r++) {
                    c.peers.add(byRank[r]);
                    c.ins.add(in[r]);
                    c.outs.add(new DataOutputStream(new BufferedOutputStream(byRank[r].getOutputStream(), 1 << 16)));
                }
                return c;
            }
            long deadline = System.nanoTime() + 60_000_000_000L;
            while (true) {
                try {
                    Socket p = new Socket();
                    p.connect(coordinator, 5000);
                    p.setTcpNoDelay(true);
                    TcpCollective c = new TcpCollective(rank, world, null);
                    c.peers.add(p);
                    c.ins.add(new DataInputStream(new BufferedInputStream(p.getInputStream(), 1 << 16)));
                    DataOutputStream out = new DataOutputStream(new BufferedOutputStream(p.getOutputStream(), 1 << 16));
                    out.writeInt(rank);
                    out.flush();
                    c.outs.add(out);
                    return c;
                } catch (IOException e) {
                    if (System.nanoTime() > deadline) {
                        throw e;
                    }
                    Thread.sleep(200);
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted", e);
        }
    }

    /** @return this process's rank */
    public int rank() {
        return rank;
    }

    /** @return the number of processes */
    public int world() {
        return world;
    }

    /**
     * Sums arrays over all processes; every process gets the sum.
     *
     * @param values this process's values, the same length everywhere; replaced by the sum
     */
    public void allReduceSum(float[] values) {
        try {
            if (rank == 0) {
                float[] other = new float[values.length];
                for (DataInputStream in : ins) {
                    read(in, other);
                    for (int i = 0; i < values.length; i++) {
                        values[i] += other[i];
                    }
                }
                for (DataOutputStream out : outs) {
                    write(out, values);
                }
            } else {
                write(outs.getFirst(), values);
                read(ins.getFirst(), values);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * Copies rank 0's array to every process.
     *
     * @param values rank 0's values to send, or the place the others receive them
     */
    public void broadcast(float[] values) {
        try {
            if (rank == 0) {
                for (DataOutputStream out : outs) {
                    write(out, values);
                }
            } else {
                read(ins.getFirst(), values);
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** Waits until every process has called it. */
    public void barrier() {
        allReduceSum(new float[1]);
    }

    /**
     * Gives every process rank 0's parameter values, as training starts.
     *
     * @param parameters the parameters, in the same order everywhere
     */
    public void broadcast(List<Var> parameters) {
        for (Var p : parameters) {
            float[] v = p.value().to(DType.FLOAT32).toArray();
            broadcast(v);
            if (rank != 0) {
                p.assign(Tensor.of(p.shape(), v).to(p.value().device()).to(p.value().dtype()));
            }
        }
    }

    /**
     * Replaces each parameter's gradient by its mean over the processes (a missing gradient
     * counts as zero), so equal shards give the full batch's gradient.
     *
     * @param parameters the parameters, in the same order everywhere
     */
    public void averageGradients(List<Var> parameters) {
        int total = 0;
        for (Var p : parameters) {
            total += Math.toIntExact(p.shape().size());
        }
        float[] all = new float[total];
        int at = 0;
        for (Var p : parameters) {
            int n = Math.toIntExact(p.shape().size());
            if (p.grad() != null) {
                System.arraycopy(p.grad().to(DType.FLOAT32).toArray(), 0, all, at, n);
            }
            at += n;
        }
        allReduceSum(all);
        at = 0;
        for (Var p : parameters) {
            int n = Math.toIntExact(p.shape().size());
            float[] g = new float[n];
            for (int i = 0; i < n; i++) {
                g[i] = all[at + i] / world;
            }
            at += n;
            Device d = p.value().device();
            p.setGrad(Tensor.of(p.shape(), g).to(d).to(p.value().dtype()));
        }
    }

    private static void write(DataOutputStream out, float[] v) throws IOException {
        ByteBuffer b = ByteBuffer.allocate(4 + 4 * v.length).order(ByteOrder.BIG_ENDIAN);
        b.putInt(v.length);
        b.asFloatBuffer().put(v);
        out.write(b.array());
        out.flush();
    }

    private static void read(DataInputStream in, float[] into) throws IOException {
        int n = in.readInt();
        if (n != into.length) {
            throw new IOException("expected " + into.length + " values, got " + n
                    + "; every process must call the same operations");
        }
        byte[] bytes = new byte[4 * n];
        in.readFully(bytes);
        ByteBuffer.wrap(bytes).order(ByteOrder.BIG_ENDIAN).asFloatBuffer().get(into);
    }

    @Override
    public void close() {
        for (Socket p : peers) {
            try {
                p.close();
            } catch (IOException e) {
                // closing anyway
            }
        }
        if (server != null) {
            try {
                server.close();
            } catch (IOException e) {
                // closing anyway
            }
        }
    }
}
