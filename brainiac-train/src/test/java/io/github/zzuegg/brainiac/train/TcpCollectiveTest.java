package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.junit.jupiter.api.Test;

class TcpCollectiveTest {

    private static final int WORLD = 3;

    @Test
    void shardsTrainLikeTheFullBatch() throws Exception {
        int port;
        try (ServerSocket s = new ServerSocket(0)) {
            port = s.getLocalPort();
        }
        InetSocketAddress at = new InetSocketAddress(InetAddress.getLoopbackAddress(), port);
        SplittableRandom r = new SplittableRandom(4);
        Tensor x = Tensor.normal(Shape.of(9, 3), 0, 1, r);
        Tensor y = Tensor.normal(Shape.of(9, 1), 0, 1, r);
        Tensor w0 = Tensor.normal(Shape.of(3, 1), 0, 1, r);

        Var ref = Var.param(w0);
        Optimizer sgd = Optimizer.sgd(0.1f);
        for (int step = 0; step < 4; step++) {
            Var.of(x).matmul(ref).sub(Var.of(y)).square().mean().backward();
            sgd.step(List.of(ref));
        }

        ExecutorService pool = Executors.newFixedThreadPool(WORLD);
        List<Future<float[]>> results = new ArrayList<>();
        for (int rank = 0; rank < WORLD; rank++) {
            int me = rank;
            results.add(pool.submit(() -> {
                try (TcpCollective c = TcpCollective.join(me, WORLD, at)) {
                    float[] sum = {me, 10 * me};
                    c.allReduceSum(sum);
                    assertArrayEquals(new float[] {3, 30}, sum);
                    // start from rank 0's weights, whatever the others had
                    Var w = Var.param(me == 0 ? w0 : Tensor.zeros(Shape.of(3, 1)));
                    c.broadcast(List.of(w));
                    Optimizer opt = Optimizer.sgd(0.1f);
                    Tensor xs = x.slice(0, 3 * me, 3 * me + 3);
                    Tensor ys = y.slice(0, 3 * me, 3 * me + 3);
                    for (int step = 0; step < 4; step++) {
                        Var.of(xs).matmul(w).sub(Var.of(ys)).square().mean().backward();
                        c.averageGradients(List.of(w));
                        opt.step(List.of(w));
                    }
                    c.barrier();
                    return w.value().toArray();
                }
            }));
        }
        for (Future<float[]> f : results) {
            assertArrayEquals(ref.value().toArray(), f.get(), 1e-5f);
        }
        pool.shutdown();
    }
}
