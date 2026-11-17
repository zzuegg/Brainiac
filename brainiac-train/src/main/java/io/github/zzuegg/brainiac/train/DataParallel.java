package io.github.zzuegg.brainiac.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * Data-parallel training over several devices in one process: each step splits the batch
 * among the devices, each computes the loss and gradients of its share on its own copy of
 * the parameters, the gradients are averaged on the host (weighted by share) and one
 * optimizer step updates the parameters, which every copy then takes again. The result
 * equals a step on the whole batch, for a loss that is a mean over rows.
 */
public final class DataParallel implements AutoCloseable {

    /** The loss of a batch given the parameters, the same function on every device. */
    @FunctionalInterface
    public interface Loss {

        /**
         * @param parameters the parameters on this device, in the order given
         * @param inputs this device's rows of the inputs
         * @param targets this device's rows of the targets
         * @return the mean loss over the rows, a scalar
         */
        Var loss(List<Var> parameters, Tensor inputs, Tensor targets);
    }

    private final List<Device> devices;
    private final List<Var> parameters;
    private final List<List<Var>> replicas = new ArrayList<>();
    private final Loss loss;
    private final Optimizer optimizer;
    private final ExecutorService pool;

    /**
     * @param devices where the copies run, at least one; a device may repeat
     * @param parameters the parameters, updated in place each step
     * @param loss computes a share's loss
     * @param optimizer updates the parameters
     */
    public DataParallel(List<Device> devices, List<Var> parameters, Loss loss, Optimizer optimizer) {
        if (devices.isEmpty()) {
            throw new IllegalArgumentException("no devices");
        }
        this.devices = List.copyOf(devices);
        this.parameters = List.copyOf(parameters);
        this.loss = Objects.requireNonNull(loss, "loss");
        this.optimizer = Objects.requireNonNull(optimizer, "optimizer");
        for (Device d : this.devices) {
            List<Var> copy = new ArrayList<>();
            for (Var p : this.parameters) {
                copy.add(Var.param(p.value().to(d)));
            }
            replicas.add(copy);
        }
        pool = Executors.newFixedThreadPool(this.devices.size(), r -> {
            Thread t = new Thread(r, "brainiac-data-parallel");
            t.setDaemon(true);
            return t;
        });
    }

    /**
     * One step on a batch.
     *
     * @param inputs {@code [batch, ...]}
     * @param targets {@code [batch, ...]}
     * @return the batch's mean loss
     */
    public float step(Tensor inputs, Tensor targets) {
        int rows = inputs.shape().dim(0);
        int n = Math.min(devices.size(), rows);
        List<Future<float[]>> work = new ArrayList<>();
        List<List<Tensor>> grads = new ArrayList<>();
        int[] counts = new int[n];
        for (int i = 0; i < n; i++) {
            int from = (int) ((long) rows * i / n);
            int to = (int) ((long) rows * (i + 1) / n);
            counts[i] = to - from;
            List<Var> replica = replicas.get(i);
            Device d = devices.get(i);
            List<Tensor> mine = new ArrayList<>();
            grads.add(mine);
            Tensor x = inputs.slice(0, from, to);
            Tensor y = targets.slice(0, from, to);
            work.add(pool.submit(() -> {
                for (int k = 0; k < replica.size(); k++) {
                    replica.get(k).assign(parameters.get(k).value().to(d));
                    replica.get(k).zeroGrad();
                }
                Var l = loss.loss(replica, x.to(d), y.to(d));
                l.backward();
                for (Var p : replica) {
                    mine.add(p.grad() == null ? null : p.grad().to(DType.FLOAT32).to(Device.cpu()));
                }
                float value = l.value().item();
                l.release();
                return new float[] {value};
            }));
        }
        double total = 0;
        try {
            for (int i = 0; i < n; i++) {
                total += work.get(i).get()[0] * counts[i];
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted", e);
        } catch (ExecutionException e) {
            throw e.getCause() instanceof RuntimeException r ? r : new IllegalStateException(e.getCause());
        }
        for (int k = 0; k < parameters.size(); k++) {
            Tensor sum = null;
            for (int i = 0; i < n; i++) {
                Tensor g = grads.get(i).get(k);
                if (g != null) {
                    Tensor w = g.mul((float) counts[i] / rows);
                    sum = sum == null ? w : sum.add(w);
                }
            }
            Var p = parameters.get(k);
            p.zeroGrad();
            if (sum != null) {
                p.setGrad(sum.to(p.value().device()).to(p.value().dtype()));
            }
        }
        optimizer.step(parameters);
        return (float) (total / rows);
    }

    /** @return the devices the copies run on */
    public List<Device> devices() {
        return devices;
    }

    @Override
    public void close() {
        pool.shutdownNow();
    }
}
