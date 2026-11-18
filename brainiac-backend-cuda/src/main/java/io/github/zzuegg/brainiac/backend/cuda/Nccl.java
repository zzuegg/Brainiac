package io.github.zzuegg.brainiac.backend.cuda;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

import io.github.zzuegg.brainiac.tensor.DType;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;
import java.util.List;
import java.util.regex.Pattern;

/**
 * Bindings to NCCL, NVIDIA's library for collective operations between GPUs (over NVLink or
 * PCIe within a machine). Linux only; {@code -Dbrainiac.cuda.nccl} names the file. Used for
 * the all-reduce of data-parallel training across the GPUs of one process.
 */
@SuppressWarnings("restricted")
final class Nccl {

    private static final int NCCL_SUCCESS = 0;
    private static final int NCCL_SUM = 0;

    final String library;

    private final MethodHandle ncclGetVersion;
    private final MethodHandle ncclCommInitAll;
    private final MethodHandle ncclCommDestroy;
    private final MethodHandle ncclAllReduce;
    private final MethodHandle ncclBroadcast;
    private final MethodHandle ncclGroupStart;
    private final MethodHandle ncclGroupEnd;
    private final MethodHandle ncclGetErrorString;

    Nccl() {
        if (Binder.WINDOWS) {
            throw new CudaException("NCCL is not available on Windows", 0);
        }
        Binder b = Binder.load("NCCL", "brainiac.cuda.nccl", List.of("libnccl.so.2", "libnccl.so"),
                Pattern.compile("libnccl\\.so\\.([0-9.]+)"), null);
        library = b.library();
        ncclGetVersion = b.bind("ncclGetVersion", FunctionDescriptor.of(JAVA_INT, ADDRESS), false);
        ncclCommInitAll = b.bind("ncclCommInitAll", FunctionDescriptor.of(JAVA_INT, ADDRESS, JAVA_INT, ADDRESS), false);
        ncclCommDestroy = b.bind("ncclCommDestroy", FunctionDescriptor.of(JAVA_INT, JAVA_LONG), false);
        // send, receive, count, type, operation, communicator, stream
        ncclAllReduce = b.bind("ncclAllReduce", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG,
                JAVA_INT, JAVA_INT, JAVA_LONG, JAVA_LONG), false);
        // send, receive, count, type, root, communicator, stream
        ncclBroadcast = b.bind("ncclBroadcast", FunctionDescriptor.of(JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG,
                JAVA_INT, JAVA_INT, JAVA_LONG, JAVA_LONG), false);
        ncclGroupStart = b.bind("ncclGroupStart", FunctionDescriptor.of(JAVA_INT), false);
        ncclGroupEnd = b.bind("ncclGroupEnd", FunctionDescriptor.of(JAVA_INT), false);
        ncclGetErrorString = b.bind("ncclGetErrorString", FunctionDescriptor.of(ADDRESS, JAVA_INT), false);
    }

    private void check(int result, String call) {
        if (result != NCCL_SUCCESS) {
            String text;
            try {
                MemorySegment s = (MemorySegment) ncclGetErrorString.invokeExact(result);
                text = s.equals(MemorySegment.NULL) ? "NCCL error" : s.reinterpret(Long.MAX_VALUE).getString(0);
            } catch (Throwable t) {
                text = "NCCL error";
            }
            throw new CudaException(call + ": " + text + " (" + result + ")", result);
        }
    }

    /** For example "2.21.5". */
    String version() {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment v = a.allocate(JAVA_INT);
            check((int) ncclGetVersion.invokeExact(v), "ncclGetVersion");
            int code = v.get(JAVA_INT, 0);
            // from 2.9: major * 10000 + minor * 100 + patch; before: major * 1000 + minor * 100 + patch
            return code >= 10000 ? code / 10000 + "." + code / 100 % 100 + "." + code % 100
                    : code / 1000 + "." + code / 100 % 10 + "." + code % 100;
        } catch (Throwable t) {
            throw Binder.fail("ncclGetVersion", t);
        }
    }

    /**
     * Communicators for GPUs of this process, one per device in the order given.
     *
     * @param ordinals the devices' CUDA indices, all different
     */
    long[] communicators(int[] ordinals) {
        try (Arena a = Arena.ofConfined()) {
            MemorySegment comms = a.allocate(JAVA_LONG, ordinals.length);
            MemorySegment list = a.allocateFrom(JAVA_INT, ordinals);
            check((int) ncclCommInitAll.invokeExact(comms, ordinals.length, list), "ncclCommInitAll");
            return comms.toArray(JAVA_LONG);
        } catch (Throwable t) {
            throw Binder.fail("ncclCommInitAll", t);
        } finally {
            CudaContext.forgetCurrent();
        }
    }

    void destroy(long comm) {
        try {
            check((int) ncclCommDestroy.invokeExact(comm), "ncclCommDestroy");
        } catch (Throwable t) {
            throw Binder.fail("ncclCommDestroy", t);
        }
    }

    static int type(DType dtype) {
        return switch (dtype) {
            case FLOAT16 -> 6;
            case FLOAT32 -> 7;
            case FLOAT64 -> 8;
            case BFLOAT16 -> 9;
            case INT32 -> 2;
            case INT64 -> 4;
            case UINT8, BOOL -> 1;
        };
    }

    void groupStart() {
        try {
            check((int) ncclGroupStart.invokeExact(), "ncclGroupStart");
        } catch (Throwable t) {
            throw Binder.fail("ncclGroupStart", t);
        }
    }

    void groupEnd() {
        try {
            check((int) ncclGroupEnd.invokeExact(), "ncclGroupEnd");
        } catch (Throwable t) {
            throw Binder.fail("ncclGroupEnd", t);
        } finally {
            CudaContext.forgetCurrent();
        }
    }

    /** Queues a sum of {@code count} values over every communicator of the group, into {@code recv}. */
    void allReduceSum(long send, long recv, long count, DType dtype, long comm, long stream) {
        try {
            check((int) ncclAllReduce.invokeExact(send, recv, count, type(dtype), NCCL_SUM, comm, stream),
                    "ncclAllReduce(" + count + " " + dtype + ")");
        } catch (Throwable t) {
            throw Binder.fail("ncclAllReduce", t);
        }
    }

    /** Queues a copy of {@code count} values from the root's {@code send} to every device's {@code recv}. */
    void broadcast(long send, long recv, long count, DType dtype, int root, long comm, long stream) {
        try {
            check((int) ncclBroadcast.invokeExact(send, recv, count, type(dtype), root, comm, stream),
                    "ncclBroadcast(" + count + " " + dtype + ")");
        } catch (Throwable t) {
            throw Binder.fail("ncclBroadcast", t);
        }
    }
}
