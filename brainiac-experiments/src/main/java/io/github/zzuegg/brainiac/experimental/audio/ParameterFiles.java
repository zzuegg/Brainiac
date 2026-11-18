package io.github.zzuegg.brainiac.experimental.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

/**
 * Checkpoints of parameter values: a magic number, a step counter, then per value its rank,
 * dimensions and floats. Loading checks the count and every shape. Optimiser state is not
 * saved.
 */
final class ParameterFiles {

    private static final int MAGIC = 0x42524e43;

    private ParameterFiles() {
    }

    static void write(Path file, List<Var> values, int steps) {
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(file)))) {
            out.writeInt(MAGIC);
            out.writeInt(steps);
            out.writeInt(values.size());
            for (Var v : values) {
                Shape s = v.value().shape();
                out.writeInt(s.rank());
                for (int d = 0; d < s.rank(); d++) {
                    out.writeInt(s.dim(d));
                }
                for (float f : v.value().toArray()) {
                    out.writeFloat(f);
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** Loads values into {@code values}; returns the saved step counter. */
    static int read(Path file, List<Var> values) {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(file)))) {
            if (in.readInt() != MAGIC) {
                throw new IllegalArgumentException("not a checkpoint: " + file);
            }
            int steps = in.readInt();
            int n = in.readInt();
            if (n != values.size()) {
                throw new IllegalArgumentException("checkpoint has " + n + " values, model " + values.size());
            }
            for (Var v : values) {
                Shape want = v.value().shape();
                int rank = in.readInt();
                int[] dims = new int[rank];
                for (int d = 0; d < rank; d++) {
                    dims[d] = in.readInt();
                }
                Shape got = Shape.of(dims);
                if (!got.equals(want)) {
                    throw new IllegalArgumentException("checkpoint shape " + got + ", model " + want);
                }
                float[] data = new float[Math.toIntExact(got.size())];
                for (int i = 0; i < data.length; i++) {
                    data[i] = in.readFloat();
                }
                v.assign(Tensor.of(got, data));
            }
            return steps;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
