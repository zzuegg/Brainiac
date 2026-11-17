package io.github.zzuegg.brainiac.io;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.UnaryOperator;

/** The {@link WeightSource} views: joined shards and renamed names. */
final class Sources {

    private Sources() {
    }

    /** A source that forwards each name to the part, and name in it, that holds it. */
    private record Routed(Map<String, WeightSource> owner, Map<String, String> inner, List<WeightSource> parts)
            implements WeightSource {

        @Override
        public List<String> names() {
            return Collections.unmodifiableList(new ArrayList<>(owner.keySet()));
        }

        @Override
        public boolean contains(String name) {
            return owner.containsKey(name);
        }

        private WeightSource part(String name) {
            WeightSource p = owner.get(name);
            if (p == null) {
                throw new IllegalArgumentException("no tensor named " + name);
            }
            return p;
        }

        @Override
        public Shape shape(String name) {
            return part(name).shape(inner.get(name));
        }

        @Override
        public String format(String name) {
            return part(name).format(inner.get(name));
        }

        @Override
        public DType naturalType(String name) {
            return part(name).naturalType(inner.get(name));
        }

        @Override
        public Tensor tensor(String name, Device device, DType dtype) {
            return part(name).tensor(inner.get(name), device, dtype);
        }

        @Override
        public void close() {
            for (WeightSource p : parts) {
                p.close();
            }
        }
    }

    static WeightSource joined(List<WeightSource> parts) {
        Map<String, WeightSource> owner = new LinkedHashMap<>();
        Map<String, String> inner = new LinkedHashMap<>();
        for (WeightSource p : parts) {
            for (String n : p.names()) {
                if (owner.put(n, p) != null) {
                    throw new IllegalArgumentException("tensor " + n + " is in more than one file");
                }
                inner.put(n, n);
            }
        }
        return new Routed(owner, inner, parts);
    }

    static WeightSource renamed(WeightSource source, UnaryOperator<String> rename) {
        Map<String, WeightSource> owner = new LinkedHashMap<>();
        Map<String, String> inner = new LinkedHashMap<>();
        for (String n : source.names()) {
            String to = rename.apply(n);
            if (to == null) {
                continue;
            }
            if (owner.put(to, source) != null) {
                throw new IllegalArgumentException("two tensors are renamed to " + to + ", the second " + n);
            }
            inner.put(to, n);
        }
        return new Routed(owner, inner, List.of(source));
    }
}
