package io.github.zzuegg.brainiac.nn;

import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Objects;
import java.util.Set;

/**
 * The settings that rebuild a layer or activation when a model is loaded, such as a
 * layer's size or a dropout rate. Learned values (parameters and buffers) are not part
 * of it: they are saved separately and restored after the layer is rebuilt.
 *
 * <p>Values can be booleans, ints, longs, floats, doubles, strings, int and float
 * arrays, nested configs and activations. An activation is stored by reference, so
 * one learnable activation shared by several layers is still shared after loading.
 *
 * <pre>{@code
 * Config c = Config.builder().put("units", 32).put("activation", Activation.RELU).build();
 * int units = c.getInt("units");
 * }</pre>
 */
public final class Config {

    /** A config without entries. */
    public static final Config EMPTY = new Config(Map.of());

    private final Map<String, Object> values;

    private Config(Map<String, Object> values) {
        this.values = values;
    }

    /** @return a builder for a new config */
    public static Builder builder() {
        return new Builder();
    }

    /** @return the keys, in insertion order */
    public Set<String> keys() {
        return values.keySet();
    }

    /**
     * @param key the key
     * @return whether the config has a value for the key
     */
    public boolean has(String key) {
        return values.containsKey(key);
    }

    /**
     * Returns a value of any type.
     *
     * @param key the key
     * @return the value
     * @throws NoSuchElementException if the key is missing
     */
    public Object get(String key) {
        Object v = values.get(key);
        if (v == null) {
            throw new NoSuchElementException("config has no `" + key + "`; keys: " + values.keySet());
        }
        return v;
    }

    private <T> T get(String key, Class<T> type) {
        Object v = get(key);
        if (!type.isInstance(v)) {
            throw new IllegalArgumentException("config value `" + key + "` is a " + v.getClass().getSimpleName()
                    + ", not a " + type.getSimpleName());
        }
        return type.cast(v);
    }

    /**
     * @param key the key
     * @return the boolean value
     */
    public boolean getBoolean(String key) {
        return get(key, Boolean.class);
    }

    /**
     * @param key the key
     * @return the int value
     */
    public int getInt(String key) {
        return get(key, Integer.class);
    }

    /**
     * @param key the key
     * @return the long value
     */
    public long getLong(String key) {
        return get(key, Long.class);
    }

    /**
     * @param key the key
     * @return the float value
     */
    public float getFloat(String key) {
        return get(key, Float.class);
    }

    /**
     * @param key the key
     * @return the double value
     */
    public double getDouble(String key) {
        return get(key, Double.class);
    }

    /**
     * @param key the key
     * @return the string value
     */
    public String getString(String key) {
        return get(key, String.class);
    }

    /**
     * @param key the key
     * @return a copy of the int array
     */
    public int[] getInts(String key) {
        return get(key, int[].class).clone();
    }

    /**
     * @param key the key
     * @return a copy of the float array
     */
    public float[] getFloats(String key) {
        return get(key, float[].class).clone();
    }

    /**
     * @param key the key
     * @return the nested config
     */
    public Config getConfig(String key) {
        return get(key, Config.class);
    }

    /**
     * @param key the key
     * @return the activation
     */
    public Activation getActivation(String key) {
        return get(key, Activation.class);
    }

    Map<String, Object> values() {
        return values;
    }

    @Override
    public boolean equals(Object o) {
        if (!(o instanceof Config other) || !values.keySet().equals(other.values.keySet())) {
            return false;
        }
        for (var e : values.entrySet()) {
            if (!Objects.deepEquals(e.getValue(), other.values.get(e.getKey()))) {
                return false;
            }
        }
        return true;
    }

    @Override
    public int hashCode() {
        return values.keySet().hashCode();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("{");
        for (var e : values.entrySet()) {
            if (sb.length() > 1) {
                sb.append(", ");
            }
            Object v = e.getValue();
            sb.append(e.getKey()).append('=').append(switch (v) {
                case int[] a -> Arrays.toString(a);
                case float[] a -> Arrays.toString(a);
                default -> String.valueOf(v);
            });
        }
        return sb.append('}').toString();
    }

    /** Collects entries for a {@link Config}. */
    public static final class Builder {

        private final Map<String, Object> values = new LinkedHashMap<>();

        private Builder() {
        }

        private Builder set(String key, Object value) {
            Objects.requireNonNull(key, "key");
            Objects.requireNonNull(value, "value");
            if (values.put(key, value) != null) {
                throw new IllegalArgumentException("config key `" + key + "` set twice");
            }
            return this;
        }

        /**
         * @param key the key
         * @param value the value
         * @return this builder
         */
        public Builder put(String key, boolean value) {
            return set(key, value);
        }

        /**
         * @param key the key
         * @param value the value
         * @return this builder
         */
        public Builder put(String key, int value) {
            return set(key, value);
        }

        /**
         * @param key the key
         * @param value the value
         * @return this builder
         */
        public Builder put(String key, long value) {
            return set(key, value);
        }

        /**
         * @param key the key
         * @param value the value
         * @return this builder
         */
        public Builder put(String key, float value) {
            return set(key, value);
        }

        /**
         * @param key the key
         * @param value the value
         * @return this builder
         */
        public Builder put(String key, double value) {
            return set(key, value);
        }

        /**
         * @param key the key
         * @param value the value
         * @return this builder
         */
        public Builder put(String key, String value) {
            return set(key, value);
        }

        /**
         * @param key the key
         * @param value the value, copied
         * @return this builder
         */
        public Builder put(String key, int[] value) {
            return set(key, value.clone());
        }

        /**
         * @param key the key
         * @param value the value, copied
         * @return this builder
         */
        public Builder put(String key, float[] value) {
            return set(key, value.clone());
        }

        /**
         * @param key the key
         * @param value a nested config
         * @return this builder
         */
        public Builder put(String key, Config value) {
            return set(key, value);
        }

        /**
         * Stores an activation by reference. It must be registered (see
         * {@link Activation#register(String, java.util.function.Function)}) for the model
         * to load.
         *
         * @param key the key
         * @param value the activation
         * @return this builder
         */
        public Builder put(String key, Activation value) {
            return set(key, value);
        }

        /** @return the config */
        public Config build() {
            return values.isEmpty() ? EMPTY : new Config(Collections.unmodifiableMap(new LinkedHashMap<>(values)));
        }
    }
}
