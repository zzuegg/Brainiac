package io.github.zzuegg.brainiac.studio.chat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

/** Reads a server-sent event stream. */
final class Sse {

    /** Receives one event. */
    @FunctionalInterface
    interface Handler {
        /**
         * Handles an event.
         *
         * @param event the event name, or {@code message} when the stream gives none
         * @param data the event's data lines joined with newlines
         * @return false to stop reading
         * @throws IOException if handling fails
         */
        boolean event(String event, String data) throws IOException;
    }

    private Sse() {
    }

    static void read(InputStream in, Handler handler) throws IOException {
        BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
        String event = "message";
        StringBuilder data = null;
        for (String line; (line = r.readLine()) != null; ) {
            if (line.isEmpty()) {
                if (data != null && !handler.event(event, data.toString())) {
                    return;
                }
                event = "message";
                data = null;
                continue;
            }
            if (line.startsWith(":")) {
                continue;   // comment, often a keep-alive
            }
            int colon = line.indexOf(':');
            String field = colon < 0 ? line : line.substring(0, colon);
            String value = colon < 0 ? "" : line.substring(colon + 1);
            if (value.startsWith(" ")) {
                value = value.substring(1);
            }
            switch (field) {
                case "event" -> event = value;
                case "data" -> data = data == null ? new StringBuilder(value) : data.append('\n').append(value);
                default -> { }
            }
        }
        if (data != null) {
            handler.event(event, data.toString());
        }
    }
}
