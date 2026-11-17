package io.github.zzuegg.brainiac.studio.server;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.chat.VastAi;
import io.github.zzuegg.brainiac.studio.train.VastRentals;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

/**
 * Renting Vast.ai machines as worker hosts.
 *
 * <ul>
 *   <li>{@code GET /api/vast/rentals}: the machines rented from the studio, with their stage and
 *       what each has cost so far, and why renting cannot work here if it cannot.
 *   <li>{@code POST /api/vast/rent} with {@code {"offer": 123, "pricePerHour": 0.36, "hours": 2}}
 *       rents an offer, after looking up its price again: it is refused when the offer now costs
 *       more than the price the user agreed to. The machine is destroyed after the hours given.
 *   <li>{@code POST /api/vast/destroy} with {@code {"id": 456}} destroys an instance;
 *       {@code POST /api/vast/cancel} with {@code {"key": "r1"}} stops setting up a rental and
 *       destroys its machine.
 * </ul>
 */
final class VastRoutes {

    private final VastAi vast;
    private final VastRentals rentals;

    VastRoutes(VastAi vast, VastRentals rentals) {
        this.vast = vast;
        this.rentals = rentals;
    }

    void register(StudioServer server) {
        server.get("/api/vast/rentals", ex -> Exchanges.json(ex, 200, rentals()));
        server.post("/api/vast/rent", ex -> {
            Map<String, Object> body = Exchanges.body(ex, 8192);
            long id = Json.integer(body, "offer", 0);
            double agreed = Json.number(body, "pricePerHour", 0);
            double hours = Json.number(body, "hours", 0);
            if (id <= 0 || !(agreed > 0)) {
                Exchanges.error(ex, 400, "choose an offer and agree to its price");
                return;
            }
            if (rentals.unavailable().isPresent()) {
                Exchanges.error(ex, 409, rentals.unavailable().get());
                return;
            }
            try {
                Optional<Map<String, Object>> offer = vast.offer(id);
                if (offer.isEmpty()) {
                    Exchanges.error(ex, 409, "that machine is no longer offered; search again");
                    return;
                }
                double now = Json.number(offer.get(), "pricePerHour", 0);
                Map<String, Object> out = new LinkedHashMap<>();
                out.put("rental", rentals.rent(id, Json.string(offer.get(), "gpu", "GPU"), now, agreed, hours));
                out.putAll(rentals());
                Exchanges.json(ex, 200, out);
            } catch (IOException e) {
                Exchanges.error(ex, 400, e.getMessage());
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                Exchanges.error(ex, 500, "interrupted");
            }
        });
        server.post("/api/vast/destroy", ex -> {
            long id = Json.integer(Exchanges.body(ex, 8192), "id", 0);
            if (id <= 0) {
                Exchanges.error(ex, 400, "say which instance to destroy");
                return;
            }
            try {
                rentals.destroy(id);
                Exchanges.json(ex, 200, rentals());
            } catch (IOException e) {
                Exchanges.error(ex, 400, e.getMessage());
            }
        });
        server.post("/api/vast/cancel", ex -> {
            if (!rentals.cancel(Json.string(Exchanges.body(ex, 8192), "key", ""))) {
                Exchanges.error(ex, 404, "no such rental is running");
                return;
            }
            Exchanges.json(ex, 200, rentals());
        });
    }

    private Map<String, Object> rentals() {
        Map<String, Object> out = new LinkedHashMap<>();
        out.put("rentals", rentals.list());
        rentals.unavailable().ifPresent(u -> out.put("unavailable", u));
        out.put("maxHours", VastRentals.MAX_HOURS);
        return out;
    }
}
