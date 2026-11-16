package io.github.zzuegg.brainiac.race;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.AffineTransform;
import java.awt.geom.Line2D;
import java.awt.geom.Path2D;
import java.awt.geom.RoundRectangle2D;
import java.awt.image.BufferedImage;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.List;
import java.util.Locale;

/**
 * Draws a race: the track, the cars, skid marks where they slide, slipstreams, the
 * sensors of one selected car, and a leaderboard. Works on screen and off screen.
 */
public final class RaceRenderer {

    private static final Color GRASS = new Color(0x1d2b22);
    private static final Color ASPHALT = new Color(0x3a3f45);
    private static final Color WALL = new Color(0xd9dde2);
    private static final Color STRIPE = new Color(0xffffff);
    private static final Color INK = new Color(0xf2f2f2);
    private static final Color MUTED = new Color(0x9aa3ad);
    private static final Color PANEL = new Color(20, 24, 28, 215);
    private static final Color SKID = new Color(15, 15, 15, 120);
    private static final Color WIND = new Color(140, 200, 255);
    private static final Color[] PALETTE = {
        new Color(0xe6194b), new Color(0x3cb44b), new Color(0xffe119), new Color(0x4363d8),
        new Color(0xf58231), new Color(0x911eb4), new Color(0x46f0f0), new Color(0xf032e6),
        new Color(0xbcf60c), new Color(0xfabebe), new Color(0x008080), new Color(0xe6beff),
        new Color(0x9a6324), new Color(0xfffac8), new Color(0x800000), new Color(0xaaffc3),
    };

    private final Race race;
    private final Track track;
    private final Path2D.Double left;
    private final Path2D.Double right;
    private final Path2D.Double center;
    private final Deque<double[]> skids = new ArrayDeque<>();
    private int lastSkidStep = -1;
    private int selected;
    private String status = "";

    /**
     * @param race the race to draw
     */
    public RaceRenderer(Race race) {
        this.race = race;
        this.track = race.track();
        left = polyline(track.leftWall());
        right = polyline(track.rightWall());
        List<double[]> c = new ArrayList<>();
        for (int i = 0; i < track.samples(); i++) {
            c.add(track.center(i));
        }
        center = polyline(c);
    }

    /**
     * @param car the car whose sensors to show, or -1 for none
     */
    public void select(int car) {
        selected = car;
    }

    /** @return the car whose sensors are shown, or -1 */
    public int selected() {
        return selected;
    }

    /**
     * @param text a line shown under the leaderboard, such as training progress
     */
    public void status(String text) {
        status = text == null ? "" : text;
    }

    /** @return the colour of a car */
    static Color color(int id) {
        Color base = PALETTE[id % PALETTE.length];
        if (id < PALETTE.length) {
            return base;
        }
        return id / PALETTE.length % 2 == 1 ? base.darker() : base.brighter();
    }

    /**
     * Draws the latest state into an image, for example to save a picture of the race.
     *
     * @param width image width
     * @param height image height
     * @return the image
     */
    public BufferedImage image(int width, int height) {
        BufferedImage img = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        paint(g, width, height);
        g.dispose();
        return img;
    }

    /**
     * Draws the latest state.
     *
     * @param g where to draw
     * @param width the area's width
     * @param height the area's height
     */
    public void paint(Graphics2D g, int width, int height) {
        RaceView view = race.view();
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g.setColor(GRASS);
        g.fillRect(0, 0, width, height);
        if (view == null) {
            return;
        }
        recordSkids(view);
        double panel = Math.min(260, width * 0.3);
        double scale = Math.min((width - panel - 20) / Track.WORLD_WIDTH, (height - 20) / Track.WORLD_HEIGHT);
        AffineTransform screen = g.getTransform();
        g.translate(10, (height - Track.WORLD_HEIGHT * scale) / 2);
        g.scale(scale, scale);

        g.setColor(ASPHALT);
        g.setStroke(new BasicStroke((float) (2 * track.halfWidth()), BasicStroke.CAP_BUTT, BasicStroke.JOIN_ROUND));
        g.draw(center);
        g.setColor(SKID);
        g.setStroke(new BasicStroke(2.2f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
        for (double[] s : skids) {
            g.draw(new Line2D.Double(s[0], s[1], s[2], s[3]));
        }
        g.setColor(WALL);
        g.setStroke(new BasicStroke(3f));
        g.draw(left);
        g.draw(right);
        g.setColor(new Color(255, 255, 255, 70));
        g.setStroke(new BasicStroke(1.2f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_ROUND, 1, new float[] {10, 12}, 0));
        g.draw(center);
        double[] l0 = track.left(0);
        double[] r0 = track.right(0);
        g.setColor(STRIPE);
        g.setStroke(new BasicStroke(5f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_ROUND, 1, new float[] {5, 5}, 0));
        g.draw(new Line2D.Double(l0[0], l0[1], r0[0], r0[1]));

        RaceConfig c = race.config();
        // slipstreams: a fading wedge behind every car that is towing another one
        for (RaceView.CarView car : view.cars()) {
            if (car.draft() <= 0) {
                continue;
            }
            float alpha = (float) Math.min(0.5, 0.15 + car.draft());
            g.setColor(new Color(WIND.getRed() / 255f, WIND.getGreen() / 255f, WIND.getBlue() / 255f, alpha));
            g.setStroke(new BasicStroke(1.5f));
            double hx = Math.cos(car.heading());
            double hy = Math.sin(car.heading());
            for (int k = -1; k <= 1; k++) {
                double sx = car.x() + hx * c.carLength() * 0.6 - hy * k * 3;
                double sy = car.y() + hy * c.carLength() * 0.6 + hx * k * 3;
                g.draw(new Line2D.Double(sx, sy, sx + hx * 14, sy + hy * 14));
            }
        }
        RaceView.CarView sel = selected >= 0 && selected < view.cars().size() ? view.cars().get(selected) : null;
        if (sel != null && !sel.crashed()) {
            drawSensors(g, sel, view, c);
        }
        for (RaceView.CarView car : view.cars()) {
            drawCar(g, car, c, car == sel);
        }
        g.setTransform(screen);
        drawBoard(g, view, width, height, panel);
    }

    private void drawSensors(Graphics2D g, RaceView.CarView car, RaceView view, RaceConfig c) {
        double[] angles = race.rayAngles();
        g.setStroke(new BasicStroke(1f));
        for (int k = 0; k < angles.length; k++) {
            double a = car.heading() + angles[k];
            double d = car.rays()[k];
            double ex = car.x() + Math.cos(a) * d;
            double ey = car.y() + Math.sin(a) * d;
            g.setColor(new Color(255, 200, 80, 110));
            g.draw(new Line2D.Double(car.x(), car.y(), ex, ey));
            if (d < c.rayLength()) {
                g.setColor(new Color(255, 200, 80));
                g.fill(new java.awt.geom.Ellipse2D.Double(ex - 2.5, ey - 2.5, 5, 5));
            }
        }
        // the cars this car sees, nearest first, with their velocity
        List<RaceView.CarView> others = new ArrayList<>();
        for (RaceView.CarView o : view.cars()) {
            if (o != car && !o.crashed() && Math.hypot(o.x() - car.x(), o.y() - car.y()) <= c.senseRadius()) {
                others.add(o);
            }
        }
        others.sort(Comparator.comparingDouble(o -> Math.hypot(o.x() - car.x(), o.y() - car.y())));
        g.setColor(new Color(120, 220, 255, 150));
        for (int i = 0; i < Math.min(c.nearestCars(), others.size()); i++) {
            RaceView.CarView o = others.get(i);
            g.setStroke(new BasicStroke(1f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_ROUND, 1, new float[] {4, 4}, 0));
            g.draw(new Line2D.Double(car.x(), car.y(), o.x(), o.y()));
            g.setStroke(new BasicStroke(1.5f));
            double vx = Math.cos(o.heading()) * o.speed() * 0.12;
            double vy = Math.sin(o.heading()) * o.speed() * 0.12;
            g.draw(new Line2D.Double(o.x(), o.y(), o.x() + vx, o.y() + vy));
        }
        g.setColor(new Color(120, 220, 255, 40));
        double r = c.senseRadius();
        g.draw(new java.awt.geom.Ellipse2D.Double(car.x() - r, car.y() - r, 2 * r, 2 * r));
    }

    private void drawCar(Graphics2D g, RaceView.CarView car, RaceConfig c, boolean selectedCar) {
        AffineTransform t = g.getTransform();
        g.translate(car.x(), car.y());
        g.rotate(car.heading());
        double l = c.carLength();
        double w = c.carWidth();
        Color base = car.crashed() ? Color.GRAY : color(car.id());
        float alpha = car.crashed() ? 0.35f : car.ghost() ? 0.45f : 1f;
        // front wheels show the steering angle
        g.setColor(new Color(10, 10, 10, (int) (alpha * 255)));
        for (int side = -1; side <= 1; side += 2) {
            AffineTransform w0 = g.getTransform();
            g.translate(l * 0.32, side * w * 0.5);
            g.rotate(car.steer());
            g.fill(new java.awt.geom.Rectangle2D.Double(-2.5, -1.2, 5, 2.4));
            g.setTransform(w0);
            g.fill(new java.awt.geom.Rectangle2D.Double(-l * 0.32 - 2.5, side * w * 0.5 - 1.2, 5, 2.4));
        }
        g.setColor(new Color(base.getRed(), base.getGreen(), base.getBlue(), (int) (alpha * 255)));
        RoundRectangle2D body = new RoundRectangle2D.Double(-l / 2, -w / 2, l, w, 4, 4);
        g.fill(body);
        g.setColor(new Color(20, 20, 20, (int) (alpha * 200)));
        g.fill(new java.awt.geom.Rectangle2D.Double(l * 0.05, -w * 0.32, l * 0.18, w * 0.64));
        if (selectedCar) {
            g.setColor(INK);
            g.setStroke(new BasicStroke(1.6f));
            g.draw(body);
        } else if (car.draft() > 0) {
            g.setColor(WIND);
            g.setStroke(new BasicStroke(1.2f));
            g.draw(body);
        }
        g.setTransform(t);
    }

    private void drawBoard(Graphics2D g, RaceView view, int width, int height, double panel) {
        int x = (int) (width - panel);
        g.setColor(PANEL);
        g.fillRect(x, 0, (int) panel, height);
        g.setFont(new Font(Font.SANS_SERIF, Font.BOLD, 15));
        g.setColor(INK);
        g.drawString("Race " + view.race() + "  " + String.format(Locale.ROOT, "%.1f s", view.time()), x + 12, 24);
        g.setFont(new Font(Font.MONOSPACED, Font.PLAIN, 12));
        g.setColor(MUTED);
        g.drawString(track.name() + ", " + view.cars().size() + " cars", x + 12, 42);
        g.drawString(" P  car lap speed  pass crash", x + 8, 64);
        List<RaceView.CarView> order = new ArrayList<>(view.cars());
        order.sort(Comparator.comparingInt(RaceView.CarView::position));
        int y = 82;
        for (RaceView.CarView car : order) {
            if (y > height - 70) {
                break;
            }
            g.setColor(color(car.id()));
            g.fillRect(x + 32, y - 9, 10, 10);
            g.setColor(car.crashed() ? MUTED : car.id() == selected ? Color.WHITE : INK);
            String line = String.format(Locale.ROOT, "%2d %3d %3d %5.0f %5d %5d%s", car.position(), car.id(),
                    Math.max(1, car.lap()), car.speed(), car.overtakes(), car.crashes(),
                    car.draft() > 0 ? " ~" : car.sliding() ? " *" : "");
            g.drawString(line.substring(0, 3), x + 8, y);
            g.drawString(line.substring(3), x + 44, y);
            y += 16;
        }
        g.setColor(MUTED);
        g.drawString("~ in a slipstream   * sliding", x + 8, y + 6);
        int sy = height - 52;
        for (String s : status.split("\n")) {
            g.setColor(INK);
            g.drawString(s, x + 8, sy - 16);
            sy += 14;
        }
        g.setColor(MUTED);
        g.drawString("space pause  r real time/fast", x + 8, height - 24);
        g.drawString("click a car / tab: sensors", x + 8, height - 10);
    }

    private void recordSkids(RaceView view) {
        if (view.step() == lastSkidStep) {
            return;
        }
        if (view.step() < lastSkidStep) {
            skids.clear();
        }
        lastSkidStep = view.step();
        for (RaceView.CarView car : view.cars()) {
            if (!car.sliding() || car.crashed()) {
                continue;
            }
            double hx = Math.cos(car.heading());
            double hy = Math.sin(car.heading());
            double bx = car.x() - hx * 6;
            double by = car.y() - hy * 6;
            skids.addLast(new double[] {bx - hy * 3, by + hx * 3, bx - hy * 3 - hx * 4, by + hx * 3 - hy * 4});
            skids.addLast(new double[] {bx + hy * 3, by - hx * 3, bx + hy * 3 - hx * 4, by - hx * 3 - hy * 4});
        }
        while (skids.size() > 6000) {
            skids.removeFirst();
        }
    }

    /**
     * @param wx a screen point
     * @param wy a screen point
     * @param width the drawing area's width
     * @param height the drawing area's height
     * @return the car nearest to that point, or -1
     */
    public int carAt(int wx, int wy, int width, int height) {
        RaceView view = race.view();
        if (view == null) {
            return -1;
        }
        double panel = Math.min(260, width * 0.3);
        double scale = Math.min((width - panel - 20) / Track.WORLD_WIDTH, (height - 20) / Track.WORLD_HEIGHT);
        double x = (wx - 10) / scale;
        double y = (wy - (height - Track.WORLD_HEIGHT * scale) / 2) / scale;
        int best = -1;
        double bd = 30;
        for (RaceView.CarView car : view.cars()) {
            double d = Math.hypot(car.x() - x, car.y() - y);
            if (d < bd) {
                bd = d;
                best = car.id();
            }
        }
        return best;
    }

    private static Path2D.Double polyline(List<double[]> pts) {
        Path2D.Double p = new Path2D.Double();
        p.moveTo(pts.getFirst()[0], pts.getFirst()[1]);
        for (int i = 1; i < pts.size(); i++) {
            p.lineTo(pts.get(i)[0], pts.get(i)[1]);
        }
        p.closePath();
        return p;
    }
}
