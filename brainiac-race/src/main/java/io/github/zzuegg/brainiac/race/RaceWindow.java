package io.github.zzuegg.brainiac.race;

import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GraphicsEnvironment;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicReference;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.Timer;

/**
 * A desktop window that shows a race as it runs. The race can be stepped by any thread,
 * for example a trainer: the window repaints about 60 times a second and, in real-time
 * mode, slows the stepping thread down to the race's own clock.
 *
 * <p>Keys: space pauses, {@code r} switches between real time and full speed, tab picks
 * the next car whose sensors are shown (or click a car).
 */
public final class RaceWindow {

    private final Race race;
    private final RaceRenderer renderer;
    private final JFrame frame;
    private volatile boolean realTime;
    private volatile boolean paused;
    private volatile boolean closed;
    private long nextStepNanos;
    private final long periodNanos;

    private RaceWindow(Race race, String title, boolean realTime) {
        this.race = race;
        this.renderer = new RaceRenderer(race);
        this.realTime = realTime;
        this.periodNanos = (long) (race.config().dt() * 1e9);
        JPanel panel = new JPanel() {
            private static final long serialVersionUID = 1L;

            @Override
            protected void paintComponent(Graphics g) {
                renderer.paint((Graphics2D) g, getWidth(), getHeight());
            }
        };
        panel.setPreferredSize(new Dimension(1280, 760));
        panel.setFocusable(true);
        panel.setFocusTraversalKeysEnabled(false);
        panel.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                switch (e.getKeyCode()) {
                    case KeyEvent.VK_SPACE -> paused = !paused;
                    case KeyEvent.VK_R -> RaceWindow.this.realTime = !RaceWindow.this.realTime;
                    case KeyEvent.VK_TAB -> renderer.select((renderer.selected() + 1) % race.count());
                    default -> { }
                }
            }
        });
        panel.addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                int car = renderer.carAt(e.getX(), e.getY(), panel.getWidth(), panel.getHeight());
                if (car >= 0) {
                    renderer.select(car);
                }
                panel.requestFocusInWindow();
            }
        });
        frame = new JFrame(title);
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.add(panel);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosed(WindowEvent e) {
                closed = true;
                paused = false;
            }
        });
        new Timer(16, e -> panel.repaint()).start();
        frame.setVisible(true);
        panel.requestFocusInWindow();
        race.onStep(v -> pace());
    }

    /**
     * Opens a window showing a race.
     *
     * @param race the race
     * @param title the window title
     * @param realTime whether to start in real time (otherwise at full speed)
     * @return the window
     * @throws java.awt.HeadlessException if there is no display
     */
    public static RaceWindow open(Race race, String title, boolean realTime) {
        if (GraphicsEnvironment.isHeadless()) {
            throw new java.awt.HeadlessException("no display to open a race window on");
        }
        AtomicReference<RaceWindow> out = new AtomicReference<>();
        try {
            SwingUtilities.invokeAndWait(() -> out.set(new RaceWindow(race, title, realTime)));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        } catch (java.lang.reflect.InvocationTargetException e) {
            throw new IllegalStateException(e.getCause());
        }
        return out.get();
    }

    /**
     * @param text a line or two shown under the leaderboard, such as training progress
     */
    public void status(String text) {
        renderer.status(text);
    }

    /**
     * @param on whether to slow the race down to real time (otherwise it runs at full speed)
     */
    public void realTime(boolean on) {
        realTime = on;
    }

    /** @return whether the window has been closed */
    public boolean closed() {
        return closed;
    }

    /**
     * Drives every car with a driver and shows the race until the window is closed;
     * crashed cars respawn.
     *
     * @param driver drives every car
     * @param seed the random seed for the grids
     */
    public void watch(Driver driver, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        race.reset(random.split());
        while (!closed) {
            float[][] actions = new float[race.count()][];
            for (int i = 0; i < actions.length; i++) {
                actions[i] = driver.drive(race, i);
            }
            var steps = race.step(actions);
            for (int i = 0; i < steps.length; i++) {
                if (steps[i].done()) {
                    race.reset(i, random.split());
                }
            }
        }
    }

    /** Called on the stepping thread after every step: pauses and keeps real time. */
    private void pace() {
        while (paused && !closed) {
            try {
                Thread.sleep(20);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return;
            }
        }
        if (!realTime || closed) {
            nextStepNanos = 0;
            return;
        }
        long now = System.nanoTime();
        long period = periodNanos;
        if (nextStepNanos == 0 || now - nextStepNanos > 200_000_000L) {
            nextStepNanos = now;
        }
        nextStepNanos += period;
        long wait = nextStepNanos - now;
        if (wait > 0) {
            try {
                Thread.sleep(wait / 1_000_000, (int) (wait % 1_000_000));
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
}
