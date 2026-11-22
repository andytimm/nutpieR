"""Run from the repository root on POSIX; see tests/README.md."""
import errno
import fcntl
import os
import pty
import re
import struct
import termios

import pyte


WIDTH = 132  # The divergence hint fits on one physical line.


def capture(rows, outcome):
    pid, fd = pty.fork()
    if pid == 0:
        fcntl.ioctl(0, termios.TIOCSWINSZ, struct.pack("HHHH", rows, WIDTH, 0, 0))
        os.environ.pop("LINES", None)  # Exercise the actual PTY height probe.
        os.environ.update(TERM="xterm-256color", COLUMNS=str(WIDTH))
        os.execvp("Rscript", ["Rscript", "--vanilla", "tests/manual/progress-screen.R", outcome, str(rows)])
    raw = bytearray()
    try:
        while chunk := os.read(fd, 65536):
            raw.extend(chunk)
    except OSError as e:
        if e.errno != errno.EIO:  # Linux signals PTY EOF this way.
            raise
    finally:
        os.close(fd)
        _, status = os.waitpid(pid, 0)
    if status:
        raise RuntimeError(raw.decode(errors="replace"))
    return raw.decode().split("\x1e")


def check(rows, outcome):
    screen = pyte.HistoryScreen(WIDTH, rows)
    stream = pyte.Stream(screen)
    before, hinted, finished = capture(rows, outcome)
    prior = [f"prior {i+1}" for i in range(rows)]

    def replay(chunk):
        stream.feed(chunk)
        history = ["".join(row[i].data for i in range(WIDTH)) for row in screen.history.top]
        lines = [s.rstrip() for s in history + screen.display if s.strip()]
        assert lines[:rows] == prior, lines
        return lines[rows:]

    def check_rows(lines, phases, percents):
        assert len(lines) == 5 and lines[0].startswith("Sampling"), lines
        visible = [s.rstrip() for s in screen.display if s.strip()]
        assert all(line in visible for line in lines), visible
        for i, (phase, pct) in enumerate(zip(phases, percents), 1):
            assert re.match(rf"c{i}\s+{phase}\s+{pct}%", lines[i]), lines

    lines = replay(before)
    if rows == 4:
        assert len(lines) == 1 and "warmup" in lines[0], lines
    else:
        check_rows(lines, ["warmup"] * 3 + ["queued"], [40, 30, 20, 0])

    lines = replay(hinted)
    assert "divergent transitions detected" in lines[0], lines
    hint = lines.pop(0)
    if rows == 4:
        assert len(lines) == 1 and "sample" in lines[0], lines
    else:
        check_rows(lines, ["sample"] + ["warmup"] * 3, [80, 45, 40, 5])

    lines = replay(finished)
    assert lines[0] == hint and lines[-1] == "after sampling", lines
    remaining = lines[1:-1]
    if outcome == "success":
        check_rows(remaining, ["done"] * 4, [100] * 4)
    else:
        assert remaining == [], lines
    assert not screen.cursor.hidden, "cursor was not restored"
    print(f"PASS {WIDTH}x{rows} {outcome}")


if __name__ == "__main__":
    check(24, "success")
    check(24, "clear")
    check(4, "clear")
