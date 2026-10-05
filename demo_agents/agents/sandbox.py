"""File tools use no-follow directory descriptors; shell tools use disposable containers."""

import os
from pathlib import Path
from contextlib import contextmanager


def relative_path(root, path):
    root = Path(root).absolute()
    candidate = Path(path)
    if candidate.is_absolute():
        candidate = candidate.relative_to(root)
    if not candidate.parts or any(p in ("..", "") for p in candidate.parts):
        raise ValueError("Path must stay within the workspace")
    return root, candidate


@contextmanager
def open_workspace(root, path, write=False):
    root, relative = relative_path(root, path)
    root.mkdir(parents=True, exist_ok=True)
    descriptors = []
    try:
        parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(parent)
        for part in relative.parts[:-1]:
            if write:
                try:
                    os.mkdir(part, dir_fd=parent)
                except FileExistsError:
                    pass
            parent = os.open(
                part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
            )
            descriptors.append(parent)
        flags = (
            os.O_NONBLOCK
            | os.O_NOFOLLOW
            | (os.O_WRONLY | os.O_CREAT | os.O_TRUNC if write else os.O_RDONLY)
        )
        fd = os.open(relative.name, flags, 0o600, dir_fd=parent)
        import stat

        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ValueError("Only regular files are allowed")
        with os.fdopen(fd, "w" if write else "r") as stream:
            yield stream
    finally:
        for fd in reversed(descriptors):
            os.close(fd)


def run_container(args, timeout=30, output_limit=12000):
    """Bound output while draining the CLI; never buffer unbounded child output."""
    import selectors
    import subprocess
    import time

    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = bytearray()
    deadline = time.monotonic() + timeout
    reason = ""
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    reason = "Command timed out"
                    break
                if not selector.select(left):
                    reason = "Command timed out"
                    break
                chunk = os.read(process.stdout.fileno(), 4096)
                if not chunk:
                    break
                remaining = output_limit - len(output)
                output.extend(chunk[:remaining])
                if len(chunk) > remaining:
                    reason = "Output limit exceeded"
                    break
        if reason:
            process.kill()
        code = process.wait(timeout=5)
        return f"{output.decode(errors='replace')}\n{reason}\nexit code: {code}"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        process.stdout.close()
