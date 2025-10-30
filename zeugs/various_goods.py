import subprocess
from pathlib import Path






def run_command(exe_path : Path, input : str, ncores : int) -> str:
    '''Runs cmd with mpi, returning stdout. BEFROE use incorprate changes done to stream version (regarding signature ex..)'''

    if not exe_path.exists():
        raise FileNotFoundError(f"Could not find executable at {exe_path}.")

    cmd = [
        "mpirun",
        "-np", str(ncores),
        str(exe_path)
    ]

    proc = subprocess.run(
        cmd,                        # the full command line as a list
        input=input,                # send this string to the program's stdin
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,                  # text=True on Py3.7+, universal_newlines=True otherwise
        check=False                 # we'll handle non-zero exit codes ourselves
    )

    output = proc.stdout

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command {cmd!r} failed with exit code {proc.returncode}.\n"
            f"Output was:\n{output}"
        )

    return output






def run_command_stream(run_cmd : list[str], exe_path : Path, input : str|None) -> str:
    '''Runs cmd with mpi, returning stdout. Additionally streams stdout while cmd is running.'''

    if not exe_path.exists():
        raise FileNotFoundError(f"Could not find executable at {exe_path}.")
    
    run_cmd.append(str(exe_path))

    proc = subprocess.Popen(
        run_cmd,                    # the full command line as a list
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,                  # text=True on Py3.7+, universal_newlines=True otherwise
        bufsize=1                   # line buffered
    )

    # Send input (if any) and close stdin so the child sees EOF
    if input is not None:
        proc.stdin.write(input)
    proc.stdin.close()

    out_lines = []
    # Read line-by-line until EOF
    for line in iter(proc.stdout.readline, ''):
        # Print live (no extra newline because 'line' already has it)
        print(line, end='', flush=True)
        out_lines.append(line)

    proc.stdout.close()
    returncode = proc.wait()

    output = ''.join(out_lines)

    if proc.returncode != 0:
        raise RuntimeError(
            f"Command {run_cmd!r} failed with exit code {returncode}.\n"
            f"Output was:\n{output}"
        )

    return output