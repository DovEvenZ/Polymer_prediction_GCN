import os
import subprocess

def calc_charge(fchk_file: str):
    fchk_dir = os.path.dirname(fchk_file)
    file_name = os.path.basename(fchk_file)
    with os.popen(f'cd {fchk_dir} && Multiwfn {file_name}', 'w') as p:
        p.write('\n')
        p.write('7\n')
        p.write('11\n')
        p.write('1\n')
        p.write('y\n')

def calc_dispersion(fchk_file: str):
    fchk_dir = os.path.dirname(fchk_file)
    file_name = os.path.basename(fchk_file)
    cmd = f'cd {fchk_dir} && Multiwfn {file_name}'
    with subprocess.Popen(
        cmd,
        shell=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
        ) as p:
        p.stdin.write(b'21\n4\n1\ny\n')
        p.stdin.flush()
        p.wait()
