import paramiko
import time
import base64
import sys

USERNAME = "h3250102096+mypod+hpc101"
KEY_PATH = r"C:\Users\20923\.ssh\id_ed25519"
HOST = "clusters.zju.edu.cn"
PORT = 443


def get_transport():
    transport = paramiko.Transport((HOST, PORT))
    transport.start_client()
    pkey = paramiko.Ed25519Key.from_private_key_file(KEY_PATH)
    transport.auth_publickey(USERNAME, pkey)
    return transport


def run_remote(cmd, wait=15):
    transport = get_transport()
    chan = transport.open_session()
    chan.exec_command(cmd)
    time.sleep(wait)
    out = chan.recv(65536).decode(errors="replace")
    err = chan.recv_stderr(65536).decode(errors="replace")
    transport.close()
    return out, err


def write_script(name, lines):
    script = "\n".join(lines) + "\n"
    b64 = base64.b64encode(script.encode()).decode()
    out, err = run_remote(f"echo {b64} | base64 -d > ~/{name} && chmod +x ~/{name}")
    return out, err


def git_pull():
    out, err = run_remote("cd ~/hpc101-lab3 && git pull 2>&1")
    return out.strip()


def submit_job(script_name, partition="lab3", time_limit="5m"):
    cmd = f"hpc submit -p {partition} -c 8 -g 1 -m 32Gi -t {time_limit} -d -n bench -o bench.out -- ~/{script_name} 2>&1"
    out, err = run_remote(cmd, wait=10)
    # Extract job ID
    for line in out.split("\n"):
        if "submitted job" in line:
            return line.strip()
    return out.strip()


def check_job(job_name="bench"):
    out, err = run_remote("hpc ls -l 5 2>&1")
    return out.strip()


def get_output(filename="bench.out"):
    out, err = run_remote(f"cat ~/{filename} 2>&1")
    return out.strip()


def wait_and_get_output(wait_seconds=120, filename="bench.out"):
    print(f"Waiting {wait_seconds}s...")
    time.sleep(wait_seconds)
    status = check_job()
    print("=== status ===")
    print(status)
    output = get_output(filename)
    print("=== output ===")
    print(output[-5000:])
    return output


def submit_and_wait(script_name="bench.sh", wait_seconds=120):
    git_pull()
    result = submit_job(script_name)
    print(f"Submit: {result}")
    return wait_and_get_output(wait_seconds)


if __name__ == "__main__":
    # Quick test
    out = git_pull()
    print(out)
    result = submit_job("bench.sh")
    print(f"Submit: {result}")
    output = wait_and_get_output(120)
    print(output)
