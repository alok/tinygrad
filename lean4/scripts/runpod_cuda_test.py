#!/usr/bin/env python3
"""
RunPod CUDA Test for TinyGrad4 Lean

Creates a pod, installs Lean 4, clones the repo, builds with CUDA, and runs tests.

Usage:
    export RUNPOD_API_KEY=your_key_here
    python runpod_cuda_test.py [--deploy | --test-only]

Options:
    --deploy      Create new pod and run tests
    --test-only   Print SSH commands for manual testing
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error

RUNPOD_API = "https://api.runpod.io/graphql"

# Setup script for installing Lean 4 and building the project
SETUP_SCRIPT = '''#!/bin/bash
set -e

echo "=== TinyGrad4 Lean CUDA Setup ==="
echo ""

# Install elan (Lean version manager)
if ! command -v elan &> /dev/null; then
    echo "Installing elan..."
    curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | bash -s -- -y
    source ~/.elan/env
fi
export PATH="$HOME/.elan/bin:$PATH"

# Check versions
echo "Lean version:"
lean --version || echo "lean not in path yet"
echo ""

# Clone repo (or update if exists)
cd ~
if [ -d "tinygrad" ]; then
    echo "Updating tinygrad repo..."
    cd tinygrad
    git fetch origin
    git checkout alok-103-conv1d || git checkout origin/alok-103-conv1d
    git pull origin alok-103-conv1d || true
else
    echo "Cloning tinygrad repo..."
    git clone https://github.com/alokbeniwal/tinygrad.git
    cd tinygrad
    git checkout alok-103-conv1d
fi

cd lean4
echo ""
echo "Building with CUDA support..."
echo ""

# Check CUDA
echo "CUDA version:"
nvcc --version
nvidia-smi --query-gpu=name --format=csv,noheader

# Build the project (CUDA will be auto-detected)
echo ""
echo "Running lake build..."
lake update
lake build cuda_smoke 2>&1 | tail -30

echo ""
echo "=== Running CUDA Smoke Test ==="
echo ""
lake exe cuda_smoke
'''


def graphql_request(query: str, variables: dict = None) -> dict:
    """Make a GraphQL request to RunPod API."""
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY environment variable not set")
        sys.exit(1)

    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        RUNPOD_API,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        print(f"HTTP Error: {e.code}")
        print(e.read().decode())
        sys.exit(1)


def get_myself():
    """Get current user info to verify API key."""
    query = """
    query {
        myself {
            id
            email
        }
    }
    """
    return graphql_request(query)


def list_pods():
    """List all pods."""
    query = """
    query {
        myself {
            pods {
                id
                name
                desiredStatus
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        privatePort
                        publicPort
                    }
                }
                machine {
                    gpuDisplayName
                }
            }
        }
    }
    """
    return graphql_request(query)


def create_pod(name: str = "lean4-cuda-test", gpu_type: str = "NVIDIA RTX 4090"):
    """Create a new pod with CUDA support."""
    query = """
    mutation createPod($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
            id
            name
            desiredStatus
            imageName
            machine {
                gpuDisplayName
            }
        }
    }
    """
    variables = {
        "input": {
            "name": name,
            # Use CUDA dev image for nvcc
            "imageName": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
            "gpuTypeId": gpu_type,
            "volumeInGb": 10,
            "containerDiskInGb": 20,
            "minVcpuCount": 2,
            "minMemoryInGb": 8,
            "cloudType": "COMMUNITY",  # More availability
            "startSsh": True
        }
    }
    return graphql_request(query, variables)


def stop_pod(pod_id: str):
    """Stop a pod."""
    query = """
    mutation stopPod($input: PodStopInput!) {
        podStop(input: $input) {
            id
            desiredStatus
        }
    }
    """
    variables = {"input": {"podId": pod_id}}
    return graphql_request(query, variables)


def terminate_pod(pod_id: str):
    """Terminate (delete) a pod."""
    query = """
    mutation terminatePod($input: PodTerminateInput!) {
        podTerminate(input: $input)
    }
    """
    variables = {"input": {"podId": pod_id}}
    return graphql_request(query, variables)


def print_test_commands():
    """Print manual test commands."""
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║           TinyGrad4 Lean CUDA Test - Manual Commands                 ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print()
    print("SSH into your RunPod instance and run:")
    print()
    print("# Quick setup and test")
    print("bash << 'SETUP'")
    print(SETUP_SCRIPT)
    print("SETUP")
    print()
    print("# Or step by step:")
    print()
    print("# 1. Install elan (Lean 4)")
    print("curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh | bash -s -- -y")
    print("source ~/.elan/env")
    print()
    print("# 2. Clone and build")
    print("git clone https://github.com/alokbeniwal/tinygrad.git")
    print("cd tinygrad && git checkout alok-103-conv1d")
    print("cd lean4")
    print("lake update && lake build cuda_smoke")
    print()
    print("# 3. Run CUDA test")
    print("lake exe cuda_smoke")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RunPod CUDA Test for TinyGrad4 Lean")
    parser.add_argument("--deploy", action="store_true", help="Create new pod")
    parser.add_argument("--test-only", action="store_true", help="Print SSH commands")
    parser.add_argument("--list", action="store_true", help="List pods")
    parser.add_argument("--stop", type=str, help="Stop a pod by ID")
    parser.add_argument("--terminate", type=str, help="Terminate a pod by ID")
    parser.add_argument("--gpu", type=str, default="NVIDIA RTX 4090", help="GPU type")
    args = parser.parse_args()

    if args.test_only:
        print_test_commands()
        return

    # Verify API key
    print("Verifying RunPod API key...")
    try:
        result = get_myself()
        if "errors" in result:
            print(f"API Error: {result['errors']}")
            sys.exit(1)
        user = result.get("data", {}).get("myself", {})
        print(f"Logged in as: {user.get('email', 'unknown')}")
    except Exception as e:
        print(f"Failed to verify API key: {e}")
        sys.exit(1)
    print()

    if args.stop:
        print(f"Stopping pod {args.stop}...")
        result = stop_pod(args.stop)
        print(result)
        return

    if args.terminate:
        print(f"Terminating pod {args.terminate}...")
        result = terminate_pod(args.terminate)
        print(result)
        return

    # List pods
    pods = list_pods()
    pod_list = pods.get("data", {}).get("myself", {}).get("pods", [])

    if args.list or not args.deploy:
        if pod_list:
            print("Active pods:")
            for pod in pod_list:
                status = pod.get("desiredStatus", "unknown")
                gpu = pod.get("machine", {}).get("gpuDisplayName", "unknown")
                uptime = pod.get("runtime", {}).get("uptimeInSeconds", 0)
                ports = pod.get("runtime", {}).get("ports", [])
                ssh_info = ""
                for p in ports:
                    if p.get("privatePort") == 22:
                        ssh_info = f" ssh root@{p.get('ip')} -p {p.get('publicPort')}"
                print(f"  [{pod['id']}] {pod['name']}: {status}, {gpu}, uptime: {uptime//60}m{ssh_info}")
        else:
            print("No pods found.")
        print()

    if args.deploy:
        print(f"Creating new pod with {args.gpu}...")
        result = create_pod(gpu_type=args.gpu)
        if "errors" in result:
            print(f"Error: {result['errors']}")
            sys.exit(1)
        pod = result.get("data", {}).get("podFindAndDeployOnDemand", {})
        print(f"Created pod: {pod.get('name')} ({pod.get('id')})")
        print(f"Status: {pod.get('desiredStatus')}")
        print(f"GPU: {pod.get('machine', {}).get('gpuDisplayName')}")
        print()
        print("Wait for pod to start, then SSH in and run the setup script.")
        print()
        print_test_commands()

    if not args.list and not args.deploy and not args.stop and not args.terminate:
        print()
        print_test_commands()


if __name__ == "__main__":
    main()
