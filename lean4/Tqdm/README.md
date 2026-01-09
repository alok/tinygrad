# Tqdm (Lean)

Minimal progress bar for Lean projects.

## Usage

```lean
import Tqdm

def main : IO Unit := do
  Tqdm.forInRange 100 (cfg := { desc := "compile" }) fun _ => do
    IO.sleep 10
```

## License

Apache License 2.0. See `LICENSE`.
