/-!
Canonical W&B file names used inside the run {lit}`files` directory.
-/

namespace Wandb.Filenames

def config : String := "config.yaml"
def history : String := "wandb-history.jsonl"
def summary : String := "wandb-summary.json"
def metadata : String := "wandb-metadata.json"
def events : String := "wandb-events.jsonl"
def output : String := "output.log"
def diff : String := "diff.patch"

end Wandb.Filenames
