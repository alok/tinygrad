import Std

/-!
Minimal HTTP helper using {lit}`curl` for TLS, built around raw request data.
-/

namespace Wandb.Http

/-- Raw HTTP response from {lit}`curl`. -/
structure Response where
  exitCode : Int
  stdout : String
  stderr : String

/-- Raw HTTP request specification. -/
structure Request where
  method : String := "POST"
  url : String
  headers : List (String × String) := []
  body : Option String := none
  dataFile : Option System.FilePath := none
  uploadFile : Option System.FilePath := none
  outputFile : Option System.FilePath := none
  timeoutMs : Option Nat := none
  failOnError : Bool := true

/-- Build {lit}`curl` arguments from a request. -/
def Request.toCurlArgs (r : Request) : Array String :=
  let base := ["-sS", "-X", r.method]
  let failArgs := if r.failOnError then ["--fail"] else []
  let headerArgs :=
    r.headers.foldl (init := []) fun acc (k, v) =>
      acc ++ ["-H", s!"{k}: {v}"]
  let timeoutArgs :=
    match r.timeoutMs with
    | none => []
    | some t =>
      let secs := (Float.ofNat t) / 1000.0
      ["--max-time", toString secs]
  let outputArgs :=
    match r.outputFile with
    | none => []
    | some path => ["-o", path.toString]
  let bodyArgs :=
    match r.uploadFile with
    | some path => ["--upload-file", path.toString]
    | none =>
      match r.dataFile with
      | some path => ["--data-binary", "@" ++ path.toString]
      | none =>
        match r.body with
        | none => []
        | some body => ["--data", body]
  (base ++ failArgs ++ headerArgs ++ timeoutArgs ++ outputArgs ++ bodyArgs ++ [r.url]).toArray

/-- Execute a raw HTTP request via {lit}`curl`. -/
def run (r : Request) : IO Response := do
  let out ← IO.Process.output {
    cmd := "curl"
    args := r.toCurlArgs
  }
  let exitCode := Int.ofNat out.exitCode.toNat
  pure { exitCode := exitCode, stdout := out.stdout, stderr := out.stderr }

/-- Convenience helper for posting JSON using {lit}`curl`. -/
def postJson (url : String) (headers : List (String × String)) (body : String) : IO Response :=
  run {
    method := "POST"
    url := url
    headers := ("Content-Type", "application/json") :: headers
    body := some body
  }

/-- Convenience helper for HTTP GET. -/
def get (url : String) (headers : List (String × String)) : IO Response :=
  run {
    method := "GET"
    url := url
    headers := headers
  }

/-- Convenience helper for HTTP PUT. -/
def put (url : String) (headers : List (String × String)) (body : Option String := none) : IO Response :=
  run {
    method := "PUT"
    url := url
    headers := headers
    body := body
  }

/-- Convenience helper for HTTP DELETE. -/
def delete (url : String) (headers : List (String × String)) : IO Response :=
  run {
    method := "DELETE"
    url := url
    headers := headers
  }

end Wandb.Http
