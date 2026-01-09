import Wandb.Http

/-! File upload helpers for W&B-signed URLs. -/

namespace Wandb

/-- Upload a file to a signed URL via {lit}`curl --upload-file`. -/
def uploadFile (url : String) (path : System.FilePath) (headers : List (String × String) := []) : IO Http.Response :=
  Http.run {
    method := "PUT"
    url := url
    headers := headers
    uploadFile := some path
  }

/-- Download a file from a URL via {lit}`curl -o`. -/
def downloadFile (url : String) (path : System.FilePath) (headers : List (String × String) := []) : IO Http.Response :=
  Http.run {
    method := "GET"
    url := url
    headers := headers
    outputFile := some path
  }

/-- Upload a file with explicit content type. -/
def uploadFileWithContentType
    (url : String)
    (path : System.FilePath)
    (contentType : String) : IO Http.Response :=
  uploadFile url path [("Content-Type", contentType)]

end Wandb
