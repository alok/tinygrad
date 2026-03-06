import TinyGrad4.Essay

open Verso.Genre.Manual (Config manualMain)

def config : Config where
  emitTeX := false
  emitHtmlSingle := false
  emitHtmlMulti := true
  htmlDepth := 2

def main := manualMain (%doc TinyGrad4.Essay) (config := config)
