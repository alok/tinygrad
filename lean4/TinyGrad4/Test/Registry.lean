import Std
import LSpec
import TinyGrad4.Test.Case
import TinyGrad4.Test.PlausibleProps
import TinyGrad4.Test.Port.TensorProps
import TinyGrad4.Test.Port.OpsProps
import TinyGrad4.Test.Port.IndexingProps
import TinyGrad4.Test.Port.CuratedMixed

namespace TinyGrad4.Test

open Std


def allCases : List TestCase :=
  Port.TensorProps.cases ++
  Port.OpsProps.cases ++
  Port.IndexingProps.cases ++
  Port.CuratedMixed.cases

private def hasTag (tc : TestCase) (tag : String) : Bool :=
  tc.tags.contains tag

private def groupMatches (group? : Option String) (tc : TestCase) : Bool :=
  match group? with
  | none => true
  | some g => tc.group == g

private def filterMatches (filter? : Option String) (tc : TestCase) : Bool :=
  match filter? with
  | none => true
  | some f => tc.name.contains f || tc.group.contains f


def selectCases (cfg : RunConfig) (group? filter? : Option String) : List TestCase :=
  allCases.filter fun tc =>
    TestProfile.allows cfg.profile tc.minProfile &&
    hasTag tc "cpu" &&
    groupMatches group? tc &&
    filterMatches filter? tc

private def insertSuite (m : HashMap String (List LSpec.TestSeq)) (group : String) (suite : LSpec.TestSeq) : HashMap String (List LSpec.TestSeq) :=
  let prev := m.getD group []
  m.insert group (prev ++ [suite])


def toSuiteMap (cfg : RunConfig) (cases : List TestCase) : HashMap String (List LSpec.TestSeq) :=
  cases.foldl (init := HashMap.emptyWithCapacity cases.length) fun m tc =>
    insertSuite m tc.group (tc.suite cfg)


def renderCase (tc : TestCase) : String :=
  let tags := String.intercalate "," tc.tags
  s!"{tc.name} (group={tc.group}, minProfile={repr tc.minProfile}, tags=[{tags}])"

end TinyGrad4.Test
