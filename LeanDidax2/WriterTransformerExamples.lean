/-
  Examples demonstrating the WriterTransformer module for logging in differentiable computations.
-/

import LeanDidax2.Basic
import LeanDidax2.Monadic
import LeanDidax2.WriterTransformer

namespace LeanDidax2.WriterTransformerExamples

open LeanDidax2
open LeanDidax2.Monadic
open LeanDidax2.WriterTransformer

/--
  Simple polynomial example with logging.
-/
def loggedPolynomial (x : Value Float) : Monadic.DiffM (LoggedOperation (Value Float)) := do
  let x² ← loggedOp "square" x x (fun a b => a * b)
  let term2 ← loggedOp "2x" (constValue 2.0) x (fun a b => a * b)
  let sum ← loggedOp "x² + 2x" x².result term2.result (fun a b => a + b)
  let result ← loggedOp "sum + 1" sum.result (constValue 1.0) (fun a b => a + b)

  let allLogs := #["Computing polynomial x² + 2x + 1"] ++
                 x².logs ++
                 term2.logs ++
                 sum.logs ++
                 result.logs ++
                 #["Polynomial computation complete"]

  pure {
    result := result.result,
    logs := allLogs
  }

/--
  Simple example of logging a polynomial computation.
-/
def simplePolynomialExample (x : Float) : IO Unit := do
  IO.println s!"Simple polynomial example with x = {x}"
  IO.println "--------------------------------------"

  let computation := loggedPolynomial (seed x)

  runDiffWithLogsIO computation (fun result logs => do
    IO.println s!"Result: {result.primal}"
    IO.println s!"Gradient: {result.tangent}"
    IO.println ""
    IO.println "Operation log:"
    IO.println (formatLogs (getOperationLogs logs))
  )

/--
  Example showing how to log multiple operations and combine logs.
-/
def manualLoggingExample (x : Float) : IO Unit := do
  IO.println s!"Manual logging example with x = {x}"
  IO.println "--------------------------------------"

  let xv := seed x

  -- Log different steps manually
  let step1 ← loggedOp "square" xv xv (fun a b => a * b)
  let step2 ← loggedOp "add_one" step1.result (constValue 1.0) (fun a b => a + b)

  -- Combine logs
  let logs := step1.logs ++ step2.logs

  -- Display results
  IO.println s!"Operation: f(x) = x² + 1"
  IO.println s!"Result: {step2.result.primal}"
  IO.println s!"Gradient: {step2.result.tangent}"
  IO.println ""
  IO.println "Detailed log:"
  IO.println (formatLogs logs)

/--
  Example showing formatted logs with timestamps.
-/
def timestampExample (x : Float) : IO Unit := do
  IO.println s!"Timestamp example with x = {x}"
  IO.println "--------------------------------------"

  let computation := loggedPolynomial (seed x)

  runDiffWithLogsIO computation (fun result logs => do
    IO.println s!"Result: {result.primal}"
    IO.println ""
    IO.println "Log with timestamps:"
    IO.println (formatLogsWithTimestamps logs)
  )

/--
  Example demonstrating filtering logs.
-/
def filteredLogsExample (x : Float) : IO Unit := do
  IO.println s!"Filtered logs example with x = {x}"
  IO.println "--------------------------------------"

  let computation := loggedPolynomial (seed x)

  runDiffWithLogsIO computation (fun result logs => do
    IO.println s!"Result: {result.primal}"
    IO.println ""

    let inputLogs := filterLogs logs (fun entry => entry.contains "Input")
    let resultLogs := filterLogs logs (fun entry => entry.contains "Result")

    IO.println "Input operations:"
    IO.println (formatLogs inputLogs)
    IO.println "Output results:"
    IO.println (formatLogs resultLogs)
  )

/--
  Main function to run all examples
-/
def main : IO Unit := do
  IO.println "\n===== WriterTransformer Examples =====\n"

  simplePolynomialExample 3.0

  IO.println "\n"
  manualLoggingExample 4.0

  IO.println "\n"
  timestampExample 2.0

  IO.println "\n"
  filteredLogsExample 5.0

end LeanDidax2.WriterTransformerExamples

/-- Main function accessible from the command line -/
def main : IO Unit := LeanDidax2.WriterTransformerExamples.main
