# LeanDidax Justfile
# This file contains common commands for development and use of the LeanDidax library.

# Default recipe to run when just is called without arguments
default:
    @just --list

# Build the project
build:
    lake build

# Run all examples
run-examples:
    lake env lean --run LeanDidax2/Examples.lean
    lake env lean --run LeanDidax2/BatchExamples.lean
    lake env lean --run LeanDidax2/CustomRulesExamples.lean
    lake env lean --run LeanDidax2/ControlFlowExamples.lean
    lake env lean --run LeanDidax2/MonadicExamples.lean
    lake env lean --run LeanDidax2/MonadTransformerExamples.lean
    lake env lean --run LeanDidax2/StateTransformerExamples.lean
    lake env lean --run LeanDidax2/ExceptTransformerExamples.lean
    lake env lean --run LeanDidax2/WriterTransformerExamples.lean

# Run specific example files
run-basic-examples:
    lake env lean --run LeanDidax2/Examples.lean

run-batch-examples:
    lake env lean --run LeanDidax2/BatchExamples.lean

run-custom-rules-examples:
    lake env lean --run LeanDidax2/CustomRulesExamples.lean

run-control-flow-examples:
    lake env lean --run LeanDidax2/ControlFlowExamples.lean

run-monadic-examples:
    lake env lean --run LeanDidax2/MonadicExamples.lean

run-transformer-examples:
    lake env lean --run LeanDidax2/MonadTransformerExamples.lean

run-state-examples:
    lake env lean --run LeanDidax2/StateTransformerExamples.lean

run-except-examples:
    lake env lean --run LeanDidax2/ExceptTransformerExamples.lean

run-writer-examples:
    lake env lean --run LeanDidax2/WriterTransformerExamples.lean

# Clean build artifacts
clean:
    lake clean

# Format all Lean files
format:
    find . -name "*.lean" -not -path "./build/*" -not -path "./lake/*" | xargs -n 1 lean --run -- format

# Check code for errors without building
check:
    lake check

# Generate documentation
docs:
    @echo "Generating documentation..."
    lake update-manifest
    lean --doc --doc-dir=docs .
    @echo "Documentation generated in docs/ directory"
    
# Create a new release (requires version argument)
release VERSION:
    @echo "Creating release v{{VERSION}}"
    sed -i "" "s/## \[[0-9]*\.[0-9]*\.[0-9]*\].*/## [{{VERSION}}] - $(date '+%Y-%m-%d')/" CHANGELOG.md
    git add CHANGELOG.md
    git commit -m "Release v{{VERSION}}"
    git tag -a "v{{VERSION}}" -m "Release v{{VERSION}}"
    @echo "Created release v{{VERSION}}"

# Lint the codebase
lint:
    lake env lean --run scripts/lint.lean

# Run tests
test:
    lake test

# Create a complete project build and run all examples
all: build run-examples

# Development workflow - build and run examples in watch mode
watch:
    watchexec -e lean "lake build && lake env lean --run LeanDidax2/Examples.lean" 