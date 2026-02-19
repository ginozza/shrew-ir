# Architecture: shrew-ir

`shrew-ir` implements the compiler stack for the Shrew declarative language (`.sw`). It is responsible for parsing source code, validating strict typing rules, optimizing the computation graph, and executing it using the underlying `shrew-core` backend.

## Core Concepts

- **AST (Abstract Syntax Tree)**: Represents the syntactic structure of a `.sw` program, including blocks like `@model`, `@config`, and `@graph`.
- **Lowering**: The process of converting the high-level AST into a directed acyclic graph (DAG) of operations executable by the runtime.
- **Shape Inference**: Statically determines the shape of every tensor in the graph before execution to ensure validity.
- **Optimization**: Applies graph rewrites (e.g., constant folding, dead code elimination) to improve performance.

## File Structure

| File | Description | Lines of Code |
| :--- | :--- | :--- |
| `parser.rs` | A recursive descent parser that transforms a token stream into an AST. Handles complex grammar rules for directives and expressions. | 1263 |
| `lower.rs` | Converts the specific AST logic into a comprehensive computation graph, resolving variable references and imports. | 886 |
| `lexer.rs` | Tokenizes raw source code strings into a stream of typed tokens (keywords, identifiers, literals). | 691 |
| `graph.rs` | Defines the `Graph` structure, `Node` types, and execution order for the runtime. | 618 |
| `optimize.rs` | Implements optimization passes that verify and simplify the graph structure (e.g., fusing operations). | 549 |
| `validate.rs` | Performs semantic analysis, type checking, and logical consistency verification on the graph modles. | 510 |
| `shapes.rs` | Implements logic for shape inference and propagation through the graph layers. | 486 |
| `ast.rs` | Defines the `Program`, `Block`, `Stmt`, and `Expr` enums that make up the Abstract Syntax Tree. | 472 |
| `token.rs` | Defines the `Token` enum and associated metadata (spans, locations). | 351 |
| `error.rs` | Compiler-specific error types with rich context for user feedback. | 75 |
| `lib.rs` | Crate entry point. | 39 |
