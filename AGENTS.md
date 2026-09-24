# Repository Rules for Coding Agents

These rules apply to the whole repository.

## Keep the implementation small

- Treat every added line, branch, helper, dependency, and backend route as a
  maintenance cost.
- Search for an existing implementation before adding a new one. Extend the
  existing path when its ownership remains clear.
- When replacing an implementation, remove the replaced path in the same
  change. Do not retain speculative, experimental, or compatibility code.
- Do not add commented-out implementations, one-off benchmark logic, or
  generated build artifacts to the package source.
- Keep benchmark campaigns and publication outputs in `fastEmbedR-extra`, not
  in this package repository.

## No hidden execution paths

- A requested backend must either run or fail explicitly. Never add a silent
  backend, precision, algorithm, or data-conversion fallback.
- Every production route must have a caller, visible result metadata, and a
  focused test. Remove private helpers and native registrations with no caller.
- Do not add aliases unless they preserve a documented public API. Give an
  approved compatibility alias a removal plan and a test.
- Keep backend selection in the existing dispatch layer. Do not reproduce it
  inside an optimizer or preprocessing helper.

## Code limits

- Keep authored lines at 80 columns or fewer.
- Keep R functions at 50 lines or fewer.
- Prefer deletion and simplification over mechanical helper extraction that
  increases total complexity.
- Generated `RcppExports` and Rd files are not hand-edited or style-counted.

## Scope and verification

- Preserve unrelated and uncommitted user changes.
- Add the smallest test that proves the changed behavior and rejects fallback.
- Run `Rscript tools/check_source_style.R` and the relevant package tests.
- Do not commit, push, publish, or clean user files without explicit approval.
