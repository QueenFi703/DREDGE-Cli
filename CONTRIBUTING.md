# Contributing to DREDGE

Thank you for your interest in contributing to DREDGE!

## Branch Naming Convention

When creating a new branch, please follow these naming conventions:

### Format

Branches should follow the pattern: `<type>/<short-description>`

### Types

- **feature/** - New features or enhancements
  - Example: `feature/add-dolly-integration`
  - Example: `feature/gpu-acceleration`

- **fix/** - Bug fixes
  - Example: `fix/memory-leak`
  - Example: `fix/typo-in-readme`

- **docs/** - Documentation updates
  - Example: `docs/update-api-reference`
  - Example: `docs/add-contributing-guide`

- **refactor/** - Code refactoring without changing functionality
  - Example: `refactor/simplify-event-handling`
  - Example: `refactor/optimize-state-persistence`

- **test/** - Adding or updating tests
  - Example: `test/add-integration-tests`
  - Example: `test/improve-coverage`

- **chore/** - Maintenance tasks, dependency updates
  - Example: `chore/update-dependencies`
  - Example: `chore/setup-ci`

### Guidelines

1. **Use lowercase** - All branch names should be lowercase
2. **Use hyphens** - Separate words with hyphens, not underscores or spaces
3. **Be descriptive** - Use clear, meaningful descriptions (but keep them concise)
4. **No special characters** - Avoid special characters except hyphens and forward slashes

### Examples

✅ Good branch names:
- `feature/add-cli-command`
- `fix/handle-empty-input`
- `docs/update-installation-guide`
- `refactor/split-large-module`

❌ Bad branch names:
- `my-branch` (no type prefix)
- `Feature/AddNewThing` (not lowercase)
- `fix_bug_123` (uses underscores)
- `update docs` (has spaces)

## Development Workflow

1. Create a new branch from `main` using the naming convention above
2. Make your changes in small, focused commits
3. Write or update tests as needed
4. Update documentation if applicable
5. Submit a pull request to `main`

## Questions?

If you're unsure about what to name your branch, consider:
- What type of change are you making?
- What is the main goal of your change?
- How would you describe it in 2-4 words?

For more information, check the [README.md](README.md) or open an issue.
