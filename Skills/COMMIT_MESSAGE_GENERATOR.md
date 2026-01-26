# Commit Message Generator Skill

This skill helps generate standardized commit messages based on the changes detected in the code.

## Purpose
Automatically generate conventional commit messages that follow standard formatting practices, making Git histories more readable and organized.

## Skill Definition
- **Name**: Commit Message Generator
- **Category**: Development Workflow
- **Description**: Generates conventional commit messages based on code changes
- **Trigger Keywords**: commit, message, git, conventional, changelog
- **Applicability**: Git repository environments

## Instructions for Qoder
When a user requests help with creating a commit message or asks about conventional commits, analyze the code changes and generate an appropriate commit message following conventional commit format.

Types:
- feat: A new feature
- fix: A bug fix
- docs: Documentation only changes
- style: Changes that do not affect the meaning of the code
- refactor: A code change that neither fixes a bug nor adds a feature
- perf: A code change that improves performance
- test: Adding missing tests or correcting existing tests
- build: Changes that affect the build system or external dependencies
- ci: Changes to CI configuration files and scripts
- chore: Other changes that don't modify src or test files

## Example Usage
```
User: "Generate a commit message for these changes"
Qoder: Analyzes the changes and generates: "feat(auth): add user login validation"

User: "What should I write for this commit?"
Qoder: Reviews the diff and suggests a conventional commit message
```

## Variables
- `{{change_type}}`: Type of change (feat, fix, docs, etc.)
- `{{scope}}`: Scope of the change (optional)
- `{{description}}`: Brief description of the change
- `{{breaking_changes}}`: Breaking changes description (if any)
- `{{issues_closed}}`: Issues closed by this commit

## Execution Flow
1. Detect the type of changes in the code (new feature, bug fix, etc.)
2. Identify the scope of changes (which module/component is affected)
3. Summarize the changes concisely
4. Construct the commit message in conventional format: `<type>(<scope>): <description>`
5. If applicable, include breaking changes or closed issues
6. Return the formatted commit message

## Dependencies
- Git must be available in the environment
- Ability to read current changes (git diff)

## Error Handling
- If unable to detect change type, ask user for clarification
- If scope is unclear, suggest a general scope or ask user
- If no changes detected, inform user to stage changes first

## Notes
- Keep descriptions concise (under 50 characters for header)
- Use imperative mood in descriptions ("add" not "adds")
- Lowercase type and scope
- This skill should help maintain consistent commit message standards across projects
