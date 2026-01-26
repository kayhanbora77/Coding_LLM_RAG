# Pull Request Review Assistant Skill

This skill helps review pull requests by checking for common issues and best practices.

## Purpose
Automate the initial review of pull requests to identify potential issues, ensure code quality, and maintain consistency with project standards.

## Skill Definition
- **Name**: PR Review Assistant
- **Category**: Code Quality
- **Description**: Assists in reviewing pull requests by identifying common issues
- **Trigger Keywords**: pr, review, pull request, code review, quality check
- **Applicability**: Pull request review processes

## Instructions for Qoder
When a user requests a code review or asks about pull request quality, analyze the code changes and provide feedback on:

- Code style and formatting
- Potential bugs or issues
- Performance considerations
- Security vulnerabilities
- Best practice violations
- Documentation completeness
- Test coverage suggestions

## Example Usage
```
User: "Can you review this PR?"
Qoder: Analyzes the code changes and provides feedback on potential issues

User: "What should I check in this pull request?"
Qoder: Provides a checklist of items to review based on the code changes
```

## Variables
- `{{code_changes}}`: The code changes to review
- `{{project_language}}`: The programming language of the project
- `{{project_standards}}`: The project's specific coding standards
- `{{review_focus}}`: Areas to focus on (security, performance, etc.)
- `{{risk_level}}`: Level of risk assessment required

## Execution Flow
1. Analyze the code changes to understand the scope and purpose
2. Check for adherence to language-specific best practices
3. Identify potential bugs or problematic patterns
4. Assess security implications of the changes
5. Evaluate performance considerations
6. Check for proper documentation and tests
7. Summarize findings with specific recommendations
8. Prioritize issues by severity

## Dependencies
- Ability to read and parse code files
- Understanding of common patterns and anti-patterns for the language
- Knowledge of security best practices

## Error Handling
- If code is unclear, request clarification
- If changes span multiple concerns, suggest breaking into smaller PRs
- If unable to assess certain aspects, note limitations

## Notes
- Reviews should be constructive and educational
- Focus on objective issues rather than personal preferences
- Provide specific suggestions for improvements
- Acknowledge positive aspects of the code too
- Consider the project's specific context and constraints
