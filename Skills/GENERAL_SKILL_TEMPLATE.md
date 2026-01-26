# General Skill Template

This is a template for creating a general skill for the Qoder CLI. This skill demonstrates the structure and format for creating custom skills.

## Purpose
This skill template provides a framework for teaching Qoder CLI how to perform a specific task. Skills are markdown files that define how Qoder should approach certain types of requests.

## Skill Definition
- **Name**: General Skill Template
- **Category**: Template
- **Description**: A template for creating custom skills for Qoder CLI
- **Trigger Keywords**: skill, template, create, custom
- **Applicability**: General purpose skill creation

## Instructions for Qoder
When a user requests to create a new skill or asks about skill structure, use this template as a reference. Follow these steps:

1. Identify the specific task the skill should perform
2. Define the skill's purpose and scope
3. Outline the expected inputs and outputs
4. Specify any required tools or dependencies
5. Define the execution steps
6. Include error handling and edge cases

## Example Usage
```
User: "Help me create a skill for code review"
Qoder: Uses this template to guide the user in creating a code review skill
```

## Variables
- `{{skill_name}}`: Name of the skill to create
- `{{skill_purpose}}`: Purpose of the skill
- `{{skill_category}}`: Category of the skill
- `{{trigger_keywords}}`: Keywords that trigger the skill
- `{{dependencies}}`: Any required dependencies
- `{{execution_steps}}`: Steps to execute the skill

## Execution Flow
1. Parse user request to identify the type of skill needed
2. Use this template as a basis for creating the new skill
3. Customize the template with specific details for the requested skill
4. Validate the skill structure
5. Present the skill template to the user

## Dependencies
- None required for this template skill

## Error Handling
- If user doesn't provide enough information, ask for clarification
- If the skill category is unknown, suggest common categories
- If dependencies are missing, inform the user of requirements

## Notes
- This is a template skill and should be customized for specific use cases
- Skills should focus on a single, specific task
- The skill should be autonomous and not require constant user input
