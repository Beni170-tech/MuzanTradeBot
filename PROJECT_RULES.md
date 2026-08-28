# PROJECT_RULES.md

## MuzanTradeBot Development Rules

These rules are non-negotiable and must be followed throughout the development process.

### Rule 1: Protect Capital First
Every feature, algorithm, and trading strategy must prioritize capital preservation. No feature is complete if it compromises the safety of the trading capital.

### Rule 2: Never Trade Without Passing All Safety Checks
Every trade execution must pass through the complete safety checklist:
- Risk assessment approved
- Position sizing calculated
- Stop-loss levels set
- Market conditions verified
- Multiple confirmation signals received

### Rule 3: Build One Feature at a Time
Focus on completing one feature fully before moving to the next. This ensures:
- Better code quality
- Easier debugging
- More thorough testing
- Clearer documentation

### Rule 4: Test Every Feature Before Moving On
No feature is considered complete until:
- Unit tests are written and passing
- Integration tests are performed
- Edge cases are handled
- Documentation is updated
- Code review is completed

### Rule 5: Never Commit Broken Code
All commits must:
- Pass all existing tests
- Not break existing functionality
- Include relevant documentation updates
- Follow the project's coding standards

## Development Workflow

1. **Plan** - Clearly define the feature or task
2. **Implement** - Write clean, documented code
3. **Test** - Verify all tests pass
4. **Review** - Check against project rules
5. **Document** - Update relevant documentation
6. **Commit** - Only after all checks pass

## Code Quality Standards

- Write self-documenting code
- Follow PEP 8 for Python code
- Include docstrings for all public functions
- Handle errors gracefully
- Log important operations and decisions

## Emergency Protocols

If any of these rules are compromised:
1. Stop development immediately
2. Assess the impact
3. Fix the issue before continuing
4. Document what happened and why
5. Update rules if necessary to prevent recurrence
