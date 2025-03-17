# Contributing to Compass

Thank you for your interest in contributing to Compass! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) in all your interactions with the project.

## Development Environment

### Setting Up

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/compass.git`
3. Navigate to the project directory: `cd compass`
4. Install development dependencies: `pip install -e ".[dev]"`

### Development Workflow

1. Create a new branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests to ensure your changes work as expected: `hatch run test:unit`
4. Run type checking: `hatch run test:types`
5. Run linting: `hatch run check`
6. Fix any issues: `hatch run fix`
7. Commit your changes: `git commit -am "Add your descriptive commit message"`
8. Push to your fork: `git push origin feature/your-feature-name`
9. Open a pull request

## Testing

- Run unit tests: `hatch run test:unit`
- Run integration tests: `hatch run test:integration`
- Run a specific test: `hatch run test:unit test/path/to/test_file.py::TestClass::test_method`

## Coding Style

We follow these conventions:

- Use type hints for all function parameters and return values
- Maximum line length is 120 characters
- Follow [PEP 8](https://peps.python.org/pep-0008/) style guidelines
- Use docstrings for all public classes and methods

## Pull Request Process

1. Ensure your code is well-tested and passes all existing tests
2. Update documentation to reflect any changes
3. Add a release note using `hatch run release-note your-feature-name`
4. The PR title should be descriptive and concise
5. Reference any related issues in your PR description
6. PRs need at least one approval before they can be merged

## Documentation

- Add docstrings to all public classes and methods
- Update README.md and other documentation as necessary
- Consider adding examples for new features

## Adding New Components

When adding a new component:

1. Create a new file in the appropriate module under `compass/components/`
2. Implement your component by inheriting from `Component`
3. Add comprehensive tests for your component
4. Update relevant documentation
5. Add your component to the appropriate `__init__.py` file

## Release Process

Releases are managed by the core maintainers. The process typically involves:

1. Updating version numbers
2. Generating release notes
3. Creating a GitHub release
4. Publishing to PyPI

## Questions?

If you have any questions or need assistance, please open an issue or reach out to the maintainers.