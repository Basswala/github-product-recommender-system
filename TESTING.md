# Testing Framework for Product Recommender System

## Overview

A comprehensive testing framework has been set up for the product recommender system using pytest. The framework includes unit tests, integration tests, and proper mocking for external dependencies.

## Test Results Summary

**Current Status**: 79 tests passed, 21 tests failed (79% pass rate)

### ✅ Working Components
- **Config Module**: Basic configuration tests working
- **Data Converter**: All CSV processing tests passing
- **Data Ingestion**: Core functionality tests passing with mocks
- **RAG Chain**: Basic initialization and history management working
- **Flask App**: Core routing and response handling working
- **Utilities**: Logger functionality working

### ⚠️ Issues to Address
1. **Environment Variable Mocking**: Real env vars overriding test mocks
2. **CustomException Implementation**: String representation differs from expected
3. **RAG Chain Mocking**: Some complex mock configurations need adjustment
4. **Metrics Testing**: Prometheus counter behavior in tests

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Shared fixtures and configuration
├── test_config.py             # Configuration module tests
├── test_data_converter.py     # CSV to Document conversion tests
├── test_data_ingestion.py     # Data ingestion pipeline tests
├── test_rag_chain.py          # RAG chain implementation tests
├── test_flask_app.py          # Flask application integration tests
├── test_utils.py              # Utility modules tests
└── data/
    └── sample_reviews.csv     # Test data for CSV processing
```

## Running Tests

### Install Test Dependencies
```bash
uv add --dev pytest pytest-cov pytest-mock pytest-flask pytest-asyncio httpx faker
```

### Basic Test Execution
```bash
# Run all tests
uv run pytest tests/

# Run without coverage
uv run pytest tests/ --no-cov

# Run specific test file
uv run pytest tests/test_config.py -v

# Run specific test
uv run pytest tests/test_config.py::TestConfig::test_model_configuration_constants -v
```

### Using the Test Runner Script
```bash
# Run all tests with coverage
./run_tests.py --coverage

# Run only unit tests
./run_tests.py --unit-only

# Run only fast tests (exclude slow/integration)
./run_tests.py --fast

# Run specific test file
./run_tests.py --specific-test tests/test_config.py
```

## Test Categories

### Unit Tests (`@pytest.mark.unit`)
- Configuration management
- Data conversion logic
- Individual component functionality
- Utility functions

### Integration Tests (`@pytest.mark.integration`)
- Flask application routes
- End-to-end data flow
- Component interactions

### Slow Tests (`@pytest.mark.slow`)
- Tests that require external API calls
- Long-running operations
- Performance tests

## Mock Strategy

The test suite uses comprehensive mocking to isolate components:

- **External APIs**: Groq LLM, HuggingFace models
- **Database**: AstraDB vector store
- **File System**: CSV file operations
- **Environment Variables**: Configuration isolation

## Coverage Goals

- **Current Coverage**: ~50% (configurable in pyproject.toml)
- **Target Coverage**: 80%+
- **Coverage Reports**: Generated in `htmlcov/index.html`

## Key Features

1. **Comprehensive Fixtures**: Reusable test data and mock objects
2. **Parameterized Tests**: Multiple test scenarios with different inputs
3. **Error Testing**: Proper exception handling validation
4. **Mock Isolation**: Clean separation of concerns
5. **Integration Testing**: End-to-end workflow validation

## Continuous Integration

The test framework is ready for CI/CD integration:

```yaml
# Example GitHub Actions workflow
- name: Run Tests
  run: |
    uv run pytest tests/ --cov=flipkart --cov=utils --cov-report=xml
    uv run pytest tests/ --cov=flipkart --cov=utils --cov-fail-under=70
```

## Next Steps

1. **Fix Environment Variable Mocking**: Implement proper isolation
2. **Adjust CustomException Tests**: Match actual implementation
3. **Refine RAG Chain Mocks**: Fix complex chain building tests
4. **Improve Metrics Testing**: Handle Prometheus counter behavior
5. **Add Performance Tests**: Benchmark critical operations
6. **Increase Coverage**: Add tests for edge cases

## Best Practices

- Use descriptive test names
- Test both success and failure scenarios
- Mock external dependencies
- Keep tests independent and isolated
- Use fixtures for common setup
- Maintain high test coverage
- Document complex test scenarios

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Mock Failures**: Check mock configuration and patching
3. **Environment Conflicts**: Use proper isolation techniques
4. **Coverage Issues**: Verify test execution paths

### Debug Mode
```bash
# Run with verbose output
uv run pytest tests/ -v -s

# Run with debug information
uv run pytest tests/ --pdb

# Show test output
uv run pytest tests/ -s
```

This testing framework provides a solid foundation for maintaining code quality and ensuring the reliability of the product recommender system.
