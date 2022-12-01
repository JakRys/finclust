# Tests

For testing, you need to install `pytest` and `pytest-cov` packages.


## Testing
To run all tests:

```bash
pytest -v tests/
```

To run a specific test:

```bash
pytest -v ./tests/test_specific_file.py
```

## Test Coverage
To get test coverage:
```bash
pytest --cov=finclust/ tests/ 
```
