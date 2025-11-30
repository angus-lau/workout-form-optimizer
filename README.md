# workout-optimizer

## setup
1. `git clone git@github.com:angus-lau/workout-form-optimizer.git`
2. `cd workout-form-optimizer`
3. `source workout/bin/activate/`
4. `pip install -r requirements.txt`

## naming
- snake_case for variables and functions
- PascalCase for classes
- UPPER_CASE for constants

## structure
- functions short and single purpose
- imports at top of file
- each module should handle one responsibility

## docstring + type hints
- add docstring for all functions and classes
```py
def example_function(path: str, output_dir: str) -> list:
    """Preprocess a raw video into standardized frames."""
```

## clean code
- avoid unused variables
- avoid commented out code left behind
- avoid giant functions

## git
- present tense e.g 'add function a'
- branch off for each feat/fix and add pull request
- delete branch locally after pull request accepted
- never push to main

