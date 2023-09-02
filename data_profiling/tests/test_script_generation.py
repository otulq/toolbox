import pandas as pd
import random
from datetime import date, timedelta
from importune import importunity
with importunity():
    from ..profile_df import (
        FloatProfile, IntProfile, BoolProfile, DateProfile, DateStrProfile,
        StringProfile, NullProfile, OneValueProfile, profile_column, DataFrameProfile
    )


def test_float_profile_sample_script():
    """Test FloatProfile.sample_script() method."""
    # Test without nulls
    profile = FloatProfile(missing_prob=0.0, min=1.0, max=10.0)
    script = profile.sample_script()
    assert script == "random.uniform(1.0, 10.0)"

    # also test the the script if run, generates a float within the range
    for _ in range(10):
        assert 1.0 <= eval(script) <= 10.0
    
    # Test with nulls
    profile_with_nulls = FloatProfile(missing_prob=0.3, min=1.0, max=10.0)
    script = profile_with_nulls.sample_script()
    # probability of not being null is 1 - missing_prob = 1 - 0.3 = 0.7
    assert script == "random.uniform(1.0, 10.0) if random.random() < 0.7 else None"

    # also test the the script if run, generates a float within the range
    for _ in range(10):
        result = eval(script)
        if result is not None:
            assert 1.0 <= result <= 10.0

    # Test ignore_missing
    script_ignore = profile_with_nulls.sample_script(ignore_missing=True)
    assert script_ignore == "random.uniform(1.0, 10.0)"

    # also test the the script if run, generates a float within the range
    for _ in range(10):
        result = eval(script_ignore)
        assert result is not None
        assert 1.0 <= result <= 10.0


def test_int_profile_sample_script():
    """Test IntProfile.sample_script() method."""
    # Test without nulls
    profile = IntProfile(missing_prob=0.0, min=1, max=10)
    script = profile.sample_script()
    assert script == "random.randint(1, 10)"
    
    # also test the the script if run, generates a int within the range
    for _ in range(10):
        assert 1 <= eval(script) <= 10

    # Test with nulls
    profile_with_nulls = IntProfile(missing_prob=0.2, min=1, max=10)
    script = profile_with_nulls.sample_script()
    # probability of not being null is 1 - missing_prob = 1 - 0.2 = 0.8
    assert script == "random.randint(1, 10) if random.random() < 0.8 else None"

    for _ in range(10):
        result = eval(script)
        if result is not None:
            assert 1 <= result <= 10


def test_bool_profile_sample_script():
    """Test BoolProfile.sample_script() method."""
    # Test without nulls
    profile = BoolProfile(missing_prob=0.0, true_prob=0.7)
    script = profile.sample_script()
    assert script == "random.random() < 0.7"
    
    # also test the the script if run, generates a bool
    for _ in range(10):
        assert eval(script) in [True, False]

    # Test with nulls
    profile_with_nulls = BoolProfile(missing_prob=0.1, true_prob=0.7)
    script = profile_with_nulls.sample_script()
    # probability of not being null is 1 - missing_prob = 1 - 0.1 = 0.9
    assert script == "random.random() < 0.7 if random.random() < 0.9 else None"

    # also test the the script if run, generates a bool
    for _ in range(10):
        result = eval(script)
        if result is not None:
            assert result in [True, False]


def test_date_profile_sample_script():
    """Test DateProfile.sample_script() method."""
    # Test without nulls
    profile = DateProfile(missing_prob=0.0, min=date(2022, 1, 1), max=date(2022, 12, 31))
    script = profile.sample_script()
    expected = "date.fromisoformat('2022-01-01') + timedelta(days=random.randint(0, (date.fromisoformat('2022-12-31') - date.fromisoformat('2022-01-01')).days))"
    assert script == expected
    
    # also test the the script if run, generates a date within the range
    for _ in range(10):
        result = eval(script)
        assert result is not None
        assert isinstance(result, date)
        assert date(2022, 1, 1) <= result <= date(2022, 12, 31)

    # Test with nulls
    profile_with_nulls = DateProfile(missing_prob=0.15, min=date(2022, 1, 1), max=date(2022, 12, 31))
    script = profile_with_nulls.sample_script()
    # probability of not being null is 1 - missing_prob = 1 - 0.15 = 0.85
    expected_with_nulls = f"{expected} if random.random() < 0.85 else None"
    assert script == expected_with_nulls

    # also test the the script if run, generates a date within the range
    for _ in range(10):
        result = eval(script)
        if result is not None:
            assert isinstance(result, date)
            assert date(2022, 1, 1) <= result <= date(2022, 12, 31)


def test_date_str_profile_sample_script():
    """Test DateStrProfile.sample_script() method."""
    # Test without nulls
    profile = DateStrProfile(missing_prob=0.0, min="2022-01-01", max="2022-12-31")
    script = profile.sample_script()
    expected = "(date.fromisoformat('2022-01-01') + timedelta(days=random.randint(0, (date.fromisoformat('2022-12-31') - date.fromisoformat('2022-01-01')).days))).isoformat()"
    assert script == expected
    
    # also test the the script if run, generates a date within the range
    for _ in range(10):
        result = eval(script)
        assert result is not None
        assert isinstance(result, str)
        date_result = date.fromisoformat(result)
        assert date(2022, 1, 1) <= date_result <= date(2022, 12, 31)

    # Test with nulls
    profile_with_nulls = DateStrProfile(missing_prob=0.25, min="2022-01-01", max="2022-12-31")
    script = profile_with_nulls.sample_script()
    # probability of not being null is 1 - missing_prob = 1 - 0.25 = 0.75
    expected_with_nulls = f"{expected} if random.random() < 0.75 else None"
    assert script == expected_with_nulls

    # also test the the script if run, generates a date within the range
    for _ in range(10):
        result = eval(script)
        if result is not None:
            assert isinstance(result, str)
            date_result = date.fromisoformat(result)
            assert date(2022, 1, 1) <= date_result <= date(2022, 12, 31)


def test_string_profile_sample_script():
    """Test StringProfile.sample_script() method."""
    # Test without nulls
    profile = StringProfile(missing_prob=0.0, values=["A", "B", "C"])
    script = profile.sample_script()
    assert script == "random.choice(['A', 'B', 'C'])"
    
    # also test the the script if run, generates a string
    for _ in range(10):
        result = eval(script)
        assert result is not None
        assert result in ["A", "B", "C"]

    # Test with nulls
    profile_with_nulls = StringProfile(missing_prob=0.2, values=["A", "B", "C"])
    script = profile_with_nulls.sample_script()
    # probability of not being null is 1 - missing_prob = 1 - 0.2 = 0.8
    assert script == "random.choice(['A', 'B', 'C']) if random.random() < 0.8 else None"

    # also test the the script if run, generates a string
    for _ in range(10):
        result = eval(script)
        if result is not None:
            assert result in ["A", "B", "C"]


def test_null_profile_sample_script():
    """Test NullProfile.sample_script() method."""
    profile = NullProfile()
    script = profile.sample_script()
    assert script == "None"
    
    # Test that script always returns None
    for _ in range(10):
        result = eval(script)
        assert result is None
    
    # Test ignore_missing (should still return None)
    script_ignore = profile.sample_script(ignore_missing=True)
    assert script_ignore == "None"
    
    for _ in range(10):
        result = eval(script_ignore)
        assert result is None


def test_one_value_profile_sample_script():
    """Test OneValueProfile.sample_script() method."""
    # Test string value without nulls
    profile_str = OneValueProfile(missing_prob=0.0, value="hello")
    script = profile_str.sample_script()
    assert script == "'hello'"
    
    # Test that script returns the string value
    for _ in range(10):
        result = eval(script)
        assert result == "hello"
    
    # Test string value with nulls
    profile_str_nulls = OneValueProfile(missing_prob=0.3, value="world")
    script = profile_str_nulls.sample_script()
    assert script == "'world' if random.random() < 0.7 else None"
    
    for _ in range(10):
        result = eval(script)
        if result is not None:
            assert result == "world"
    
    # Test integer value without nulls
    profile_int = OneValueProfile(missing_prob=0.0, value=42)
    script = profile_int.sample_script()
    assert script == "42"
    
    for _ in range(10):
        result = eval(script)
        assert result == 42
    
    # Test integer value with nulls
    profile_int_nulls = OneValueProfile(missing_prob=0.2, value=99)
    script = profile_int_nulls.sample_script()
    assert script == "99 if random.random() < 0.8 else None"
    
    for _ in range(10):
        result = eval(script)
        if result is not None:
            assert result == 99
    
    # Test boolean value
    profile_bool = OneValueProfile(missing_prob=0.0, value=True)
    script = profile_bool.sample_script()
    assert script == "True"
    
    for _ in range(10):
        result = eval(script)
        assert result is True
    
    # Test date value
    test_date = date(2023, 9, 2)
    profile_date = OneValueProfile(missing_prob=0.0, value=test_date)
    script = profile_date.sample_script()
    assert script == "date.fromisoformat('2023-09-02')"
    
    for _ in range(10):
        result = eval(script)
        assert result == test_date
    
    # Test date value with nulls
    profile_date_nulls = OneValueProfile(missing_prob=0.1, value=test_date)
    script = profile_date_nulls.sample_script()
    assert script == "date.fromisoformat('2023-09-02') if random.random() < 0.9 else None"
    
    for _ in range(10):
        result = eval(script)
        if result is not None:
            assert result == test_date
    
    # Test ignore_missing with nulls
    profile_with_nulls = OneValueProfile(missing_prob=0.5, value="test")
    script_ignore = profile_with_nulls.sample_script(ignore_missing=True)
    assert script_ignore == "'test'"
    
    for _ in range(10):
        result = eval(script_ignore)
        assert result == "test"


def test_samples_script_single_line():
    """Test samples_script() method in single-line mode."""
    # Test without nulls
    profile = IntProfile(missing_prob=0.0, min=1, max=10)
    script = profile.samples_script(5, single_line=True)
    assert script == "[random.randint(1, 10) for _ in range(5)]"
    
    # also test the the script if run, generates a list of ints
    for _ in range(10):
        result = eval(script)
        assert len(result) == 5
        assert all(isinstance(x, int) and 1 <= x <= 10 for x in result)

    # Test with nulls
    profile_with_nulls = FloatProfile(missing_prob=0.3, min=1.0, max=10.0)
    script = profile_with_nulls.samples_script(3, single_line=True)
    # probability of not being null is 1 - missing_prob = 1 - 0.3 = 0.7
    expected = "[random.uniform(1.0, 10.0) if random.random() < 0.7 else None for _ in range(3)]"
    assert script == expected

    # also test the the script if run, generates a list of floats
    for _ in range(10):
        result = eval(script)
        assert len(result) == 3
        assert all((x is None) or (isinstance(x, float) and 1.0 <= x <= 10.0) for x in result)


def test_samples_script_multi_line():
    """Test samples_script() method in multi-line mode."""
    # Test without nulls
    profile = StringProfile(missing_prob=0.0, values=["X", "Y"])
    script = profile.samples_script(4, single_line=False)
    expected = "[\n    random.choice(['X', 'Y'])\n    for _ in range(4)\n]"
    assert script == expected
    
    # also test the the script if run, generates a list of strings
    for _ in range(10):
        result = eval(script)
        assert len(result) == 4
        assert all(isinstance(x, str) and x in ["X", "Y"] for x in result)

    # Test with nulls
    profile_with_nulls = BoolProfile(missing_prob=0.1, true_prob=0.8)
    script = profile_with_nulls.samples_script(2, single_line=False, indent="    ")
    # probability of not being null is 1 - missing_prob = 1 - 0.1 = 0.9
    expected = (
        "[\n"
        "    random.random() < 0.8 if random.random() < 0.9 else None\n"
        "    for _ in range(2)\n"
        "]"
    )
    assert script == expected

    # also test the the script if run, generates a list of bools
    for _ in range(10):
        result = eval(script)
        assert len(result) == 2
        assert all((x is None) or (isinstance(x, bool) and x in [True, False]) for x in result)


def test_samples_script_custom_indent():
    """Test samples_script() with custom indentation."""
    profile = IntProfile(missing_prob=0.0, min=5, max=15)
    script = profile.samples_script(3, indent="  ", single_line=False)
    expected = (
        "[\n"
        "  random.randint(5, 15)\n"
        "  for _ in range(3)\n"
        "]"
    )
    assert script == expected

    # also test the the script if run, generates a list of ints
    for _ in range(10):
        result = eval(script)
        assert len(result) == 3
        assert all(isinstance(x, int) and 5 <= x <= 15 for x in result)


def test_generate_df_script():
    """Test DataFrameProfile.generate_df_script() method."""
    # Create a simple DataFrame for profiling
    df = pd.DataFrame({
        'int_col': [1, 2, 3, 4, 5],
        'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
        'str_col': ['A', 'B', 'A', 'C', 'B'],
        'bool_col': [True, False, True, True, False]
    })
    
    # Create profile
    profile = DataFrameProfile.from_df(df)
    
    # Generate script
    script = profile.generate_df_script(10)
    
    # Debug: print the script to see what it looks like
    print("Generated script:")
    print(script)
    
    # Basic checks - script should contain DataFrame creation
    assert script.startswith("pd.DataFrame({")
    assert script.endswith("})")
    assert "int_col" in script
    assert "float_col" in script
    assert "str_col" in script
    assert "bool_col" in script
    assert "for _ in range(10)" in script

    """
    Generated script should be:
    pd.DataFrame({
        int_col: [random.randint(1, 5) for _ in range(10)],
        float_col: [random.uniform(1.54, 5.0600000000000005) for _ in range(10)],
        str_col: [random.choice(['A', 'B', 'C']) for _ in range(10)],
        bool_col: [random.random() < 0.6 for _ in range(10)]
    })
    """

    # also test the the script if run, generates a DataFrame with the same structure
    print("Generated script:")
    print(script)
    df_result = eval(script)
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.shape == (10, 4)
    assert all(isinstance(x, int) and 1 <= x <= 5 for x in df_result["int_col"].tolist())
    assert all(isinstance(x, float) and 1.1 <= x <= 5.5 for x in df_result["float_col"].tolist())
    assert all(isinstance(x, str) and x in ["A", "B", "C"] for x in df_result["str_col"].tolist())
    assert all(isinstance(x, bool) and x in [True, False] for x in df_result["bool_col"].tolist())


def test_script_execution_integration():
    """Integration test: execute generated scripts and verify they work."""
    # Set up environment for script execution
    import random
    from datetime import date, timedelta
    import pandas as pd
    
    # Test IntProfile script execution
    profile = IntProfile(missing_prob=0.0, min=1, max=5)
    script = profile.sample_script()
    result = eval(script)
    assert isinstance(result, int)
    assert 1 <= result <= 5
    
    # Test FloatProfile script execution
    profile = FloatProfile(missing_prob=0.0, min=1.0, max=5.0)
    script = profile.sample_script()
    result = eval(script)
    assert isinstance(result, float)
    assert 1.0 <= result <= 5.0
    
    # Test BoolProfile script execution
    profile = BoolProfile(missing_prob=0.0, true_prob=1.0)
    script = profile.sample_script()
    result = eval(script)
    assert result is True
    
    # Test StringProfile script execution
    profile = StringProfile(missing_prob=0.0, values=["test"])
    script = profile.sample_script()
    result = eval(script)
    assert result == "test"
    
    # Test DateProfile script execution
    profile = DateProfile(missing_prob=0.0, min=date(2022, 1, 1), max=date(2022, 1, 2))
    script = profile.sample_script()
    result = eval(script)
    assert isinstance(result, date)
    assert date(2022, 1, 1) <= result <= date(2022, 1, 2)
    
    # Test DateStrProfile script execution
    profile = DateStrProfile(missing_prob=0.0, min="2022-01-01", max="2022-01-02")
    script = profile.sample_script()
    result = eval(script)
    assert isinstance(result, str)
    assert result in ["2022-01-01", "2022-01-02"]


def test_script_with_nulls_execution():
    """Test that scripts with null handling execute correctly."""
    import random
    
    # Test with missing values - this is probabilistic so we run it multiple times
    profile = IntProfile(missing_prob=1.0, min=1, max=5)  # Always missing
    script = profile.sample_script()
    result = eval(script)
    assert result is None
    
    profile_never_missing = IntProfile(missing_prob=0.0, min=1, max=5)  # Never missing
    script = profile_never_missing.sample_script()
    result = eval(script)
    assert result is not None
    assert 1 <= result <= 5


def test_samples_script_execution():
    """Test that samples_script generates executable code."""
    import random
    
    # Test samples script execution
    profile = IntProfile(missing_prob=0.0, min=1, max=3)
    script = profile.samples_script(5)
    result = eval(script)
    assert len(result) == 5
    assert all(isinstance(x, int) and 1 <= x <= 3 for x in result)
    
    # Test with nulls
    profile_with_nulls = IntProfile(missing_prob=0.5, min=1, max=3)
    script = profile_with_nulls.samples_script(10)
    result = eval(script)
    assert len(result) == 10
    assert all(x is None or (isinstance(x, int) and 1 <= x <= 3) for x in result)


def test_null_profile_samples_script():
    """Test NullProfile.samples_script() method."""
    profile = NullProfile()
    
    # Test single line
    script = profile.samples_script(5, single_line=True)
    assert script == "[None for _ in range(5)]"
    
    # Test that script generates list of Nones
    result = eval(script)
    assert len(result) == 5
    assert all(x is None for x in result)
    
    # Test multi-line
    script_multi = profile.samples_script(3, single_line=False)
    expected_multi = "[\n    None\n    for _ in range(3)\n]"
    assert script_multi == expected_multi


def test_one_value_profile_samples_script():
    """Test OneValueProfile.samples_script() method."""
    # Test without nulls
    profile = OneValueProfile(missing_prob=0.0, value="hello")
    script = profile.samples_script(3, single_line=True)
    assert script == "['hello' for _ in range(3)]"
    
    # Test that script generates list of the value
    result = eval(script)
    assert len(result) == 3
    assert all(x == "hello" for x in result)
    
    # Test with nulls
    profile_with_nulls = OneValueProfile(missing_prob=0.4, value=42)
    script = profile_with_nulls.samples_script(4, single_line=True)
    expected = "[42 if random.random() < 0.6 else None for _ in range(4)]"
    assert script == expected
    
    # Test that script can generate the value or None
    for _ in range(10):
        result = eval(script)
        assert len(result) == 4
        assert all(x == 42 or x is None for x in result)


def test_samples_script_with_variable():
    """Test samples_script() with use_variable parameter."""
    # Test IntProfile without nulls
    profile = IntProfile(missing_prob=0.0, min=1, max=10)
    
    # Default behavior (hard-coded)
    script_default = profile.samples_script(5)
    assert script_default == "[random.randint(1, 10) for _ in range(5)]"
    
    # With variable (default name)
    script_var = profile.samples_script(5, use_variable=True)
    assert script_var == "[random.randint(1, 10) for _ in range(n_values)]"
    
    # With custom variable name
    script_custom = profile.samples_script(5, use_variable=True, variable_name="num_rows")
    assert script_custom == "[random.randint(1, 10) for _ in range(num_rows)]"
    
    # Test with nulls
    profile_nulls = FloatProfile(missing_prob=0.3, min=1.0, max=10.0)
    script_nulls = profile_nulls.samples_script(3, use_variable=True)
    expected = "[random.uniform(1.0, 10.0) if random.random() < 0.7 else None for _ in range(n_values)]"
    assert script_nulls == expected
    
    # Test multi-line
    script_multi = profile.samples_script(4, single_line=False, use_variable=True)
    expected_multi = "[\n    random.randint(1, 10)\n    for _ in range(n_values)\n]"
    assert script_multi == expected_multi


def test_null_profile_samples_script_with_variable():
    """Test NullProfile.samples_script() with use_variable parameter."""
    profile = NullProfile()
    
    # Default behavior
    script_default = profile.samples_script(3)
    assert script_default == "[None for _ in range(3)]"
    
    # With variable
    script_var = profile.samples_script(3, use_variable=True)
    assert script_var == "[None for _ in range(n_values)]"
    
    # With custom variable name
    script_custom = profile.samples_script(3, use_variable=True, variable_name="sample_count")
    assert script_custom == "[None for _ in range(sample_count)]"
    
    # Multi-line with variable
    script_multi = profile.samples_script(3, single_line=False, use_variable=True)
    expected_multi = "[\n    None\n    for _ in range(n_values)\n]"
    assert script_multi == expected_multi


def test_generate_df_script_with_variable():
    """Test DataFrameProfile.generate_df_script() with use_variable parameter."""
    # Create simple DataFrame for testing
    df = pd.DataFrame({
        'int_col': [1, 2, 3],
        'str_col': ['A', 'B', 'A']
    })
    profile = DataFrameProfile.from_df(df)
    
    # Default behavior (hard-coded)
    script_default = profile.generate_df_script(10)
    assert "for _ in range(10)" in script_default
    assert script_default.startswith("pd.DataFrame({")
    assert "'int_col':" in script_default
    assert "'str_col':" in script_default
    
    # With variable (default name)
    script_var = profile.generate_df_script(10, use_variable=True)
    assert "for _ in range(n_values)" in script_var
    assert "for _ in range(10)" not in script_var
    assert script_var.startswith("pd.DataFrame({")
    
    # With custom variable name
    script_custom = profile.generate_df_script(10, use_variable=True, variable_name="dataset_size")
    assert "for _ in range(dataset_size)" in script_custom
    assert "for _ in range(10)" not in script_custom
    assert "for _ in range(n_values)" not in script_custom
    
    # Test that the script can still execute (with variable defined)
    import random
    n_values = 5
    result = eval(script_var)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (5, 2)
    assert list(result.columns) == ['int_col', 'str_col']
    
    # Test custom variable execution
    dataset_size = 3
    result_custom = eval(script_custom)
    assert isinstance(result_custom, pd.DataFrame)
    assert result_custom.shape == (3, 2)


def test_float_formatting_in_scripts():
    """Test that float formatting produces clean numbers in scripts."""
    # Test FloatProfile with messy decimal values
    profile_messy = FloatProfile(missing_prob=0.0, min=1.23456789, max=9.87654321)
    script = profile_messy.sample_script()
    
    # Should have clean rounded values
    assert "1.23" in script or "1.234" in script  # Some reasonable rounding
    assert "9.88" in script or "9.877" in script  # Some reasonable rounding
    assert "1.23456789" not in script  # Should not have the full precision
    assert "9.87654321" not in script  # Should not have the full precision
    
    # Test that script executes correctly
    import random
    result = eval(script)
    assert isinstance(result, float)
    assert 1.2 <= result <= 10.0  # Should be in reasonable range
    
    # Test OneValueProfile with messy float
    profile_one = OneValueProfile(missing_prob=0.0, value=3.14159265359)
    script_one = profile_one.sample_script()
    assert script_one in ["3.14", "3.141", "3.1416"]  # Should be reasonably rounded
    assert "3.14159265359" not in script_one
    
    # Test BoolProfile with messy probability
    profile_bool = BoolProfile(missing_prob=0.0, true_prob=0.666666666666)
    script_bool = profile_bool.sample_script()
    assert "0.67" in script_bool or "0.667" in script_bool  # Clean rounding
    assert "0.666666666666" not in script_bool


def test_float_formatting_preserves_distinctions():
    """Test that float formatting keeps different values distinct."""
    # Test very close values that could round to the same thing
    profile_close = FloatProfile(missing_prob=0.0, min=1.0001, max=1.0009)
    script = profile_close.sample_script()
    
    # Extract the min and max values from the script
    import re
    matches = re.findall(r'random\.uniform\(([^,]+), ([^)]+)\)', script)
    assert len(matches) == 1
    min_str, max_str = matches[0]
    min_val = float(min_str)
    max_val = float(max_str)
    
    # They should be different
    assert min_val != max_val, f"Values rounded to same: {min_val} == {max_val}"
    assert min_val < max_val, f"Min should be less than max: {min_val} < {max_val}"
    
    # Test with even closer values
    profile_very_close = FloatProfile(missing_prob=0.0, min=1.00001, max=1.00002)
    script_very_close = profile_very_close.sample_script()
    matches = re.findall(r'random\.uniform\(([^,]+), ([^)]+)\)', script_very_close)
    min_str, max_str = matches[0]
    min_val = float(min_str)
    max_val = float(max_str)
    assert min_val != max_val, "Even very close values should remain distinct"


def test_float_formatting_edge_cases():
    """Test float formatting edge cases."""
    from profile_df import _format_float_for_script
    
    # Test basic rounding
    assert _format_float_for_script(1.23456) == "1.23"
    assert _format_float_for_script(9.87654) == "9.88"
    
    # Test with other_value to ensure distinction
    assert _format_float_for_script(1.111111, 1.222222) in ["1.11", "1.111"]
    assert _format_float_for_script(1.222222, 1.111111) in ["1.22", "1.222"]
    
    # Test very close values
    result1 = _format_float_for_script(1.0001, 1.0002)
    result2 = _format_float_for_script(1.0002, 1.0001)
    assert float(result1) != float(result2), f"Close values should be distinct: {result1} != {result2}"
    
    # Test exact values
    assert _format_float_for_script(1.0) == "1.0"
    assert _format_float_for_script(2.5) == "2.5"
    
    # Test None/NaN handling
    import pandas as pd
    import numpy as np
    assert "nan" in _format_float_for_script(np.nan).lower()
    assert "none" in _format_float_for_script(None).lower()


def test_comprehensive_script_formatting():
    """Test that all script types use clean float formatting."""
    import pandas as pd
    
    # Create DataFrame with messy floats
    df = pd.DataFrame({
        'messy_floats': [1.111111111, 2.222222222, 3.333333333],
        'messy_probs': [True, True, False],  # Will create 0.666666... probability
        'single_messy': [4.444444444, 4.444444444, 4.444444444]  # OneValueProfile
    })
    
    from profile_df import DataFrameProfile
    profile = DataFrameProfile.from_df(df)
    script = profile.generate_df_script(100)
    
    # Check that no long decimal numbers appear
    import re
    long_decimals = re.findall(r'\d+\.\d{7,}', script)
    assert len(long_decimals) == 0, f"Found long decimals: {long_decimals}"
    
    # Check that script executes without errors
    import random
    result = eval(script)
    assert isinstance(result, pd.DataFrame)
    assert result.shape == (100, 3)


def test_float_formatting_with_missing_values():
    """Test float formatting in profiles with missing values."""
    # FloatProfile with nulls
    profile_nulls = FloatProfile(missing_prob=0.3333333, min=1.111111, max=9.999999)
    script = profile_nulls.sample_script()
    
    # Should have clean formatting for both the range and probability
    assert "1.11" in script or "1.111" in script
    assert "10.0" in script or "9.999" in script or "10" in script
    assert "0.67" in script or "0.667" in script  # 1 - 0.3333 = 0.6667
    
    # OneValueProfile with nulls and messy float
    profile_one_nulls = OneValueProfile(missing_prob=0.2222222, value=2.718281828)
    script_one_nulls = profile_one_nulls.sample_script()
    
    # Should have clean value and clean probability
    assert "2.72" in script_one_nulls or "2.718" in script_one_nulls
    assert "0.78" in script_one_nulls or "0.777" in script_one_nulls  # 1 - 0.222 = 0.778
