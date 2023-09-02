import pandas as pd
import random
from datetime import date, timedelta
from importune import importunity
with importunity():
    from ..profile_df import (
        FloatProfile, IntProfile, BoolProfile, DateProfile, DateStrProfile,
        StringProfile, profile_column, DataFrameProfile
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
