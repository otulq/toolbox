import pandas as pd
from importune import importunity
with importunity():
    from ..profile_df import (
        IntStrProfile, FloatStrProfile, StringProfile, OneValueProfile,
        profile_column
    )


# ────────────────────────────────────────────────────────────────────────────
# IntStrProfile edge cases
# ────────────────────────────────────────────────────────────────────────────
def test_IntStrProfile_rejects_decimal_floats():
    """Test that IntStrProfile correctly rejects strings with decimal points."""
    decimal_cases = ["1.0", "42.5", "3.14159", "0.5", "-1.0", "+1.5", " 3.14 ", "1.", ".5"]
    
    for case in decimal_cases:
        assert not IntStrProfile.type_check(case), f"IntStrProfile should reject '{case}'"
        assert FloatStrProfile.type_check(case), f"FloatStrProfile should accept '{case}'"


def test_IntStrProfile_rejects_scientific_notation():
    """Test that IntStrProfile correctly rejects scientific notation."""
    scientific_cases = ["1e1", "1E1", "2.5e1", "1e-2", "1e+2", "1E-3"]
    
    for case in scientific_cases:
        assert not IntStrProfile.type_check(case), f"IntStrProfile should reject '{case}'"
        assert FloatStrProfile.type_check(case), f"FloatStrProfile should accept '{case}'"


def test_IntStrProfile_accepts_pure_integers():
    """Test that IntStrProfile correctly accepts pure integer strings."""
    integer_cases = ["1", "42", "999", "-5", "+10", " 7 ", "0", "123456789"]
    
    for case in integer_cases:
        assert IntStrProfile.type_check(case), f"IntStrProfile should accept '{case}'"
        assert not FloatStrProfile.type_check(case), f"FloatStrProfile should reject '{case}'"


def test_IntStrProfile_rejects_invalid_strings():
    """Test that IntStrProfile correctly rejects non-numeric strings."""
    invalid_cases = ["hello", "1.2.3", "not_a_number", "", "1a", "a1"]
    
    for case in invalid_cases:
        assert not IntStrProfile.type_check(case), f"IntStrProfile should reject '{case}'"
        assert not FloatStrProfile.type_check(case), f"FloatStrProfile should reject '{case}'"


def test_IntStrProfile_series_with_decimals():
    """Test that IntStrProfile rejects pandas Series containing decimal floats."""
    df_float = pd.DataFrame({"x": ["1.0", "2.0", "3.0"]})
    assert not IntStrProfile.type_check(df_float["x"])
    assert FloatStrProfile.type_check(df_float["x"])


def test_IntStrProfile_series_with_pure_ints():
    """Test that IntStrProfile accepts pandas Series containing pure integers."""
    df_int = pd.DataFrame({"x": ["1", "2", "3"]})
    assert IntStrProfile.type_check(df_int["x"])
    assert not FloatStrProfile.type_check(df_int["x"])


def test_IntStrProfile_series_with_scientific():
    """Test that IntStrProfile rejects pandas Series containing scientific notation."""
    df_sci = pd.DataFrame({"x": ["1e2", "2e1", "3e0"]})
    assert not IntStrProfile.type_check(df_sci["x"])
    assert FloatStrProfile.type_check(df_sci["x"])


# ────────────────────────────────────────────────────────────────────────────
# FloatStrProfile edge cases
# ────────────────────────────────────────────────────────────────────────────
def test_FloatStrProfile_requires_decimal_or_scientific():
    """Test that FloatStrProfile requires decimal point or scientific notation."""
    # Should accept - has decimal or scientific markers
    float_cases = ["1.0", "42.5", "1e1", "2.5e1", "1.", ".5", "1E-3"]
    for case in float_cases:
        assert FloatStrProfile.type_check(case), f"FloatStrProfile should accept '{case}'"
    
    # Should reject - no decimal or scientific markers
    int_cases = ["1", "42", "-5", "+10"]
    for case in int_cases:
        assert not FloatStrProfile.type_check(case), f"FloatStrProfile should reject '{case}'"


def test_FloatStrProfile_handles_edge_formats():
    """Test FloatStrProfile with edge case float formats."""
    edge_cases = [
        ("1.", True),           # Trailing decimal
        (".5", True),           # Leading decimal missing
        ("1e+2", True),         # Positive exponent
        ("1E-3", True),         # Uppercase scientific
        ("-1.0", True),         # Negative float
        ("+1.5", True),         # Positive sign float
        (" 3.14 ", True),       # Whitespace float
        (".", False),           # Just decimal point
    ]
    
    for case, expected in edge_cases:
        result = FloatStrProfile.type_check(case)
        assert result == expected, f"FloatStrProfile.type_check('{case}') should be {expected}, got {result}"


# ────────────────────────────────────────────────────────────────────────────
# profile_column integration tests
# ────────────────────────────────────────────────────────────────────────────
def test_profile_column_chooses_correct_numeric_str_type():
    """Test that profile_column chooses the right numeric string profile type."""
    # Should choose IntStrProfile
    df_int = pd.DataFrame({"col": ["1", "2", "3", "4"]})
    prof_int = profile_column(df_int, column="col")
    assert isinstance(prof_int, IntStrProfile)
    assert prof_int.min == 1
    assert prof_int.max == 4
    
    # Should choose FloatStrProfile  
    df_float = pd.DataFrame({"col": ["1.0", "2.0", "3.0", "4.0"]})
    prof_float = profile_column(df_float, column="col")
    assert isinstance(prof_float, FloatStrProfile)
    assert prof_float.min == 1.0
    assert prof_float.max == 4.0
    
    # Should choose FloatStrProfile for scientific
    df_sci = pd.DataFrame({"col": ["1e1", "2e1", "3e1"]})
    prof_sci = profile_column(df_sci, column="col")
    assert isinstance(prof_sci, FloatStrProfile)
    assert prof_sci.min == 10.0
    assert prof_sci.max == 30.0


def test_profile_column_zero_vs_zero_decimal():
    """Test that profile_column distinguishes between '0' and '0.0'."""
    # Integer zero
    df_int_zero = pd.DataFrame({"col": ["0", "1", "2"]})
    prof_int_zero = profile_column(df_int_zero, column="col")
    assert isinstance(prof_int_zero, IntStrProfile)
    
    # Float zero
    df_float_zero = pd.DataFrame({"col": ["0.0", "1.0", "2.0"]})
    prof_float_zero = profile_column(df_float_zero, column="col")
    assert isinstance(prof_float_zero, FloatStrProfile)


def test_profile_column_mixed_numeric_int_float():
    """Test that mixed int/float strings become FloatStrProfile (not StringProfile)."""
    # This is the key case we fixed - mixed numeric should be treated as numeric range
    df_mixed_numeric = pd.DataFrame({"col": ["23.14", "42", "100", "12.123"]})
    prof_mixed_numeric = profile_column(df_mixed_numeric, column="col")
    assert isinstance(prof_mixed_numeric, FloatStrProfile)
    assert prof_mixed_numeric.min == 12.123
    assert prof_mixed_numeric.max == 100.0
    # Should generate uniform range, not categorical choice
    script = prof_mixed_numeric.sample_script()
    assert "random.uniform" in script
    assert "random.choice" not in script


def test_profile_column_mixed_detection_edge_cases():
    """Test edge cases for mixed numeric detection."""
    # Pure ints should remain IntStrProfile (no mixed treatment)
    df_pure_int = pd.DataFrame({"col": ["42", "100", "5"]})
    prof_pure_int = profile_column(df_pure_int, column="col")
    assert isinstance(prof_pure_int, IntStrProfile)
    assert prof_pure_int.min == 5
    assert prof_pure_int.max == 100
    script = prof_pure_int.sample_script()
    assert "random.randint" in script
    
    # Pure floats should remain FloatStrProfile (no mixed treatment)  
    df_pure_float = pd.DataFrame({"col": ["23.14", "2341.123"]})
    prof_pure_float = profile_column(df_pure_float, column="col")
    assert isinstance(prof_pure_float, FloatStrProfile)
    assert prof_pure_float.min == 23.14
    assert prof_pure_float.max == 2341.123
    script = prof_pure_float.sample_script()
    assert "random.uniform" in script


def test_profile_column_mixed_data_fallback():
    """Test that mixed numeric/text data falls back to StringProfile."""
    df_mixed = pd.DataFrame({"col": ["1", "hello", "2.5", "world"]})
    prof_mixed = profile_column(df_mixed, column="col")
    assert isinstance(prof_mixed, StringProfile)
    assert set(prof_mixed.values) == {"1", "2.5", "hello", "world"}


def test_profile_column_single_value_numeric_str():
    """Test that single numeric string values use OneValueProfile."""
    # Single int string
    df_single_int = pd.DataFrame({"col": ["42", "42", "42"]})
    prof_single_int = profile_column(df_single_int, column="col")
    assert isinstance(prof_single_int, OneValueProfile)
    assert prof_single_int.value == "42"
    
    # Single float string
    df_single_float = pd.DataFrame({"col": ["1.5", "1.5", "1.5"]})
    prof_single_float = profile_column(df_single_float, column="col")
    assert isinstance(prof_single_float, OneValueProfile)
    assert prof_single_float.value == "1.5"


# ────────────────────────────────────────────────────────────────────────────
# Cross-validation tests
# ────────────────────────────────────────────────────────────────────────────
def test_mutually_exclusive_classification():
    """Test that IntStrProfile and FloatStrProfile are mutually exclusive."""
    test_cases = [
        "1", "42", "-5", "+10",         # Should be int only
        "1.0", "42.5", "1e1", "2.5e1",  # Should be float only
        "hello", "1.2.3", "",           # Should be neither
    ]
    
    for case in test_cases:
        int_result = IntStrProfile.type_check(case)
        float_result = FloatStrProfile.type_check(case)
        
        # They should never both be True
        assert not (int_result and float_result), f"Both profiles accepted '{case}'"
        
        # Verify specific expectations
        if case in ["1", "42", "-5", "+10"]:
            assert int_result and not float_result, f"'{case}' should be int only"
        elif case in ["1.0", "42.5", "1e1", "2.5e1"]:
            assert float_result and not int_result, f"'{case}' should be float only"
        elif case in ["hello", "1.2.3", ""]:
            assert not int_result and not float_result, f"'{case}' should be neither"


def test_numeric_str_profile_ordering_in_detection():
    """Test that numeric string detection happens before other string checks."""
    # This should be detected as IntStrProfile, not caught by date/datetime string checks
    df_pure_nums = pd.DataFrame({"col": ["1", "2", "3", "10", "20"]})
    prof = profile_column(df_pure_nums, column="col")
    assert isinstance(prof, IntStrProfile)
    
    # This should be detected as FloatStrProfile
    df_float_nums = pd.DataFrame({"col": ["1.1", "2.2", "3.3"]})
    prof = profile_column(df_float_nums, column="col")
    assert isinstance(prof, FloatStrProfile)
    
    # Verify the detection happens early enough to avoid date parsing attempts
    df_ambiguous = pd.DataFrame({"col": ["1", "2", "11", "12"]})  # Could look like months
    prof = profile_column(df_ambiguous, column="col")
    assert isinstance(prof, IntStrProfile)  # Should be numeric, not date 