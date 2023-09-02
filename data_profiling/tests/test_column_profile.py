import pandas as pd
from datetime import date, datetime
from importune import importunity
with importunity():
    from ..profile_df import (
        FloatProfile, IntProfile, FloatStrProfile, IntStrProfile, BoolProfile, DateProfile, DateStrProfile,
        DatetimeProfile, DatetimeStrProfile, StringProfile, NullProfile, OneValueProfile, 
        profile_column, DataFrameProfile
    )

# ────────────────────────────────────────────────────────────────────────────
# basic helpers
# ────────────────────────────────────────────────────────────────────────────
def _non_none(samples):
    return [x for x in samples if x is not None]

# ────────────────────────────────────────────────────────────────────────────
# FloatProfile
# ────────────────────────────────────────────────────────────────────────────
def test_FloatProfile_sample():
    prof = FloatProfile(missing_prob=0.0, min=5.0, max=10.0)
    vals = _non_none(prof.samples(100))
    assert all(5.0 <= v <= 10.0 for v in vals)

# ────────────────────────────────────────────────────────────────────────────
# IntProfile
# ────────────────────────────────────────────────────────────────────────────
def test_IntProfile_sample():
    prof = IntProfile(missing_prob=0.0, min=1, max=3)
    vals = _non_none(prof.samples(100))
    assert set(vals).issubset({1, 2, 3})

# ────────────────────────────────────────────────────────────────────────────
# BoolProfile
# ────────────────────────────────────────────────────────────────────────────
def test_BoolProfile_sample():
    prof = BoolProfile(missing_prob=0.0, true_prob=1.0)
    assert all(prof.samples(20))  # always True

# ────────────────────────────────────────────────────────────────────────────
# DateProfile
# ────────────────────────────────────────────────────────────────────────────
def test_DateProfile_sample():
    d0, d1 = date(2020, 1, 1), date(2020, 12, 31)
    prof = DateProfile(missing_prob=0.0, min=d0, max=d1)
    vals = _non_none(prof.samples(50))
    assert all(d0 <= v <= d1 for v in vals)

# ────────────────────────────────────────────────────────────────────────────
# StringProfile
# ────────────────────────────────────────────────────────────────────────────
def test_StringProfile_sample():
    prof = StringProfile(missing_prob=0.0, values=["A", "B"])
    assert set(_non_none(prof.samples(50))).issubset({"A", "B"})

# ────────────────────────────────────────────────────────────────────────────
# NullProfile
# ────────────────────────────────────────────────────────────────────────────
def test_NullProfile_sample():
    prof = NullProfile()
    assert prof.sample() is None
    assert all(x is None for x in prof.samples(20))

def test_NullProfile_has_nulls():
    prof = NullProfile()
    assert prof.has_nulls() is True
    assert prof.missing_prob == 1.0

def test_NullProfile_sample_script():
    prof = NullProfile()
    assert prof.sample_script() == "None"
    assert prof.sample_script(ignore_missing=True) == "None"

def test_NullProfile_contains():
    prof = NullProfile()
    assert None in prof
    assert 42 not in prof
    assert "hello" not in prof

# ────────────────────────────────────────────────────────────────────────────
# OneValueProfile
# ────────────────────────────────────────────────────────────────────────────
def test_OneValueProfile_sample_no_missing():
    prof = OneValueProfile(missing_prob=0.0, value=42)
    vals = prof.samples(20)
    assert all(x == 42 for x in vals)

def test_OneValueProfile_sample_with_missing():
    prof = OneValueProfile(missing_prob=0.8, value="hello")
    vals = prof.samples(100)
    non_null_vals = _non_none(vals)
    null_vals = [x for x in vals if x is None]
    
    # Should have some nulls due to high missing probability
    assert len(null_vals) > 0
    # All non-null values should be "hello"
    assert all(x == "hello" for x in non_null_vals)

def test_OneValueProfile_sample_script_string():
    prof = OneValueProfile(missing_prob=0.0, value="test")
    assert prof.sample_script() == "'test'"
    
    prof_with_missing = OneValueProfile(missing_prob=0.3, value="test")
    script = prof_with_missing.sample_script()
    assert "'test'" in script
    assert "random.random()" in script
    assert "else None" in script

def test_OneValueProfile_sample_script_numeric():
    prof = OneValueProfile(missing_prob=0.0, value=42)
    assert prof.sample_script() == "42"
    
    prof_float = OneValueProfile(missing_prob=0.0, value=3.14)
    assert prof_float.sample_script() == "3.14"

def test_OneValueProfile_sample_script_bool():
    prof_true = OneValueProfile(missing_prob=0.0, value=True)
    assert prof_true.sample_script() == "True"
    
    prof_false = OneValueProfile(missing_prob=0.0, value=False)
    assert prof_false.sample_script() == "False"

def test_OneValueProfile_sample_script_date():
    test_date = date(2023, 9, 2)
    prof = OneValueProfile(missing_prob=0.0, value=test_date)
    script = prof.sample_script()
    assert "date.fromisoformat('2023-09-02')" in script


# ────────────────────────────────────────────────────────────────────────────
# DatetimeProfile
# ────────────────────────────────────────────────────────────────────────────
def test_DatetimeProfile_sample():
    dt0 = datetime(2020, 1, 1, 10, 30, 0)
    dt1 = datetime(2020, 12, 31, 15, 45, 30)
    prof = DatetimeProfile(missing_prob=0.0, min=dt0, max=dt1)
    vals = _non_none(prof.samples(50))
    assert all(dt0 <= v <= dt1 for v in vals)
    assert all(isinstance(v, datetime) for v in vals)


def test_DatetimeProfile_sample_script():
    dt0 = datetime(2023, 9, 2, 8, 3, 0)
    dt1 = datetime(2023, 9, 2, 10, 2, 32)
    prof = DatetimeProfile(missing_prob=0.0, min=dt0, max=dt1)
    script = prof.sample_script()
    assert "datetime.fromisoformat" in script
    assert "2023-09-02T08:03:00" in script
    assert "2023-09-02T10:02:32" in script


def test_DatetimeProfile_type_check():
    # Test individual datetime objects
    assert DatetimeProfile.type_check(datetime.now())
    assert not DatetimeProfile.type_check(date.today())
    assert not DatetimeProfile.type_check("2023-01-01")
    
    # Test pandas datetime64 series
    df_dt64 = pd.DataFrame({"x": pd.to_datetime(["2023-01-01 10:30:00", "2023-01-02 15:45:30"])})
    assert DatetimeProfile.type_check(df_dt64["x"])
    
    # Test object series with datetime objects
    df_obj = pd.DataFrame({"x": [datetime(2023, 1, 1, 10, 30), datetime(2023, 1, 2, 15, 45)]})
    assert DatetimeProfile.type_check(df_obj["x"])


# ────────────────────────────────────────────────────────────────────────────
# DatetimeStrProfile
# ────────────────────────────────────────────────────────────────────────────
def test_DatetimeStrProfile_sample():
    dt_min = "2020-01-01T10:30:00"
    dt_max = "2020-12-31T15:45:30"
    prof = DatetimeStrProfile(missing_prob=0.0, min=dt_min, max=dt_max)
    vals = _non_none(prof.samples(50))
    assert all(isinstance(v, str) for v in vals)
    # Check that all samples can be parsed as datetimes
    parsed_vals = [datetime.fromisoformat(v) for v in vals]
    dt0 = datetime.fromisoformat(dt_min)
    dt1 = datetime.fromisoformat(dt_max)
    assert all(dt0 <= pv <= dt1 for pv in parsed_vals)


def test_DatetimeStrProfile_sample_script():
    dt_min = "2023-09-02T08:03:00"
    dt_max = "2023-09-02T10:02:32"
    prof = DatetimeStrProfile(missing_prob=0.0, min=dt_min, max=dt_max)
    script = prof.sample_script()
    assert "datetime.fromisoformat" in script
    assert "2023-09-02T08:03:00" in script
    assert "2023-09-02T10:02:32" in script
    assert ".isoformat()" in script


def test_DatetimeStrProfile_type_check():
    # Test datetime strings with time components
    assert DatetimeStrProfile.type_check("2023-01-01T10:30:00")
    assert DatetimeStrProfile.type_check("2023-01-01 10:30:00")
    assert not DatetimeStrProfile.type_check("2023-01-01")  # Date only, no time
    assert not DatetimeStrProfile.type_check(datetime.now())  # Actual datetime object
    
    # Test pandas series of datetime strings
    df_dt_str = pd.DataFrame({"x": ["2023-01-01T10:30:00", "2023-01-02T15:45:30"]})
    assert DatetimeStrProfile.type_check(df_dt_str["x"])
    
    # Test that date-only strings don't match
    df_date_str = pd.DataFrame({"x": ["2023-01-01", "2023-01-02"]})
    assert not DatetimeStrProfile.type_check(df_date_str["x"])


def test_OneValueProfile_sample_script_datetime():
    test_datetime = datetime(2023, 9, 2, 8, 3, 0)
    prof = OneValueProfile(missing_prob=0.0, value=test_datetime)
    script = prof.sample_script()
    assert "datetime.fromisoformat('2023-09-02T08:03:00')" in script


# ────────────────────────────────────────────────────────────────────────────
# FloatStrProfile
# ────────────────────────────────────────────────────────────────────────────
def test_FloatStrProfile_sample():
    prof = FloatStrProfile(missing_prob=0.0, min=1.5, max=9.7)
    vals = _non_none(prof.samples(50))
    assert all(isinstance(v, str) for v in vals)
    # Convert to float to check range
    float_vals = [float(v) for v in vals]
    assert all(1.5 <= fv <= 9.7 for fv in float_vals)


def test_FloatStrProfile_sample_script():
    prof = FloatStrProfile(missing_prob=0.0, min=1.5, max=9.7)
    script = prof.sample_script()
    assert "str(random.uniform(" in script
    assert "1.5" in script
    assert "9.7" in script


def test_FloatStrProfile_sample_script_with_missing():
    prof = FloatStrProfile(missing_prob=0.3, min=1.0, max=5.0)
    script = prof.sample_script()
    assert "str(random.uniform(" in script
    assert "if random.random() <" in script
    assert "else None" in script


def test_FloatStrProfile_type_check():
    # Test individual float strings
    assert FloatStrProfile.type_check("1.5")
    assert FloatStrProfile.type_check("3.14159")
    assert FloatStrProfile.type_check("1e10")
    assert not FloatStrProfile.type_check("1")  # Integer string
    assert not FloatStrProfile.type_check("hello")  # Non-numeric string
    assert not FloatStrProfile.type_check(1.5)  # Actual float
    
    # Test pandas series of float strings
    df_float_str = pd.DataFrame({"x": ["1.5", "2.3", "4.7", "1.2"]})
    assert FloatStrProfile.type_check(df_float_str["x"])
    
    # Test that integer strings don't match
    df_int_str = pd.DataFrame({"x": ["1", "2", "3"]})
    assert not FloatStrProfile.type_check(df_int_str["x"])


def test_FloatStrProfile_contains():
    prof = FloatStrProfile(missing_prob=0.0, min=1.0, max=5.0)
    assert prof.contains("3.5")
    assert prof.contains("1.0")
    assert prof.contains("5.0")
    assert not prof.contains("0.5")  # Below min
    assert not prof.contains("6.0")  # Above max
    assert not prof.contains("hello")  # Non-numeric


# ────────────────────────────────────────────────────────────────────────────
# IntStrProfile
# ────────────────────────────────────────────────────────────────────────────
def test_IntStrProfile_sample():
    prof = IntStrProfile(missing_prob=0.0, min=10, max=50)
    vals = _non_none(prof.samples(50))
    assert all(isinstance(v, str) for v in vals)
    # Convert to int to check range
    int_vals = [int(v) for v in vals]
    assert all(10 <= iv <= 50 for iv in int_vals)


def test_IntStrProfile_sample_script():
    prof = IntStrProfile(missing_prob=0.0, min=10, max=50)
    script = prof.sample_script()
    assert "str(random.randint(" in script
    assert "10" in script
    assert "50" in script


def test_IntStrProfile_sample_script_with_missing():
    prof = IntStrProfile(missing_prob=0.2, min=1, max=10)
    script = prof.sample_script()
    assert "str(random.randint(" in script
    assert "if random.random() <" in script
    assert "else None" in script


def test_IntStrProfile_type_check():
    # Test individual int strings
    assert IntStrProfile.type_check("1")
    assert IntStrProfile.type_check("42")
    assert IntStrProfile.type_check("999")
    assert not IntStrProfile.type_check("1.5")  # Float string
    assert not IntStrProfile.type_check("hello")  # Non-numeric string
    assert not IntStrProfile.type_check(42)  # Actual int
    
    # Test pandas series of int strings
    df_int_str = pd.DataFrame({"x": ["1", "42", "999"]})
    assert IntStrProfile.type_check(df_int_str["x"])
    
    # Test that float strings don't match
    df_float_str = pd.DataFrame({"x": ["1.5", "2.3", "4.7"]})
    assert not IntStrProfile.type_check(df_float_str["x"])


def test_IntStrProfile_contains():
    prof = IntStrProfile(missing_prob=0.0, min=10, max=50)
    assert prof.contains("25")
    assert prof.contains("10")
    assert prof.contains("50")
    assert not prof.contains("5")  # Below min
    assert not prof.contains("55")  # Above max
    assert not prof.contains("hello")  # Non-numeric

def test_OneValueProfile_contains():
    prof = OneValueProfile(missing_prob=0.2, value=42)
    assert 42 in prof
    assert 43 not in prof
    assert None in prof  # because missing_prob > 0

def test_OneValueProfile_contains_no_missing():
    prof = OneValueProfile(missing_prob=0.0, value="hello")
    assert "hello" in prof
    assert "world" not in prof
    assert None not in prof  # because missing_prob == 0

# ────────────────────────────────────────────────────────────────────────────
# profile_column – numeric inference
# ────────────────────────────────────────────────────────────────────────────
def test_profile_column_numeric():
    df = pd.DataFrame({"x": [1, 2, 3, None]})
    prof = profile_column(df, column="x")
    assert isinstance(prof, FloatProfile)  # pandas upcasts to float with nulls
    assert prof.min == 1.2 and prof.max == 2.8  # 10th and 90th percentiles
    assert prof.missing_prob == 0.25

# ────────────────────────────────────────────────────────────────────────────
# profile_column – null column inference
# ────────────────────────────────────────────────────────────────────────────
def test_profile_column_all_null():
    df = pd.DataFrame({"x": [None, None, None, None]})
    prof = profile_column(df, column="x")
    assert isinstance(prof, NullProfile)
    assert prof.missing_prob == 1.0

def test_profile_column_all_null_mixed_types():
    df = pd.DataFrame({"x": [None, pd.NaT, pd.NA, None]})
    prof = profile_column(df, column="x")
    assert isinstance(prof, NullProfile)
    assert prof.missing_prob == 1.0

# ────────────────────────────────────────────────────────────────────────────
# profile_column – single value inference
# ────────────────────────────────────────────────────────────────────────────
def test_profile_column_single_value_no_nulls():
    df = pd.DataFrame({"x": [42, 42, 42, 42]})
    prof = profile_column(df, column="x")
    assert isinstance(prof, OneValueProfile)
    assert prof.value == 42
    assert prof.missing_prob == 0.0

def test_profile_column_single_value_with_nulls():
    df = pd.DataFrame({"x": ["hello", "hello", None, "hello"]})
    prof = profile_column(df, column="x")
    assert isinstance(prof, OneValueProfile)
    assert prof.value == "hello"
    assert prof.missing_prob == 0.25

def test_profile_column_single_value_bool():
    df = pd.DataFrame({"x": [True, True, True, None]})
    prof = profile_column(df, column="x")
    assert isinstance(prof, OneValueProfile)
    assert prof.value is True
    assert prof.missing_prob == 0.25

def test_profile_column_single_value_date():
    test_date = date(2023, 9, 2)
    df = pd.DataFrame({"x": [test_date, test_date, test_date]})
    prof = profile_column(df, column="x")
    assert isinstance(prof, OneValueProfile)
    assert prof.value == test_date
    assert prof.missing_prob == 0.0

def test_profile_column_quantile_edge_case():
    """Test when quantiles create min=max, should return OneValueProfile not FloatProfile."""
    # Create data where tight quantiles will result in min=max
    df = pd.DataFrame({"x": [0.0] * 100 + [0.1]})  # 99% zeros
    prof = profile_column(df, column="x", q_low=0.01, q_high=0.01)
    assert isinstance(prof, OneValueProfile)
    assert prof.value == 0.0
    assert prof.missing_prob == 0.0
    
    # Should not generate random.uniform(0.0, 0.0)
    script = prof.sample_script()
    assert script == "0.0"
    assert "random.uniform" not in script

def test_profile_column_numpy_types():
    """Test that numpy types are converted to native Python types in OneValueProfile."""
    import numpy as np
    
    # Test numpy int64
    df_int = pd.DataFrame({"x": [np.int64(42), np.int64(42), np.int64(42)]})
    prof_int = profile_column(df_int, column="x")
    assert isinstance(prof_int, OneValueProfile)
    assert type(prof_int.value) is int  # Native Python int, not numpy.int64
    assert prof_int.value == 42
    assert prof_int.sample_script() == "42"  # Should not error
    
    # Test numpy float64
    df_float = pd.DataFrame({"x": [np.float64(3.14), np.float64(3.14), np.float64(3.14)]})
    prof_float = profile_column(df_float, column="x")
    assert isinstance(prof_float, OneValueProfile)
    assert type(prof_float.value) is float  # Native Python float, not numpy.float64
    assert prof_float.value == 3.14
    assert prof_float.sample_script() == "3.14"  # Should not error
    
    # Test numpy bool_
    df_bool = pd.DataFrame({"x": [np.bool_(True), np.bool_(True), np.bool_(True)]})
    prof_bool = profile_column(df_bool, column="x")
    assert isinstance(prof_bool, OneValueProfile)
    assert type(prof_bool.value) is bool  # Native Python bool, not numpy.bool_
    assert prof_bool.value is True
    assert prof_bool.sample_script() == "True"  # Should not error

def test_profile_column_none_with_single_value():
    """Test that None values mixed with single unique value creates OneValueProfile with the non-null value."""
    # Test None + string (was causing "Invalid value type: NoneType" error)
    df1 = pd.DataFrame({"x": [None, None, None, "hello"]})
    prof1 = profile_column(df1, column="x")
    assert isinstance(prof1, OneValueProfile)
    assert prof1.value == "hello"  # Should be the non-null value, not None
    assert prof1.missing_prob == 0.75
    # Should generate valid script without error
    script = prof1.sample_script()
    assert "'hello'" in script
    assert "random.random()" in script
    assert "else None" in script
    
    # Test None + int
    df2 = pd.DataFrame({"x": [None, 42, None, 42]})
    prof2 = profile_column(df2, column="x")
    assert isinstance(prof2, OneValueProfile)
    assert prof2.value == 42.0  # Pandas converts to float when mixed with None
    assert prof2.missing_prob == 0.5
    script2 = prof2.sample_script()
    assert "42.0" in script2
    
    # Test None first
    df3 = pd.DataFrame({"x": [None, "test", "test", "test"]})
    prof3 = profile_column(df3, column="x")
    assert isinstance(prof3, OneValueProfile)
    assert prof3.value == "test"  # Should be the non-null value
    assert prof3.missing_prob == 0.25

def test_profile_one_value_int():
    df = pd.DataFrame({"x": [1, 1, 1, None]})
    prof = profile_column(df, column="x")
    assert isinstance(prof, OneValueProfile)
    assert prof.value == 1
    assert prof.missing_prob == 0.25

# ────────────────────────────────────────────────────────────────────────────
# profile_column – datetime inference
# ────────────────────────────────────────────────────────────────────────────
def test_profile_column_datetime():
    # Test pandas datetime64 column
    df = pd.DataFrame({"dt": pd.to_datetime(["2023-01-01 10:30:00", "2023-01-02 15:45:30", None])})
    prof = profile_column(df, column="dt")
    assert isinstance(prof, DatetimeProfile)
    assert prof.missing_prob == 0.33


def test_profile_column_datetime_objects():
    # Test column with Python datetime objects
    dt1 = datetime(2023, 1, 1, 10, 30, 0)
    dt2 = datetime(2023, 1, 2, 15, 45, 30)
    df = pd.DataFrame({"dt": [dt1, dt2, dt1]})
    prof = profile_column(df, column="dt")
    assert isinstance(prof, DatetimeProfile)
    assert prof.missing_prob == 0.0
    assert prof.min == dt1
    assert prof.max == dt2


def test_profile_column_datetime_str():
    # Test datetime strings with time components
    df = pd.DataFrame({"dt_str": ["2023-01-01T10:30:00", "2023-01-02T15:45:30", None]})
    prof = profile_column(df, column="dt_str")
    assert isinstance(prof, DatetimeStrProfile)
    assert prof.min == "2023-01-01T10:30:00"
    assert prof.max == "2023-01-02T15:45:30"
    assert prof.missing_prob == 0.33


def test_profile_column_datetime_str_space_format():
    # Test datetime strings with space separator
    df = pd.DataFrame({"dt_str": ["2023-01-01 10:30:00", "2023-01-02 15:45:30"]})
    prof = profile_column(df, column="dt_str")
    assert isinstance(prof, DatetimeStrProfile)


def test_profile_column_single_value_datetime():
    test_datetime = datetime(2023, 9, 2, 8, 3, 0)
    df = pd.DataFrame({"x": [test_datetime, test_datetime, test_datetime]})
    prof = profile_column(df, column="x")
    assert isinstance(prof, OneValueProfile)
    assert prof.value == test_datetime
    assert prof.missing_prob == 0.0


# ────────────────────────────────────────────────────────────────────────────
# profile_column – numeric string inference
# ────────────────────────────────────────────────────────────────────────────
def test_profile_column_int_str():
    # Test column with integer strings
    df = pd.DataFrame({"int_str": ["1", "42", "15", None]})
    prof = profile_column(df, column="int_str")
    assert isinstance(prof, IntStrProfile)
    assert prof.min == 1
    assert prof.max == 42
    assert prof.missing_prob == 0.25


def test_profile_column_float_str():
    # Test column with float strings
    df = pd.DataFrame({"float_str": ["1.5", "2.3", "4.7", None]})
    prof = profile_column(df, column="float_str")
    assert isinstance(prof, FloatStrProfile)
    assert prof.min == 1.5
    assert prof.max == 4.7
    assert prof.missing_prob == 0.25


def test_profile_column_float_str_scientific():
    # Test column with scientific notation
    df = pd.DataFrame({"sci_str": ["1e2", "2.5e1", "3.14e0"]})
    prof = profile_column(df, column="sci_str")
    assert isinstance(prof, FloatStrProfile)
    assert prof.min == 3.14
    assert prof.max == 100.0
    assert prof.missing_prob == 0.0


def test_profile_column_numeric_str_vs_categorical():
    # Test that mixed numeric/non-numeric strings fall back to StringProfile
    df = pd.DataFrame({"mixed": ["1", "hello", "2.5", "world"]})
    prof = profile_column(df, column="mixed")
    assert isinstance(prof, StringProfile)
    assert set(prof.values) == {"1", "2.5", "hello", "world"}


def test_profile_column_int_str_vs_float_str():
    # Test that integer strings are detected as IntStrProfile, not FloatStrProfile
    df = pd.DataFrame({"pure_ints": ["1", "2", "3", "42"]})
    prof = profile_column(df, column="pure_ints")
    assert isinstance(prof, IntStrProfile)
    assert not isinstance(prof, FloatStrProfile)


def test_profile_column_single_value_numeric_str():
    # Test single value numeric strings
    df = pd.DataFrame({"single_int": ["42", "42", "42"]})
    prof = profile_column(df, column="single_int")
    assert isinstance(prof, OneValueProfile)
    assert prof.value == "42"


# ────────────────────────────────────────────────────────────────────────────
# profile_column – date-string inference
# ────────────────────────────────────────────────────────────────────────────
def test_profile_column_date_str():
    df = pd.DataFrame({"d": ["2022-01-01", None, "2022-02-01"]})
    prof = profile_column(df, column="d")
    assert isinstance(prof, DateStrProfile)
    assert prof.min == "2022-01-01"
    assert prof.max == "2022-02-01"

# ────────────────────────────────────────────────────────────────────────────
# DataFrameProfile end-to-end
# ────────────────────────────────────────────────────────────────────────────
def test_DataFrameProfile_generate_df():
    df = pd.DataFrame({
        'int_col': [1, 2, 3, 4, 5],
        'int_col_with_nulls_is_float': [1, 2, 3, None, 5],
        'float_col': [1.0, 2.0, 3.0, 4.0, 5.0],
        'float_col_with_nulls': [1.0, 2.0, 3.0, None, 5.0],
        'date_col': [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3), date(2022, 1, 4), date(2022, 1, 5)],
        'date_col_with_nulls': [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3), None, date(2022, 1, 5)],
        'date_str_col': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05'],
        'date_str_col_with_nulls': ['2022-01-01', '2022-01-02', '2022-01-03', None, '2022-01-05'],
        'string_col': ['aaa', 'bbb', 'ccc', 'ddd', 'eee'],
        'string_col_with_nulls': ['aaa', 'bbb', 'ccc', None, 'eee'],
        'bool_col': [True, False, True, False, True],
        'bool_col_with_nulls': [True, False, True, None, True],
    })
    dp = DataFrameProfile.from_df(df)
    synth = dp.generate_df(5)
    assert synth.shape == (5, 12)
    # every column should come back with nullable values
    assert set(synth.columns) == {"int_col", "int_col_with_nulls_is_float", "float_col", "float_col_with_nulls", "date_col", "date_col_with_nulls", "date_str_col", "date_str_col_with_nulls", "string_col", "string_col_with_nulls", "bool_col", "bool_col_with_nulls"}

def test_DataFrameProfile_generate_df_and_check_values_are_in_range():
    df = pd.DataFrame({
        'int_col': [1, 2, 3, 4, 5],
        'int_col_with_nulls_is_float': [1, 2, 3, None, 5],
        'float_col': [1.0, 2.0, 3.0, 4.0, 5.0],
        'float_col_with_nulls': [1.0, 2.0, 3.0, None, 5.0],
        'date_col': [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3), date(2022, 1, 4), date(2022, 1, 5)],
        'date_col_with_nulls': [date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3), None, date(2022, 1, 5)],
        'date_str_col': ['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05'],
        'date_str_col_with_nulls': ['2022-01-01', '2022-01-02', '2022-01-03', None, '2022-01-05'],
        'string_col': ['aaa', 'bbb', 'ccc', 'ddd', 'eee'],
        'string_col_with_nulls': ['aaa', 'bbb', 'ccc', None, 'eee'],
        'bool_col': [True, False, True, False, True],
        'bool_col_with_nulls': [True, False, True, None, True],
    })
    dp = DataFrameProfile.from_df(df)
    assert isinstance(dp['int_col'], IntProfile)
    assert isinstance(dp['int_col_with_nulls_is_float'], FloatProfile)
    assert isinstance(dp['float_col'], FloatProfile)
    assert isinstance(dp['float_col_with_nulls'], FloatProfile)
    assert isinstance(dp['date_col'], DateProfile) # if dates (not strings) are in the column, it's a DateProfile and not DateStrProfile
    assert isinstance(dp['date_col_with_nulls'], DateProfile)
    assert isinstance(dp['date_str_col'], DateStrProfile)
    assert isinstance(dp['date_str_col_with_nulls'], DateStrProfile)
    synth = dp.generate_df(5)
    assert synth.shape == (5, 12)
    # Check that generated values are within the expected ranges
    assert all(1 <= x <= 5 for x in synth['int_col'].dropna())
    assert all(1.3 <= x <= 4.4 for x in synth['int_col_with_nulls_is_float'].dropna())  # quantile range
    assert all(1.0 <= x <= 5.0 for x in synth['float_col'].dropna())
    assert all(1.0 <= x <= 5.0 for x in synth['float_col_with_nulls'].dropna())

    # dates are dates, not strings, therefore they are not in the DateStrProfile, they are in the DateProfile
    assert synth['date_col'].isin([date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3), date(2022, 1, 4), date(2022, 1, 5)]).all()
    assert synth['date_col_with_nulls'].isin([date(2022, 1, 1), date(2022, 1, 2), date(2022, 1, 3), date(2022, 1, 4), date(2022, 1, 5), None]).all()
    assert synth['date_str_col'].isin(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05']).all()
    assert synth['date_str_col_with_nulls'].isin(['2022-01-01', '2022-01-02', '2022-01-03', '2022-01-04', '2022-01-05', None]).all()
    assert synth['string_col'].isin(['aaa', 'bbb', 'ccc', 'ddd', 'eee']).all()
    assert synth['string_col_with_nulls'].isin(['aaa', 'bbb', 'ccc', 'ddd', 'eee', None]).all()
    assert synth['bool_col'].isin([True, False, True, False, True]).all()
    assert (synth['bool_col_with_nulls'].isin([True, False]) | synth['bool_col_with_nulls'].isna()).all()

    # test using type_check
    assert IntProfile.type_check(synth['int_col'])
    assert FloatProfile.type_check(synth['int_col_with_nulls_is_float'])
    assert FloatProfile.type_check(synth['float_col'])
    assert FloatProfile.type_check(synth['float_col_with_nulls'])
    assert DateProfile.type_check(synth['date_col'])
    assert DateProfile.type_check(synth['date_col_with_nulls'])
    assert DateStrProfile.type_check(synth['date_str_col'])
    assert DateStrProfile.type_check(synth['date_str_col_with_nulls'])
    assert StringProfile.type_check(synth['string_col'])
    assert StringProfile.type_check(synth['string_col_with_nulls'])

    # test type_check that should return False
    assert not IntProfile.type_check(synth['date_str_col'])
    assert not IntProfile.type_check(synth['date_str_col_with_nulls'])
    assert not IntProfile.type_check(synth['string_col'])
    assert not IntProfile.type_check(synth['string_col_with_nulls'])
    assert not IntProfile.type_check(synth['bool_col'])
    assert not IntProfile.type_check(synth['bool_col_with_nulls'])

    assert not FloatProfile.type_check(synth['date_col'])
    assert not FloatProfile.type_check(synth['date_col_with_nulls'])
    assert not FloatProfile.type_check(synth['date_str_col'])
    assert not FloatProfile.type_check(synth['date_str_col_with_nulls'])
    assert not FloatProfile.type_check(synth['string_col'])
    assert not FloatProfile.type_check(synth['string_col_with_nulls'])
    assert not FloatProfile.type_check(synth['bool_col'])
    assert not FloatProfile.type_check(synth['bool_col_with_nulls'])

    assert not DateProfile.type_check(synth['int_col'])
    assert not DateProfile.type_check(synth['int_col_with_nulls_is_float'])
    assert not DateProfile.type_check(synth['float_col'])
    assert not DateProfile.type_check(synth['float_col_with_nulls'])
    assert not DateProfile.type_check(synth['date_str_col'])
    assert not DateProfile.type_check(synth['date_str_col_with_nulls'])
    assert not DateProfile.type_check(synth['string_col'])
    assert not DateProfile.type_check(synth['string_col_with_nulls'])
    assert not DateProfile.type_check(synth['bool_col'])
    assert not DateProfile.type_check(synth['bool_col_with_nulls'])

    assert not DateStrProfile.type_check(synth['int_col'])
    assert not DateStrProfile.type_check(synth['int_col_with_nulls_is_float'])
    assert not DateStrProfile.type_check(synth['float_col'])
    assert not DateStrProfile.type_check(synth['float_col_with_nulls'])
    assert not DateStrProfile.type_check(synth['date_col'])
    assert not DateStrProfile.type_check(synth['date_col_with_nulls'])
    assert not DateStrProfile.type_check(synth['bool_col'])
    assert not DateStrProfile.type_check(synth['bool_col_with_nulls'])

# ────────────────────────────────────────────────────────────────────────────
# Decimal support tests
# ────────────────────────────────────────────────────────────────────────────
def test_profile_column_decimal_basic():
    """Test that decimal.Decimal values are profiled as FloatProfile."""
    from decimal import Decimal
    
    df = pd.DataFrame({
        "decimal_col": [Decimal('10.50'), Decimal('20.75'), Decimal('15.25')]
    })
    prof = profile_column(df, column="decimal_col")
    
    assert isinstance(prof, FloatProfile)
    assert prof.missing_prob == 0.0
    # Should use quantile-based range, not exact min/max
    assert prof.min < 20.75
    assert prof.max > 10.50


def test_profile_column_decimal_with_nulls():
    """Test decimal.Decimal values with missing data."""
    from decimal import Decimal
    
    df = pd.DataFrame({
        "decimal_col": [Decimal('10.50'), None, Decimal('20.75'), Decimal('15.25')]
    })
    prof = profile_column(df, column="decimal_col")
    
    assert isinstance(prof, FloatProfile)
    assert prof.missing_prob == 0.25
    assert prof.min < 20.75
    assert prof.max > 10.50


def test_profile_column_decimal_single_value():
    """Test single decimal.Decimal value becomes OneValueProfile."""
    from decimal import Decimal
    
    df = pd.DataFrame({
        "decimal_col": [Decimal('42.50'), Decimal('42.50'), Decimal('42.50')]
    })
    prof = profile_column(df, column="decimal_col")
    
    assert isinstance(prof, OneValueProfile)
    assert prof.value == 42.5  # Converted to float
    assert prof.missing_prob == 0.0


def test_profile_column_decimal_mixed_precision():
    """Test decimal.Decimal values with different precision levels."""
    from decimal import Decimal
    
    df = pd.DataFrame({
        "decimal_col": [Decimal('10.5'), Decimal('20.75'), Decimal('15.250'), Decimal('30')]
    })
    prof = profile_column(df, column="decimal_col")
    
    assert isinstance(prof, FloatProfile)
    assert prof.missing_prob == 0.0


def test_FloatProfile_type_check_decimal():
    """Test FloatProfile.type_check recognizes decimal.Decimal values."""
    from decimal import Decimal
    
    # Single Decimal value
    assert FloatProfile.type_check(Decimal('42.5'))
    
    # Series of Decimal values
    df = pd.DataFrame({"col": [Decimal('10.5'), Decimal('20.75')]})
    assert FloatProfile.type_check(df["col"])
    
    # Mixed Decimal and None
    df_nulls = pd.DataFrame({"col": [Decimal('10.5'), None, Decimal('20.75')]})
    assert FloatProfile.type_check(df_nulls["col"])


def test_FloatProfile_contains_decimal():
    """Test FloatProfile contains method works with decimal.Decimal values."""
    from decimal import Decimal
    
    prof = FloatProfile(missing_prob=0.0, min=10.0, max=20.0)
    
    # Decimal values within range
    assert Decimal('15.5') in prof
    assert Decimal('10.0') in prof
    assert Decimal('20.0') in prof
    
    # Decimal values outside range
    assert Decimal('5.0') not in prof
    assert Decimal('25.0') not in prof
    
    # Mixed types within range
    assert 15.5 in prof  # float
    assert Decimal('15.5') in prof  # Decimal


def test_convert_numpy_to_python_decimal():
    """Test _convert_numpy_to_python handles decimal.Decimal values."""
    from decimal import Decimal
    from data_profiling.profile_df import _convert_numpy_to_python
    
    decimal_val = Decimal('42.75')
    result = _convert_numpy_to_python(decimal_val)
    
    assert isinstance(result, float)
    assert result == 42.75


def test_FloatProfile_script_generation_decimal():
    """Test script generation works correctly for Decimal-derived FloatProfile."""
    from decimal import Decimal
    
    df = pd.DataFrame({
        "decimal_col": [Decimal('10.50'), Decimal('20.75'), Decimal('15.25')]
    })
    prof = profile_column(df, column="decimal_col")
    
    assert isinstance(prof, FloatProfile)
    script = prof.sample_script()
    assert "random.uniform" in script
    assert "random.choice" not in script
    
    # Test samples_script
    samples_script = prof.samples_script(10)
    assert "random.uniform" in samples_script
    assert "for _ in range(10)" in samples_script