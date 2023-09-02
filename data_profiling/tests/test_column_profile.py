import pandas as pd
from datetime import date
from importune import importunity
with importunity():
    from ..profile_df import (
        FloatProfile, IntProfile, BoolProfile, DateProfile, DateStrProfile,
        StringProfile, NullProfile, OneValueProfile, profile_column, DataFrameProfile
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

def test_profile_one_value_int():
    df = pd.DataFrame({"x": [1, 1, 1, None]})
    prof = profile_column(df, column="x")
    assert isinstance(prof, OneValueProfile)
    assert prof.value == 1
    assert prof.missing_prob == 0.25

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