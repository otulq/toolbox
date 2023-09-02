import pandas as pd
from datetime import date
from importune import importunity
with importunity():
    from ..profile_df import (
        FloatProfile, IntProfile, BoolProfile, DateProfile, DateStrProfile,
        StringProfile, profile_column, DataFrameProfile
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
# profile_column – numeric inference
# ────────────────────────────────────────────────────────────────────────────
def test_profile_column_numeric():
    df = pd.DataFrame({"x": [1, 2, 3, None]})
    prof = profile_column(df, column="x")
    assert isinstance(prof, FloatProfile)  # pandas upcasts to float with nulls
    assert prof.min == 1.2 and prof.max == 2.8  # 10th and 90th percentiles
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