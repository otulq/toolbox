import pytest

def test_failing_import():
    # if we do from .example_module import example_function_1 we should get an ImportError
    # because the module is not in the current directory
    with pytest.raises(ImportError):
        from .example_module import example_function_1

    with pytest.raises(ImportError):
        from .example_module import example_function_1, example_function_2

    with pytest.raises(ImportError):
        from .smart_import import smart_import, smart_import_guard

    with pytest.raises(ImportError):
        from .example_folder.example_module_2 import example_function_3

    with pytest.raises(ImportError):
        from .example_folder.example_module_2 import example_function_3, example_function_4

def test_smart_import():
    import smart_import
    from smart_import import smart_import, smart_import_guard

    smart_import('from .example_module import example_function_1')
    assert example_function_1() == 'example_output_1'

    smart_import('from .example_module import example_function_2')
    assert example_function_2() == 'example_output_2'

    smart_import('from .example_module import example_function_1, example_function_2')
    assert example_function_1() == 'example_output_1'
    assert example_function_2() == 'example_output_2'

def test_smart_import_guard():
    import smart_import
    from smart_import import smart_import, smart_import_guard

    # test importing a single function
    with smart_import_guard():
        from .example_module import example_function_1
    assert example_function_1() == 'example_output_1'

    with smart_import_guard():
        from .example_module import example_function_2
    assert example_function_2() == 'example_output_2'

    # test importing multiple functions in a single statement
    with smart_import_guard():
        from .example_module import example_function_1, example_function_2
    assert example_function_1() == 'example_output_1'
    assert example_function_2() == 'example_output_2'

    # test importing multiple functions in multiple statements
    with smart_import_guard():
        from .example_module import example_function_1
        from .example_module import example_function_2
    assert example_function_1() == 'example_output_1'
    assert example_function_2() == 'example_output_2'

def test_smart_import_from_folder_above():
    import smart_import
    from smart_import import smart_import, smart_import_guard

    with smart_import_guard():
        from ..example_file_for_test import example_function_5
    assert example_function_5() == 'example_output_5'

    with smart_import_guard():
        from ..example_file_for_test import example_function_5, example_function_6
    assert example_function_5() == 'example_output_5'

def test_smart_import_guard_from_folder_above():
    import smart_import
    from smart_import import smart_import, smart_import_guard

    with smart_import_guard():
        from ..example_file_for_test import example_function_5
    assert example_function_5() == 'example_output_5'

    with smart_import_guard():
        from ..example_file_for_test import example_function_5, example_function_6
    assert example_function_5() == 'example_output_5'
    assert example_function_6() == 'example_output_6'

def test_smart_import_from_2_levels_above():
    import smart_import
    from smart_import import smart_import, smart_import_guard

    with smart_import_guard():
        from ...dynamic_imports.example_file_for_test import example_function_5
    assert example_function_5() == 'example_output_5'

    with smart_import_guard():
        from ...dynamic_imports.example_file_for_test import example_function_5, example_function_6
    assert example_function_5() == 'example_output_5'
    assert example_function_6() == 'example_output_6'

def test_smart_import_from_2_levels_above_guard():
    import smart_import
    from smart_import import smart_import, smart_import_guard

    with smart_import_guard():
        from ...dynamic_imports.example_file_for_test import example_function_5