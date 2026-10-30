import unittest
from typing import Any
from unittest.mock import patch


from ravelin.components.others.custom_function_executor import CustomFunctionExecutor


def sample_function(a: float, b, c: int = 0):
    return {"result": a + b + c}


def sample_function_with_kwargs(*args, **kwargs):
    return {"result": kwargs.get("a", 0) + kwargs.get("b", 0)}


class TestCustomFunctionExecutor(unittest.TestCase):
    def setUp(self):
        self.custom_function = sample_function
        self.output_types = {"result": int}
        self.executor = CustomFunctionExecutor(self.custom_function, self.output_types)

    def test_init_valid(self):
        self.assertEqual(self.executor.custom_function, self.custom_function)
        self.assertDictEqual(self.executor.output_types, self.output_types)
        self.assertTrue(hasattr(self.executor, "__haystack_input__"))
        input_types = getattr(self.executor, "__haystack_input__")._sockets_dict
        self.assertEqual(3, len(input_types))
        self.assertDictEqual({"a": float, "b": Any, "c": int}, {i.name: i.type for i in input_types.values()})
        self.assertTrue(input_types["a"].is_mandatory)
        self.assertTrue(input_types["b"].is_mandatory)
        self.assertEqual(0, input_types["c"].default_value)

    def test_init_non_keyword_arguments(self):
        with self.assertRaisesRegex(ValueError, "Custom function input '.+' must be a keyword argument."):
            CustomFunctionExecutor(sample_function_with_kwargs, {"result": int})

    def test_to_dict(self):
        expected_dict = {
            "type": "ravelin.components.others.custom_function_executor.CustomFunctionExecutor",
            "init_parameters": {
                "custom_function": "ravelin.tests.components.others.custom_function_executor_test.sample_function",
                "output_types": {"result": "int"},
            },
        }
        with patch(
            "ravelin.components.others.custom_function_executor.serialize_callable",
            return_value="ravelin.tests.components.others.custom_function_executor_test.sample_function",
        ):
            result = self.executor.to_dict()
            self.assertEqual(expected_dict, result)

    def test_from_dict(self):
        data = {
            "type": "ravelin.components.others.custom_function_executor.CustomFunctionExecutor",
            "init_parameters": {
                "custom_function": "ravelin.tests.components.others.custom_function_executor_test.sample_function",
                "output_types": {"result": "int"},
            },
        }
        with patch(
            "ravelin.components.others.custom_function_executor.deserialize_callable",
            return_value=self.custom_function,
        ):
            executor = CustomFunctionExecutor.from_dict(data)
            self.assertIsInstance(executor, CustomFunctionExecutor)
            self.assertEqual(executor.custom_function, self.custom_function)
            self.assertEqual(executor.output_types, {"result": int})

    def test_run_successful(self):
        result = self.executor.run(a=2, b=3)
        expected = {"result": 5}
        self.assertEqual(result, expected)
        result = self.executor.run(a=2.0, b=3, c=-5)
        expected = {"result": 0}
        self.assertEqual(result, expected)

    def test_run_missing_arguments(self):
        with self.assertRaisesRegex(
            RuntimeError, "Error executing custom function: .+ missing 1 required positional argument: 'b'"
        ):
            self.executor.run(a=1)  # Missing 'b'

    def test_run_custom_function_exception(self):
        def faulty_function(a, b):
            raise ValueError("Intentional Error")

        executor = CustomFunctionExecutor(faulty_function, {"result": int})
        with self.assertRaisesRegex(RuntimeError, "Error executing custom function: Intentional Error"):
            executor.run(a=1, b=2)


if __name__ == "__main__":
    unittest.main()
