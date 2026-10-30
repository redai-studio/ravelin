import abc
import unittest
from typing import TypeVar

from ravelin.assets._base import _Asset
from ravelin.utils.registry import RegistryMeta, Registry


# Define test classes using the metaclass for testing
class BaseTestAsset(_Asset, metaclass=RegistryMeta):
    """Abstract base class for testing AssetRegistryMeta"""

    TYPE_IDENTIFIER = "TEST_TYPE"

    @abc.abstractmethod
    def abstract_method(self) -> None:
        """Abstract method to make this class abstract"""
        pass


# Create a registry for test types
TestAssetT = TypeVar("TestAssetT", bound=BaseTestAsset)
TEST_REGISTRY = Registry[BaseTestAsset]("TEST_TYPE")

# Create decorator for registration
register_test_type = TEST_REGISTRY.create_decorator()


class TestRegistryMeta(unittest.TestCase):
    """Tests for the RegistryMeta metaclass"""

    def test_type_identifier_setup(self):
        """Test that TYPE_IDENTIFIER is properly set up during class creation"""

        class TestAssetWithType(metaclass=RegistryMeta):
            TYPE_IDENTIFIER = "CUSTOM_TYPE"

        # Verify the private attribute was created
        self.assertTrue(hasattr(TestAssetWithType, "__CUSTOM_TYPE"))
        self.assertEqual(NotImplemented, getattr(TestAssetWithType, "__CUSTOM_TYPE"))

    def test_unregistered_class_fails(self):
        """Test that concrete classes without registration fail validation"""

        # Define a concrete class without registration
        class UnregisteredAsset(BaseTestAsset):
            def abstract_method(self) -> None:
                pass

        # Should raise TypeError when instantiated since it's not registered
        with self.assertRaises(TypeError) as context:
            UnregisteredAsset()

        error_msg = str(context.exception)
        self.assertIn("not properly registered", error_msg)
        self.assertIn("Use the appropriate registration decorator", error_msg)

    def test_inheritance_validation(self):
        """Test that inheritance doesn't bypass registration requirements"""

        # Create a properly registered parent class
        @register_test_type("parent_type")
        class ParentAsset(BaseTestAsset):
            def abstract_method(self) -> None:
                pass

        # Create a child class without registering it
        class ChildAsset(ParentAsset):
            def abstract_method(self) -> None:
                pass

        # The child class should fail validation despite inheriting from a registered class
        with self.assertRaises(TypeError) as context:
            ChildAsset()

        error_msg = str(context.exception)
        self.assertIn("not properly registered", error_msg)
        self.assertIn("Use the appropriate registration decorator", error_msg)

    def test_abstract_class_no_check(self):
        """Test that abstract classes aren't checked for registration"""

        # Create an abstract class without registration
        class AbstractAsset(BaseTestAsset):
            @abc.abstractmethod
            def another_abstract_method(self) -> None:
                pass

            def abstract_method(self) -> None:
                pass

        # Should not raise an error (abstract classes aren't checked)
        # We can't instantiate it, but we can check it doesn't have errors
        self.assertTrue(abc.ABC in AbstractAsset.__mro__ or hasattr(AbstractAsset, "__abstractmethods__"))


class TestRegistry(unittest.TestCase):
    """Tests for the Registry class"""

    def setUp(self):
        # Create a fresh registry for each test
        self.registry = Registry[BaseTestAsset]("TEST_TYPE")

    def test_register(self):
        """Test registering a class"""

        # Create a class to register
        class TestAsset(BaseTestAsset):
            def abstract_method(self) -> None:
                pass

        # Register the class
        registered_class = self.registry.register("test_type", TestAsset)

        # Verify the class was registered correctly
        self.assertEqual(TestAsset, registered_class)
        self.assertEqual(TestAsset, self.registry.get("test_type"))

        # Verify the attribute was set on the class
        attr_name = "_TestAsset__TEST_TYPE"
        self.assertEqual("test_type", getattr(TestAsset, attr_name))

    def test_register_duplicate(self):
        """Test that registering a duplicate type value raises an error"""

        # Create two classes to register
        class FirstAsset(BaseTestAsset):
            def abstract_method(self) -> None:
                pass

        class SecondAsset(BaseTestAsset):
            def abstract_method(self) -> None:
                pass

        # Register the first class
        self.registry.register("duplicate_type", FirstAsset)

        # Registering the second class with the same type should raise an error
        with self.assertRaises(KeyError) as context:
            self.registry.register("duplicate_type", SecondAsset)

        error_msg = str(context.exception)
        self.assertIn("is already used by", error_msg)
        self.assertIn("FirstAsset", error_msg)

    def test_create_decorator(self):
        """Test creating and using a decorator"""

        # Create a decorator
        decorator_factory = self.registry.create_decorator()

        # Use the decorator to register a class
        @decorator_factory("decorator_type")
        class DecoratedAsset(BaseTestAsset):
            def abstract_method(self) -> None:
                pass

        # Verify the class was registered correctly
        self.assertEqual(DecoratedAsset, self.registry.get("decorator_type"))

        # Verify the attribute was set on the class
        attr_name = "_DecoratedAsset__TEST_TYPE"
        self.assertEqual("decorator_type", getattr(DecoratedAsset, attr_name))

    def test_decorator_abstract_class(self):
        """Test that decorating an abstract class raises an error"""

        # Create a decorator
        decorator_factory = self.registry.create_decorator()

        # Using the decorator on an abstract class should raise an error
        with self.assertRaises(TypeError) as context:

            @decorator_factory("abstract_type")
            class AbstractAsset(BaseTestAsset):
                @abc.abstractmethod
                def another_method(self) -> None:
                    pass

                def abstract_method(self) -> None:
                    pass

        error_msg = str(context.exception)
        self.assertIn("Cannot register abstract class", error_msg)

    def test_decorator_duplicate(self):
        """Test that decorating with a duplicate type value raises an error early"""

        # Create a decorator
        decorator_factory = self.registry.create_decorator()

        # Register a class
        @decorator_factory("early_duplicate")
        class FirstAsset(BaseTestAsset):
            def abstract_method(self) -> None:
                pass

        # Creating a decorator with the same type should raise an error immediately
        with self.assertRaises(KeyError) as context:

            @decorator_factory("early_duplicate")
            class SecondAsset(BaseTestAsset):
                def abstract_method(self) -> None:
                    pass

        error_msg = str(context.exception)
        self.assertIn("is already used by", error_msg)
        self.assertIn("FirstAsset", error_msg)

    def test_get(self):
        """Test retrieving a class by type value"""

        # Create and register a class
        class TestAsset(BaseTestAsset):
            def abstract_method(self) -> None:
                pass

        self.registry.register("get_test", TestAsset)

        # Retrieve the class
        retrieved_class = self.registry.get("get_test")

        # Verify the correct class was retrieved
        self.assertEqual(TestAsset, retrieved_class)

    def test_get_unregistered(self):
        """Test that retrieving an unregistered type value raises an error"""

        # Try to get an unregistered type
        with self.assertRaises(KeyError) as context:
            self.registry.get("unregistered_type")

        error_msg = str(context.exception)
        self.assertIn("is not registered", error_msg)
        self.assertIn("No types registered yet", error_msg)

        # Register a type and try again with a different unregistered type
        class TestAsset(BaseTestAsset):
            def abstract_method(self) -> None:
                pass

        self.registry.register("registered_type", TestAsset)

        with self.assertRaises(KeyError) as context:
            self.registry.get("another_unregistered_type")

        error_msg = str(context.exception)
        self.assertIn("is not registered", error_msg)
        self.assertIn("Available types:", error_msg)
        self.assertIn("registered_type", error_msg)

    def test_get_all(self):
        """Test retrieving all registered classes"""

        # Create and register multiple classes
        class FirstAsset(BaseTestAsset):
            def abstract_method(self) -> None:
                pass

        class SecondAsset(BaseTestAsset):
            def abstract_method(self) -> None:
                pass

        self.registry.register("first_type", FirstAsset)
        self.registry.register("second_type", SecondAsset)

        # Get all registered classes
        all_classes = self.registry.get_all()

        # Verify the registry contains all registered classes
        self.assertEqual(2, len(all_classes))
        self.assertEqual(FirstAsset, all_classes["first_type"])
        self.assertEqual(SecondAsset, all_classes["second_type"])

    def test_contains(self):
        """Test checking if a type value is registered"""

        # Create and register a class
        class TestAsset(BaseTestAsset):
            def abstract_method(self) -> None:
                pass

        self.registry.register("contains_test", TestAsset)

        # Check if types are registered
        self.assertTrue(self.registry.contains("contains_test"))
        self.assertFalse(self.registry.contains("nonexistent_type"))


if __name__ == "__main__":
    unittest.main()
