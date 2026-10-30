"""
This module provides utilities for registering and managing different types of assets.

It includes:
1. A metaclass that enforces registration of concrete asset classes
2. A generic registry for managing type-to-class mappings
"""

import abc
import inspect
from typing import Callable, Dict, Generic, Type, TypeVar


# Generic type for registry contents
T = TypeVar("T")


class RegistryMeta(abc.ABCMeta):
    """
    Metaclass that enforces registration of concrete subclasses.

    This metaclass checks if concrete classes have their type identifier properly set,
    ensuring they've been registered using the appropriate decorator before being instantiated.

    Classes using this metaclass should define a TYPE_IDENTIFIER attribute
    that specifies the name of the type identifier (without leading underscores).
    """

    def __new__(mcs, name, bases, namespace):
        # Get the type identifier name
        type_id_name = namespace.get("TYPE_IDENTIFIER", None)

        # If this class defines a type identifier, ensure it's properly set up
        if type_id_name is not None:
            # Create a private attribute to hold the value
            namespace[f"__{type_id_name}"] = NotImplemented

        return super().__new__(mcs, name, bases, namespace)

    def __call__(cls, *args, **kwargs):
        # Only check concrete classes (not abstract) and only at instantiation time
        if not inspect.isabstract(cls):
            # Get the type identifier name directly from the class
            type_id_name = getattr(cls, "TYPE_IDENTIFIER", None)

            # Skip check if class doesn't define a type identifier
            if type_id_name is not None:
                # Build the mangled attribute name (how Python actually stores it)
                mangled_attr = f"_{cls.__name__}__" + type_id_name

                # Get the value using the mangled name
                type_value = getattr(cls, mangled_attr, None)

                if type_value is NotImplemented or type_value is None:
                    raise TypeError(
                        f"Concrete class {cls.__name__} is not properly registered. "
                        f"Use the appropriate registration decorator."
                    )

        # Call the parent's __call__ method to create the instance
        return super().__call__(*args, **kwargs)


class Registry(Generic[T]):
    """
    Generic registry for asset types that helps manage registration of classes.

    This class provides a central store for mapping type identifiers to their
    corresponding classes, along with methods for registration and lookup.

    :param type_attr: The name of the type identifier (without leading underscores)
    """

    def __init__(self, type_attr: str):
        """
        Initialize a new registry.

        :param type_attr: The name of the type identifier (without leading underscores)
        """
        self._registry: Dict[str, Type[T]] = {}
        self.type_attr = type_attr

    def register(self, type_value: str, cls: Type[T]) -> Type[T]:
        """
        Register a class with a specific type value.

        :param type_value: The unique type value
        :param cls: The class to register
        :return: The registered class (unchanged)
        :raises KeyError: If the type value is already registered
        """
        if type_value in self._registry:
            existing_class = self._registry[type_value]
            raise KeyError(
                f"{self.type_attr} '{type_value}' is already used by {existing_class.__name__}. "
                f"Each class must have a unique {self.type_attr}."
            )

        self._registry[type_value] = cls

        # Set the attribute directly with the fully mangled name
        # This follows Python's name mangling pattern: _ClassName__attribute
        setattr(cls, f"_{cls.__name__}__{self.type_attr}", type_value)

        return cls

    def create_decorator(self) -> Callable[[str], Callable[[Type[T]], Type[T]]]:
        """
        Create a decorator function for registering classes with this registry.

        The created decorator takes a type value string and returns a function that
        can be used to decorate a class, registering it with the given type value.

        Usage:
            register_class = registry.create_decorator()

            @register_class("my_type")
            class MyClass:
                pass

        :return: A decorator factory function that can be used to register classes
        """

        def decorator_factory(type_value: str) -> Callable[[Type[T]], Type[T]]:
            # Check for duplicate type_value early
            if type_value in self._registry:
                existing_class = self._registry[type_value]
                raise KeyError(
                    f"{self.type_attr} '{type_value}' is already used by {existing_class.__name__}. "
                    f"Each class must have a unique {self.type_attr}."
                )

            def decorator(cls: Type[T]) -> Type[T]:
                if inspect.isclass(cls) and bool(getattr(cls, "__abstractmethods__", {})):
                    raise TypeError(
                        f"Cannot register abstract class {cls.__name__}. "
                        f"Only concrete classes should use this decorator."
                    )

                # Register the class
                return self.register(type_value, cls)

            return decorator

        return decorator_factory

    def get(self, type_value: str) -> Type[T]:
        """
        Get a class by its type value.

        :param type_value: The type value to look up
        :return: The registered class
        :raises KeyError: If the type value is not registered
        """
        try:
            return self._registry[type_value]
        except KeyError as err:
            registered_types = ", ".join(sorted(self._registry.keys()))
            available_msg = f"Available types: {registered_types}" if self._registry else "No types registered yet."
            raise KeyError(
                f"Type '{type_value}' is not registered in the {self.type_attr} registry. {available_msg}"
            ) from err

    def get_all(self) -> Dict[str, Type[T]]:
        """
        Get all registered types.

        :return: A copy of the registry mapping type values to classes
        """
        return self._registry.copy()

    def contains(self, type_value: str) -> bool:
        """
        Check if a type value is registered.

        :param type_value: The type value to check
        :return: True if the type is registered, False otherwise
        """
        return type_value in self._registry
