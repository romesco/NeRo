# Copyright (c) 2020, romesco, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
from typing import Dict, Optional, Union
from abc import ABC
from contextlib import contextmanager
import wrapt

from nero.core.neural_types import NeuralType, NeuralTypeComparisonResult

_TYPECHECK_ENABLED = True

class Typing(ABC):
    """
    An interface which endows module with neural types
    """

    @property
    def input_types(self) -> Optional[Dict[str, NeuralType]]:
        """Define these to enable input neural type checks"""
        return None

    @property
    def output_types(self) -> Optional[Dict[str, NeuralType]]:
        """Define these to enable output neural type checks"""
        return None

    def _validate_input_types(self, input_types=None, **kwargs):
        """
        This function does a few things.
        1) It ensures that len(self.input_types <non-optional>) <= len(kwargs) <= len(self.input_types).
        2) For each (keyword name, keyword value) passed as input to the wrapped function:
            - Check if the keyword name exists in the list of valid self.input_types names.
            - Check if keyword value has the `neural_type` property.
                - If it does, then perform a comparative check and assert that neural types
                    are compatible (SAME or GREATER).
            - Check if keyword value is a container type (list or tuple). If yes,
                then perform the elementwise test of neural type above on each element
                of the nested structure, recursively.
        Args:
            input_types: Either the `input_types` defined at class level, or the local function
                overridden type definition.
            kwargs: Dictionary of argument_name:argument_value pairs passed to the wrapped
                function upon call.
        """
        # TODO: Properly implement this
        if input_types is not None:
            total_input_types = len(input_types)
            mandatory_input_types = len(
                [type_val for type_key, type_val in input_types.items() if not type_val.optional]
            )

            if len(kwargs) < mandatory_input_types or len(kwargs) > total_input_types:
                raise TypeError(
                    f"Number of input arguments provided ({len(kwargs)}) is not as expected. Function has "
                    f"{total_input_types} total inputs with {mandatory_input_types} mandatory inputs."
                )

            for key, value in kwargs.items():
                # Check if keys exists in the defined input types
                if key not in input_types:
                    raise TypeError(
                        f"Input argument {key} has no corresponding input_type match. "
                        f"Existing input_types = {input_types.keys()}"
                    )

                # Perform neural type check
                if hasattr(value, 'neural_type') and not input_types[key].compare(value.neural_type) in (
                    NeuralTypeComparisonResult.SAME,
                    NeuralTypeComparisonResult.GREATER,
                ):
                    error_msg = [
                        f"{input_types[key].compare(value.neural_type)} :",
                        f"Input type expected : {input_types[key]}",
                        f"Input type found : {value.neural_type}",
                    ]
                    for i, dict_tuple in enumerate(input_types[key].elements_type.type_parameters.items()):
                        error_msg.insert(i + 2, f'  input param_{i} : {dict_tuple[0]}: {dict_tuple[1]}')
                    for i, dict_tuple in enumerate(value.neural_type.elements_type.type_parameters.items()):
                        error_msg.append(f'  input param_{i} : {dict_tuple[0]}: {dict_tuple[1]}')
                    raise TypeError("\n".join(error_msg))

                # Perform input ndim check
                if hasattr(value, 'shape'):
                    value_shape = value.shape
                    type_shape = input_types[key].axes
                    name = key

                    if type_shape is not None and len(value_shape) != len(type_shape):
                        raise TypeError(
                            f"Input shape mismatch occured for {name} in module {self.__class__.__name__} : \n"
                            f"Input shape expected = {input_types[key].axes} | \n"
                            f"Input shape found : {value_shape}"
                        )

                # Perform recursive neural type check for homogeneous elements
                elif isinstance(value, list) or isinstance(value, tuple):
                    for ind, val in enumerate(value):
                        self.__check_neural_type(val, input_types[key], name=key)

    def _attach_and_validate_output_types(self, out_objects, output_types=None):
        """
        This function does a few things.
        1) It ensures that len(out_object) == len(self.output_types).
        2) If the output is a tensor (or list/tuple of list/tuple ... of tensors), it
            attaches a neural_type to it. For objects without the neural_type attribute,
            such as python objects (dictionaries and lists, primitive data types, structs),
            no neural_type is attached.
            Note: tensor.neural_type is only checked during _validate_input_types which is
            called prior to forward().
        Args:
            output_types: Either the `output_types` defined at class level, or the local function
                overridden type definition.
            out_objects: The outputs of the wrapped function.
        """
        # TODO: Properly implement this
        if output_types is not None:
            out_types_list = list(output_types.items())

            # First convert all outputs to list/tuple format to check correct number of outputs
            if type(out_objects) in (list, tuple):
                out_container = out_objects
            else:
                out_container = [out_objects]

            if len(output_types) != len(out_container):
                raise TypeError(
                    "Number of output arguments provided ({}) is not as expected ({})".format(
                        len(out_container), len(output_types)
                    )
                )

            # Attach types recursively, if possible
            if not isinstance(out_objects, tuple) and not isinstance(out_objects, list):
                try:
                    out_objects.neural_type = out_types_list[0][1]
                except Exception:
                    pass

                # Perform output ndim check
                if hasattr(out_objects, 'shape'):
                    value_shape = out_objects.shape
                    type_shape = out_types_list[0][1].axes
                    name = out_types_list[0][0]

                    if type_shape is not None and len(value_shape) != len(type_shape):
                        raise TypeError(
                            f"Output shape mismatch occured for {name} in module {self.__class__.__name__} : \n"
                            f"Output shape expected = {type_shape} | \n"
                            f"Output shape found : {value_shape}"
                        )
            else:
                for ind, res in enumerate(out_objects):
                    self.__attach_neural_type(res, out_types_list[ind][1], name=out_types_list[ind][0])

    def __check_neural_type(self, obj, type_val, name=None):
        if isinstance(obj, tuple) or isinstance(obj, list):
            for elem in obj:
                self.__check_neural_type(elem, type_val, name=name)
            return  # after processing nest, return to avoid testing nest itself

        if hasattr(obj, 'neural_type') and not type_val.compare(obj.neural_type) in (
            NeuralTypeComparisonResult.SAME,
            NeuralTypeComparisonResult.GREATER,
        ):
            raise TypeError(
                f"{type_val.compare(obj.neural_type)} : \n"
                f"Input type expected = {type_val} | \n"
                f"Input type found : {obj.neural_type}"
            )

        # Perform input ndim check
        if hasattr(obj, 'shape'):
            value_shape = obj.shape
            type_shape = type_val.axes

            if type_shape is not None and len(value_shape) != len(type_shape):
                raise TypeError(
                    f"Input shape mismatch occured for {name} in module {self.__class__.__name__} : \n"
                    f"Input shape expected = {type_shape} | \n"
                    f"Input shape found : {value_shape}"
                )

    def __attach_neural_type(self, obj, type_val, name=None):
        if isinstance(obj, tuple) or isinstance(obj, list):
            for elem in obj:
                self.__attach_neural_type(elem, type_val, name=name)
            return  # after processing nest, return to avoid argument insertion into nest itself

        try:
            obj.neural_type = type_val
        except Exception:
            pass

        # Perform output ndim check
        if hasattr(obj, 'shape'):
            value_shape = obj.shape
            type_shape = type_val.axes

            if type_shape is not None and len(value_shape) != len(type_shape):
                raise TypeError(
                    f"Output shape mismatch occured for {name} in module {self.__class__.__name__} : \n"
                    f"Output shape expected = {type_shape} | \n"
                    f"Output shape found : {value_shape}"
                )



def is_typecheck_enabled():
    """
    Getter method for typechecking state.
    """
    return _TYPECHECK_ENABLED

class typecheck:
    class TypeState(Enum):
        """
        Placeholder to denote the default value of type information provided.
        If the constructor of this decorator is used to override the class level type definition,
        this enum value indicate that types will be overridden.
        """

        UNINITIALIZED = 0

    def __init__(
        self,
        input_types: Union[TypeState, Dict[str, NeuralType]] = TypeState.UNINITIALIZED,
        output_types: Union[TypeState, Dict[str, NeuralType]] = TypeState.UNINITIALIZED,
    ):
        """
        A decorator which performs input-output neural type checks, and attaches
        neural types to the output of the function that it wraps.
        Requires that the class inherit from `nemo.core.Typing` in order to perform
        type checking, and will raise an error if that is not the case.
        # Usage (Class level type support)
        @typecheck()
        def fn(self, arg1, arg2, ...):
            ...
        # Usage (Function level type support)
        @typecheck(input_types=..., output_types=...)
        def fn(self, arg1, arg2, ...):
            ...
        Points to be noted:
        1) The brackets () in `@typecheck()` are necessary.
            You will encounter a TypeError: __init__() takes 1 positional argument but X
            were given without those brackets.
        2) The function can take any number of positional arguments during definition.
            When you call this function, all arguments must be passed using kwargs only.
        """
        self.input_types = input_types
        self.output_types = output_types

        if input_types == self.TypeState.UNINITIALIZED:
            self.input_override = False
        else:
            self.input_override = True

        if output_types == self.TypeState.UNINITIALIZED:
            self.output_override = False
        else:
            self.output_override = True

    @wrapt.decorator(enabled=is_typecheck_enabled)
    def __call__(self, wrapped, instance: Typing, args, kwargs):
        if instance is None:
            raise RuntimeError("Only classes which inherit nemo.core.Typing can use this decorator !")

        if not isinstance(instance, Typing):
            raise RuntimeError("Only classes which inherit nemo.core.Typing can use this decorator !")

        if hasattr(instance, 'input_ports') or hasattr(instance, 'output_ports'):
            raise RuntimeError(
                "Typing requires override of `input_types()` and `output_types()`, "
                "not `input_ports() and `output_ports()`"
            )

        # Preserve type information
        if self.input_types is typecheck.TypeState.UNINITIALIZED:
            self.input_types = instance.input_types

        if self.output_types is typecheck.TypeState.UNINITIALIZED:
            self.output_types = instance.output_types

        # Resolve global type or local overridden type
        if self.input_override:
            input_types = self.input_types
        else:
            input_types = instance.input_types

        if self.output_override:
            output_types = self.output_types
        else:
            output_types = instance.output_types

        # If types are not defined, skip type checks and just call the wrapped method
        if input_types is None and output_types is None:
            return wrapped(*args, **kwargs)

        # Check that all arguments are kwargs
        if input_types is not None and len(args) > 0:
            raise TypeError("All arguments must be passed by kwargs only for typed methods")

        # Perform rudimentary input checks here
        instance._validate_input_types(input_types=input_types, **kwargs)

        # Call the method - this can be forward, or any other callable method
        outputs = wrapped(*args, **kwargs)

        instance._attach_and_validate_output_types(output_types=output_types, out_objects=outputs)

        return outputs

    @staticmethod
    def set_typecheck_enabled(enabled: bool = True):
        global _TYPECHECK_ENABLED
        _TYPECHECK_ENABLED = enabled

    @staticmethod
    @contextmanager
    def disable_checks():
        typecheck.set_typecheck_enabled(enabled=False)
        try:
            yield
        finally:
            typecheck.set_typecheck_enabled(enabled=True)

