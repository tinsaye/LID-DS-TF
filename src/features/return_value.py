from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall


class ReturnValueWithError(BuildingBlock):
    """
    calculate system call return value for all syscalls.
    Returns:
        - return value as is if it is an integer
        - Error code if it is an error
        - 0 if it is not an integer and not an error
    """

    def __init__(self):
        super().__init__()

    def _calculate(self, syscall: Syscall):
        return_value_string = syscall.param('res')
        if return_value_string is not None:
            try:
                return_value = int(return_value_string)
            except ValueError:
                if '(E' in return_value_string:
                    return_value = return_value_string
                else:
                    return_value = 0
            return return_value

    def depends_on(self):
        return []
