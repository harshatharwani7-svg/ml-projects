
import sys

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Build a detailed error string with file name and line number.
    Should be called inside an `except` block.
    """
    # exc_info returns (type, value, traceback) for the current exception
    _, _, exc_tb = error_detail.exc_info()

    # Fallback in case exc_info() is None (e.g., called outside except)
    if exc_tb is None:
        return f"Error message: {error}"

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno

    return (
        f"Error occurred in python script [{file_name}] "
        f"at line [{line_number}] with message: {error}"
    )


class CustomException(Exception):
    """
    Custom exception that includes file name and line number in its message.
    """
    def __init__(self, error_message: Exception, error_detail: sys):
        # Store the original message in the base Exception
        super().__init__(str(error_message))
        # Build and store the detailed message
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message
