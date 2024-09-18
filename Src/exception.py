import sys
import traceback


def error_message_detail(error):
    """
    Extract detailed error information including the file name, line number, and error message.
    """
    exc_type, exc_obj, exc_tb = sys.exc_info()  # Extract the current exception details
    file_name = (
        exc_tb.tb_frame.f_code.co_filename
    )  # Get the file name where the error occurred
    line_number = exc_tb.tb_lineno  # Get the line number where the error occurred
    error_message = (
        "Error occurred in script: [{0}] at line: [{1}], with error: [{2}]".format(
            file_name, line_number, str(error)
        )
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)  # Properly initialize the base Exception class
        self.error_message = error_message_detail(error_message)

    def __str__(self):
        return self.error_message
