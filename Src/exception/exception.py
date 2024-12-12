import sys


def error_message_detail(error, error_detail: sys):
    """
    Extract detailed error information including the file name, line number, and error message.
    Args:
        error (Exception): The original exception object.
        error_detail (sys): The sys module, used to extract exception details.
    Returns:
        str: Detailed error message with file name and line number.
    """
    _, _, exc_tb = error_detail.exc_info()  # Extract the current exception details
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the file name where the error occurred
    line_number = exc_tb.tb_lineno  # Get the line number where the error occurred
    error_message = (
        "Error occurred in script: [{0}] at line: [{1}], with error: [{2}]".format(
            file_name, line_number, str(error)
        )
    )
    return error_message


class CustomException(Exception):
    def __init__(self, error_message: str, error_detail: sys):
        """
        Custom Exception class to capture detailed exception information.
        Args:
            error_message (str): Custom error message.
            error_detail (sys): sys module to extract detailed error traceback information.
        """
        super().__init__(error_message)  # Initialize the base Exception class
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
