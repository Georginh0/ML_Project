import os
from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    This function returns a list of requirements.
    """
    if not os.path.exists(file_path):
        return []  # Return an empty list if the file doesn't exist

    with open(file_path, "r") as file_obj:
        requirements = [line.strip() for line in file_obj if line.strip()]

    # Remove '-e .' if it's present in the list
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)

    return requirements


# Use os.path.join to construct the file path
requirements_path = os.path.join(
    "/Users/georgensamuel",
    "End to End ML Project",
    "project-template",
    "requirements.txt",
)

# Fetch requirements
install_requires = get_requirements(requirements_path)

setup(
    name="MLproject",
    version="0.0.1",
    author="George Dogo",
    author_email="George.sam@live.co.uk",
    packages=find_packages(),
    install_requires=install_requires,
)
