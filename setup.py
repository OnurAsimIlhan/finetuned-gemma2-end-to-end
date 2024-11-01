from setuptools import find_packages, setup
from typing import List

def get_requirements()->List[str]:
    requirements_lst = []
    try:
        with open('requirements.txt', 'r') as file:
            lines=file.readlines()
            for line in lines:
                requirement = line.strip()
                if requirement and requirement != '-e .':
                    requirements_lst.append(requirement)
    except FileNotFoundError:
        print("requirements.txt not found")

    return requirements_lst

setup(
    name="Bluearf_Task1",
    version="0.0.1",
    author="Onur",
    author_email="onurasimilhan01@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements()
)