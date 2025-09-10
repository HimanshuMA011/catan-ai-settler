"""Setup script for Catan Assistant."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("docker/requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="catan-assistant",
    version="1.0.0",
    author="LLM Assistant",
    description="LLM-powered Settlers of Catan game analysis and strategy assistant",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "catan-assistant=catan_assistant.api.fastapi_app:run_server",
            "catan-train=catan_assistant.data.training_generator:generate_training_data",
        ],
    },
    include_package_data=True,
    package_data={
        "catan_assistant": ["data/*.json", "config/*.yaml"],
    },
)