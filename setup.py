from setuptools import setup, find_packages

setup(
    name="template-nn",
    version="0.1.3",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'torch>=2.5.0',
        'pandas',
        'numpy',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",

    description="A neural network model architecture template",
    url="https://gabrielchoong.github.io/template-nn",

    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
