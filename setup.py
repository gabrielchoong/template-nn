from setuptools import setup, find_packages

setup(
    name="template-nn",
    version="0.1.5.dev0",
    packages=find_packages(where="src",
                           include=["template_nn", "template_nn.*"]),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.6.0",
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "mealpy>=3.0.2",
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
