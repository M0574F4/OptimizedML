from setuptools import setup, find_packages

setup(
    name="OptimizedML",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "requests",
    ],
    author="M0574F4",
    author_email="mostafa.naseri1991@gmail.com",
    description="A short description of the project.",
    license="MIT",
    url="https://github.com/yourusername/OptimizedML",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
