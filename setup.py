import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

USER_NAME="jrk90210us"
PROJECT_NAME="ANN-Implementation-modular"

setuptools.setup(
    name="src",
    version="0.0.1",
    author="Jatin",
    author_email="jrk90210us@gmail.com",
    description="Packaging ANN implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{USER_NAME}/{PROJECT_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{USER_NAME}/{PROJECT_NAME}/issues",
    },
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "tensorflow",
        "numpy",
        "matplotlib",
        "pandas",
        "seaborn"
    ]
)