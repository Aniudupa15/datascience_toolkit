from setuptools import setup, find_packages

setup(
    name="datascience_toolkitt",
    version="1.0.7",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "mlxtend",
        "nltk",
        "networkx"
    ],
    description="All assignment programs packaged into one toolkit",
    author="Anirudha",
    python_requires=">=3.8",
)
