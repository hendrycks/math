import setuptools
  
setuptools.setup(
    name="math_equivalence",
    description="A utility for determining whether 2 answers for a problem in the MATH dataset are equivalent.",
    url="https://github.com/hendrycks/math",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "modeling"},
    py_modules = ["math_equivalence"],
)