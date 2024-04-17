from setuptools import setup, find_packages

extras = {}

setup(
    name="Ca_Naxca",
    version="0.0.0.1",
    author="Jesús Armenta-Segura",
    author_email="jesus.jorge.armenta@gmail.com",
    description="Librería con documentación en nahuatl... técnico (NI SIQUIERA LOS HABLANTES NATIVOS LO ENTENDERÍAN).",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP",
    license="Apache",
    url="https://github.com/gmihaila/Ca_Naxca",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "numpy",
        "tqdm >= 4.27",
    ],
    extras_require=extras,
    python_requires=">=3.6.0",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)