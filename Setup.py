from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='Ca_Naxca',
    url='https://github.com/JesusASmx/Ca_Naxca',
    author='Jesús Armenta-Segura',
    author_email='jesus.jorge.armenta@gmail.com',
    # Needed to actually package something
    packages=['Ca_Naxca'],
    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version=0.0.0.1,
    # The license can be anything you like
        #license='MIT',
        #description='',
    #long_description=open('README.txt').read(),
)
