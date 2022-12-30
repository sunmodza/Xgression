from setuptools import setup

setup(
   name='Xgression',
   version='2.5',
   description='The Algorithm that find any relationship between data in equation form',
   author='Thaphon Chinnakornsakul',
   author_email='osunchizaza@gmail.com',
   packages=['src'],  #same as name
   install_requires=['numpy', 'sympy', 'scipy', 'matplotlib'],
)