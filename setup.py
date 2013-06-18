try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

setup(
    name='irlbpy',
    version='0.1.0',
    author='Bryan W. Lewis',
    author_email='blewis@illposed.net',
    packages=['irlb'],
    url='https://github.com/bwlewis/irlbpy',
    license='LICENSE.txt',
    description='Truncated SVD by implicitly restarted Lanczos '
                'bidiagonalization',
    long_description=open('README.md').read(),
    install_requires=["numpy"],
)

