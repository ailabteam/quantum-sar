# File: setup.py

from setuptools import setup, find_packages

setup(
    name='quantum_sar',
    version='0.1.0',
    author='Phuc Hao Do', # Thay bằng tên của bạn
    author_email='do.hf@sut.ru', # Thay bằng email của bạn
    description='A QUBO Framework for InSAR Phase Unwrapping',
    packages=find_packages(),
    install_requires=[
        # Chúng ta có thể liệt kê các dependencies ở đây,
        # hoặc để pytest đọc từ requirements.txt.
        # Để trống bây giờ cũng được.
    ],
    python_requires='>=3.8',
)
