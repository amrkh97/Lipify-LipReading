from setuptools import setup, find_packages

setup(
    name='Lipify-LipReading',
    version='0.1.0',
    url='',
    license='The MIT License (MIT)',
    author='Amr Khaled',
    packages=find_packages,
    author_email='amrkh97@gmail.com',
    description='',
    install_requires=["tensorflow>=2.1.0",
                      "opencv-python>=4.2.0",
                      "imutils>=0.5.3",
                      "dlib",
                      "moviepy>=1.0.1",
                      "h5py",
                      "tqdm",
                      "numpy>=1.18.1",
                      "pandas>=1.0.1",
                      "python>=3.7.1"]
)
