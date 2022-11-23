from setuptools import setup, find_packages    

setup(
    name="evalign",
    version="0.1",
    url="",

    author="Dirk Van Compernolle",
    author_email="compi@esat.kuleuven.be",

    description="Sequence Matching Utilities for ALIGNning and ASR EVAluation",
    license = "free",
    
    packages = find_packages(),

    # a dictionary refering to required data not in .py files    
    # include_package_data=True  
    package_data = {'evalign':['data/*']},
    
    install_requires=[
        'pandas >= 1.3, <1.8 ',
    ],
    
    python_requires='>=3.7',
    
    classifiers=['Development Status: Functional, Beta',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9'],
                 


)
