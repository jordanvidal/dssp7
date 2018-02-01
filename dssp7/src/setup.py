from setuptools import setup, find_packages

version = '0.0.1'

setup(name='dssp7',
      version=version,
      description="dssp7et de fin d'annee",
      url='http://datalabpf.group.echonet:8080/',
      packages=find_packages(),
      test_suite='tests',
      include_package_data=True,
      scripts=['scripts/dssp7_run_fit.py', 'scripts/dssp7_run_predict.py'],
      install_requires=["numpy==1.13.3",
                        "pandas==0.20.3",
                        "python-dateutil==2.6.1",
                        "pytz==2017.3",
                        "scikit-learn==0.19.1",
                        "six==1.11.0",
                        "scipy==1.0.0"
                        ],
      zip_safe=False)
