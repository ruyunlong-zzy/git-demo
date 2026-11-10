from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
setup(**generate_distutils_setup(packages=['humanoid_navigation'], package_dir={'': 'src'}))
