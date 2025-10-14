from setuptools import find_packages, setup

# Required dependencies
required = [
    # Please keep alphabetized
    'gymnasium>=1.1.1',
    'mujoco>=3.2.3',
    'numpy>=1.21.0',
    'pyyaml>=6.0',
]

setup(
    name='dphand_env',
    version='0.1.0',
    description='DPHand: A robotic hand environment for diffusion policy learning',
    author='Yu Hexi',
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
    python_requires='>=3.8'
)
