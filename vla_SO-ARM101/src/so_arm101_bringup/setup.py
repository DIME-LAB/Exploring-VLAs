from setuptools import find_packages, setup
from glob import glob

package_name = 'so_arm101_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aldrin Inbaraj',
    maintainer_email='aldrininbarajjunk@gmail.com',
    description='Real-hardware bringup for SO-ARM101 — USB camera publishers and launch files',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'camera_publisher = so_arm101_bringup.camera_publisher:main',
        ],
    },
)
