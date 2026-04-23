from setuptools import find_packages, setup
from glob import glob

package_name = 'sim_ground_truth'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Aldrin Inbaraj',
    maintainer_email='aldrininbarajjunk@gmail.com',
    description='Gazebo sim ground-truth pose + bbox publisher mirroring aruco_camera_localizer real-side contract',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ground_truth_publisher = sim_ground_truth.ground_truth_publisher:main',
        ],
    },
)
