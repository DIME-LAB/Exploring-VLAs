import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'so_arm101_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join(package_name, '*.yaml'))),
    ],
    package_data={package_name: ['*.yaml']},
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='LycheeAI',
    maintainer_email='contact@lycheeai-hub.com',
    description='GUI control, servo driver, and EE pose publisher for SO-ARM101',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'control_gui = so_arm101_control.control_gui:main',
            'servo_driver = so_arm101_control.servo_driver:main',
            'ee_pose_publisher = so_arm101_control.ee_pose_publisher:main',
            'test_ik_solvers = so_arm101_control.test_ik_solvers:main',
            'test_planning = so_arm101_control.test_planning:main',
            'test_debug_services = so_arm101_control.test_debug_services:main',
            'compute_workspace = so_arm101_control.compute_workspace:main',
        ],
    },
)
