from setuptools import setup
from glob import glob

package_name = 'plant'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'environment.yaml', 'environment_map.yaml']),
        ('share/' + package_name + '/xml', glob('xml/*.*')),
        ('share/' + package_name + '/rviz', glob('rviz/*')),
        ('share/' + package_name + '/xml/mesh', glob('xml/mesh/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Ryung',
    maintainer_email='tlsgksfbd@seoultech.ac.kr',
    description='MuJoCo multirotor plant loader',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'plant = plant.plant:main',
            'multirotor_viewer = plant.multirotor_viewer:main',
            'map_edt = plant.map_edt:main',
            'map_camera = plant.map_camera:main',
            'slam_state = plant.slam_state:main',
            'kick_planning = plant.kick_planning:main',
        ],
    },
)
