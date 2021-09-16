from setuptools import setup

setup(
    name='fantasy_maps',
    version='0.1.0',    
    description='A processor of fantasy maps, powered by Vertex AI',
    url='https://github.com/telpirion/FantasyMaps',
    author='Eric Schmidt',
    author_email='erschmid@google.com',
    license='MIT',
    packages=['fantasy_maps'],
    install_requires=['google-cloud-aiplatform==1.3.0',
                      'google-cloud-storage',
                      'imgaug', 
                      'jsonlines',
                      'pillow',
                      'pytest'                   
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Other Audience',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3.6',
    ],
)