from setuptools import find_packages,setup 
from future_typing import List

HYPHEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    This function will return  list of reqiurments
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements= file_obj.readlines()
        [req.replace('\n',' ')for req in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements


setup(
name='Mohammad Talha',
version='0.0.1',
author='Talha',
author_email='talhamohammad766@gmail.com',
packages=find_packages(),
install_reqires=get_requirements('requirements.txt'),
)