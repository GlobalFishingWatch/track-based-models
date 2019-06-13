# Copyright 2019 Global Fishing Watch.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import setuptools

DEPENDENCIES = [
    "keras",
    "numpy",
    "pandas",
    # "tensorflow",
    "python-dateutil",
    "google-api-python-client"
]


setuptools.setup(
    name='track_based_models',
    version='0.0.1',
    author='Tim Hochberg',
    author_email='tim@globalfishingwatch.com',
    package_data={},
    packages=['track_based_models'],
    install_requires=DEPENDENCIES 
)

