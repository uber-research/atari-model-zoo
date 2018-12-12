# Copyright (c) 2018 Uber Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import click
import json
import numpy as np
from process_helper import assemble_data, reduce_dim, disassemble, download_data


@click.command()
@click.argument('path_to_config', nargs=1)
@click.option('--download', is_flag=True)
def main(path_to_config, download):
    """
    PATH_TO_CONFIG: Path to config.json
    """
    print("loading", path_to_config)

    with open(path_to_config) as f:
        config = json.load(f)

    print(config['data'])
    print(config['method'])
    print(config['data']['key'])

    if download:
        download_data(config['data'])

    Xs, npz_files, npz_dims, _, _, _ = assemble_data(config['data'], dict_key=config['data']['key'])

    X = np.vstack(Xs)

    print(npz_files)
    print(npz_dims)
    print(X.shape)

    X_r = reduce_dim(X, config['method'])

    #print(X_r.shape)

    disassemble(X_r, npz_files, npz_dims, config['method']['name'], dict_key=config['data']['key'])

if __name__ == '__main__':
    main()

