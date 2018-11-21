# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from caffe2.python import core, workspace, caffe2_pb2
import numpy as np


def run_add5_and_add5gradient_op(device):
    # clear the workspace before running the operator
    workspace.ResetWorkspace()
    add5 = core.CreateOperator("Add5",
                               ["X"],
                               ["Y"],
                               device_option=device)
    print("==> Running Add5 op:")
    workspace.FeedBlob("X", (np.random.rand(5, 5)), device_option=device)
    print("Input of Add5: ", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(add5)
    print("Output of Add5: ", workspace.FetchBlob("Y"))

    print("\n\n==> Running Add5Gradient op:")
    print("Input of Add5Gradient: ", workspace.FetchBlob("Y"))
    add5gradient = core.CreateOperator("Add5Gradient",
                                       ["Y"],
                                       ["Z"],
                                       device_option=device)
    workspace.RunOperatorOnce(add5gradient)
    print("Output of Add5Gradient: ", workspace.FetchBlob("Z"))


def main():
    # try device_type=caffe2_pb2.CUDA if CUDA is available in our build
    device = caffe2_pb2.DeviceOption(device_type=caffe2_pb2.CPU)
    run_add5_and_add5gradient_op(device)


if __name__ == "__main__":
    main()
