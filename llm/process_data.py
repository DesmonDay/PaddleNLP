# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json

out = []
with open("test_1009.json") as f:
    result = json.load(f)
    for re in result:
        src = re["instruction"]
        tgt = re["output"]
        out.append({"src": src, "tgt": tgt})

with open("test.json", "w") as f:
    json.dump(out, f, indent=4, ensure_ascii=False)
