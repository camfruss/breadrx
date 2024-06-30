# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
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

import datasets


_DESCRIPTION = """
The bread_proofing dataset. All crumb images are gathered from Reddit's 
/r/Breadit and /r/Sourdough and post-processed to be 512x512. 
"""

_HOMEPAGE = "https://github.com/camfruss/breadrx"

_URLS = {
    "train": "data/test-00000-of-00001.parquet",
    "validation": "data/valid-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet"
}


class BreadProofingDataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.1")

    def _info(self):
        features = datasets.Features(
            {
                "file_name": datasets.Image(),
                "date": datasets.Value("string"),
                "post_id": datasets.Value("string"),
                "upvotes": datasets.Value("int32"),
                "over_proof": datasets.Value("float16"),
                "under_proof": datasets.Value("float16"),
                "perfect_proof": datasets.Value("float16"),
                "unsure_proof": datasets.Value("float16")
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            features=features
        )

    def _split_generators(self, dl_manager):
        """ Returns SplitGenerators """
        path = dl_manager.download(_URLS)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "images": [dl_manager.iter_archive(image) for image in path["train"]],
                    "split": "train"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "images": [dl_manager.iter_archive(image) for image in path["validate"]],
                    "split": "validate"
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "images": [dl_manager.iter_archive(image) for image in path["test"]],
                    "split": "test"
                }
            )
        ]

    def _generate_examples(self, images, split):
        """ Yields Examples """
        for key, (path, image) in enumerate(images):
            yield key, {
                "image": {
                    "path": path,
                    "bytes": image.read(),
                },
                # "post_id":
                # "date": ,
                # "upvotes": ,
                # "post_link": ,
                # "image_link": ,
                # "over_proof": ,
                # "under_proof": ,
                # "perfect_proof": ,
                # "unsure_proof":
            }
