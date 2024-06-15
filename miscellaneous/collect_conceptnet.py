"""Collect data from ConceptNet."""

import json
import os

import requests
import yaml
from tqdm import tqdm


class DataCollector:
    """Data (conceptnet) collector class."""

    def __init__(
        self,
        relation: str,
        conceptnet_data_path: str,
        conceptnet_data_refresh: bool,
        semantic_knowledge_path: str,
        api_url: str,
    ):
        """Data (conceptnet) collector class.

        Args:
            relation: See https://github.com/commonsense/conceptnet5/wiki/Relations for
                all relations.
            conceptnet_data_path: Where to save raw queried conceptnet data path
            conceptnet_data_refresh: Whether to download the conceptnet data again or not.
            semantic_knowledge_path: Where to save pre-trained semantic (factual) knowledge
            api_url: e.g., http://api.conceptnet.io/, http://127.0.0.1:8084/, etc.

        """
        self.relation = relation
        self.relation_simple = self.relation.split("/")[-1]

        self.conceptnet_data_path = conceptnet_data_path
        self.conceptnet_data_refresh = conceptnet_data_refresh
        self.semantic_knowledge_path = semantic_knowledge_path
        self.weighting_mode = "highest"
        self.api_url = api_url

        self.read_mscoco()
        os.makedirs("./room_env/data", exist_ok=True)

    def read_mscoco(self, path: str = "./room_env/data/ms-coco-80-categories") -> None:
        """Return ms coco 80 object categories.

        Args:
            path: The path to the mscoco object category list.

        """
        with open(path, "r") as stream:
            self.mscoco = stream.readlines()
        self.mscoco = [line.strip() for line in self.mscoco]
        self.mscoco = ["_".join(foo.split()) for foo in self.mscoco]

    def get_from_conceptnet(self) -> None:
        """Get data from ConceptNet API by HTTP get query."""

        if self.conceptnet_data_refresh:
            self.raw_data = {}
            for object_category in tqdm(self.mscoco):
                query = (
                    f"{self.api_url}"
                    f"query?start=/c/en/{object_category}&rel={self.relation}"
                )
                response = requests.get(query).json()

                if len(response["edges"]) == 0:
                    continue

                self.raw_data[object_category] = []

                for edge in tqdm(response["edges"]):
                    self.raw_data[object_category].append(
                        {
                            "start": edge["start"],
                            "end": edge["end"],
                            "weight": edge["weight"],
                            "surfaceText": edge["surfaceText"],
                        }
                    )

            with open(self.conceptnet_data_path, "w") as stream:
                json.dump(self.raw_data, stream, indent=4, sort_keys=False)
        else:
            with open(self.conceptnet_data_path, "r") as stream:
                self.raw_data = json.load(stream)

        self.semantic_knowledge = {}

        for key, val in self.raw_data.items():
            head = key

            self.semantic_knowledge[head] = {self.relation_simple: []}

            if self.weighting_mode == "highest" and len(val) > 0:
                tail = sorted(val, key=lambda x: x["weight"], reverse=True)[0]
                tail = tail["end"]["@id"].split("/")[-1]
                self.semantic_knowledge[head][self.relation_simple].append(
                    {"tail": tail, "weight": 1}
                )

            else:
                for val_ in val:
                    tail = val_["end"]["@id"].split("/")[-1]

                    weight = 1 if self.weighting_mode is None else round(val_["weight"])

                    self.semantic_knowledge[head][self.relation_simple].append(
                        {"tail": tail, "weight": weight}
                    )

        with open(self.semantic_knowledge_path, "w") as stream:
            json.dump(self.semantic_knowledge, stream, indent=4, sort_keys=False)


def main(**kwargs) -> None:
    """Collect data. See ./collect_conceptnet.yaml for the config."""
    dc = DataCollector(**kwargs)
    dc.get_from_conceptnet()


if __name__ == "__main__":
    with open("./collect_conceptnet.yaml", "r") as stream:
        config = yaml.safe_load(stream)
    print("Arguments:")
    for k, v in config.items():
        print(f"  {k:>21} : {v}")

    main(**config)
