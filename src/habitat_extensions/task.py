import gzip
import json
import os
from typing import Dict, List, Optional, Union

import attr
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.datasets.pointnav.pointnav_dataset import ALL_SCENES_MASK
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import VLNEpisode


DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"
ALL_LANGUAGES_MASK = "*"
ALL_ROLES_MASK = "*"


@attr.s(auto_attribs=True)
class ExtendedInstructionData:
    instruction_text: str = attr.ib(default=None)
    instruction_id: Optional[str] = attr.ib(default=None)
    language: Optional[str] = attr.ib(default=None)
    annotator_id: Optional[str] = attr.ib(default=None)
    edit_distance: Optional[float] = attr.ib(default=None)
    timed_instruction: Optional[List[Dict[str, Union[float, str]]]] = attr.ib(
        default=None
    )
    instruction_tokens: Optional[List[str]] = attr.ib(default=None)
    split: Optional[str] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class VLNExtendedEpisode(VLNEpisode):
    goals: Optional[List[NavigationGoal]] = attr.ib(default=None)
    reference_path: Optional[List[List[float]]] = attr.ib(default=None)
    instruction: ExtendedInstructionData = attr.ib(default=None)
    trajectory_id: Optional[Union[int, str]] = attr.ib(default=None)


@registry.register_dataset(name="RxR-VLN-CE-v1")
class RxRVLNCEDatasetV1(Dataset):
    """Loads the RxR VLN-CE dataset in Habitat format."""

    episodes: List[VLNEpisode]
    annotation_roles: List[str] = ["guide", "follower"]
    languages: List[str] = ["en-US", "en-IN", "hi-IN", "te-IN"]

    @staticmethod
    def _scene_from_episode(episode: VLNEpisode) -> str:
        return os.path.splitext(os.path.basename(episode.scene_id))[0]

    @staticmethod
    def _language_from_episode(episode: VLNExtendedEpisode) -> Optional[str]:
        return episode.instruction.language

    @classmethod
    def get_scenes_to_load(cls, config) -> List[str]:
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        return sorted(
            {cls._scene_from_episode(episode) for episode in dataset.episodes}
        )

    @classmethod
    def extract_roles_from_config(cls, config) -> List[str]:
        roles = getattr(config, "roles", ["guide"])
        if ALL_ROLES_MASK in roles:
            return cls.annotation_roles
        return roles

    @classmethod
    def extract_languages_from_config(cls, config) -> List[str]:
        languages = getattr(config, "languages", [ALL_LANGUAGES_MASK])
        if ALL_LANGUAGES_MASK in languages:
            return cls.languages
        return languages

    @classmethod
    def _format_data_path(cls, config, role: str) -> str:
        split = getattr(config, "split", "train")
        data_path = getattr(config, "data_path")
        return data_path.format(split=split, role=role)

    @classmethod
    def check_config_paths_exist(cls, config) -> bool:
        roles = cls.extract_roles_from_config(config)
        return all(
            os.path.exists(cls._format_data_path(config, role))
            for role in roles
        ) and os.path.exists(config.scenes_dir)

    def __init__(self, config: Optional[object] = None) -> None:
        self.episodes = []
        self.config = config

        if config is None:
            return

        for role in self.extract_roles_from_config(config):
            with gzip.open(
                self._format_data_path(config, role), "rt"
            ) as f:
                self.from_json(
                    f.read(),
                    scenes_dir=config.scenes_dir,
                    split=getattr(config, "split", "train"),
                )

        content_scenes = getattr(config, "content_scenes", [ALL_SCENES_MASK])
        if ALL_SCENES_MASK not in content_scenes:
            scenes_to_load = set(content_scenes)
            self.episodes = [
                episode
                for episode in self.episodes
                if self._scene_from_episode(episode) in scenes_to_load
            ]

        languages_to_load = self.extract_languages_from_config(config)
        if ALL_LANGUAGES_MASK not in languages_to_load:
            languages_set = set(languages_to_load)
            self.episodes = [
                episode
                for episode in self.episodes
                if self._language_from_episode(episode) in languages_set
            ]

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None, split: str = ""
    ) -> None:
        deserialized = json.loads(json_str)

        for episode in deserialized["episodes"]:
            episode = VLNExtendedEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]
                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = ExtendedInstructionData(
                **episode.instruction
            )
            episode.instruction.split = split
            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)
