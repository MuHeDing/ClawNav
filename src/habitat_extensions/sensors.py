from typing import Any, Dict

from gym import spaces
from habitat.core.registry import registry
from habitat.core.simulator import Observations, Sensor, SensorTypes

from habitat_extensions.task import VLNExtendedEpisode


@registry.register_sensor(name="RxRInstructionSensor")
class RxRInstructionSensor(Sensor):
    cls_uuid: str = "instruction"

    def __init__(self, **kwargs: Any):
        self.uuid = self.cls_uuid
        self.observation_space = spaces.Dict()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.MEASUREMENT

    def get_observation(
        self,
        observations: Dict[str, Observations],
        episode: VLNExtendedEpisode,
        **kwargs: Any,
    ):
        return {
            "text": episode.instruction.instruction_text,
            "tokens": episode.instruction.instruction_tokens,
            "trajectory_id": episode.trajectory_id,
        }
