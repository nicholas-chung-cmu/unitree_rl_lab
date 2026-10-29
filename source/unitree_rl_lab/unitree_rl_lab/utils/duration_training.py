"""Fixed duration weights with persistent stock velocity curriculum on resume."""
from .march_training import MarchRunner


class DurationRunner(MarchRunner):
    def phase_mode(self):
        return 'duration_v1'

    def save(self,path,infos=None):
        infos=dict(infos or {})
        ranges=self.env.unwrapped.command_manager.get_term('base_velocity').cfg.ranges
        infos['duration_ranges']={k:getattr(ranges,k) for k in ('lin_vel_x','lin_vel_y','ang_vel_z')}
        super().save(path,infos)

    def load(self,path,load_optimizer=True,map_location=None):
        infos=super().load(path,load_optimizer,map_location)
        ranges=self.env.unwrapped.command_manager.get_term('base_velocity').cfg.ranges
        for k,v in infos['duration_ranges'].items():
            setattr(ranges,k,v)
        return infos


class DurationStrongRunner(DurationRunner):
    def phase_mode(self):
        return 'duration_strong_v1'


class DurationAverageRunner(DurationRunner):
    def phase_mode(self):
        return 'duration_average_v1'
